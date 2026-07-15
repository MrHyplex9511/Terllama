/*
 * main.cpp — Terllama: ternary LLM inference
 *
 * Usage:
 *   ./terllama "Your prompt here" [max_tokens=40] [temperature=0.7]
 *
 * Auto-detects CPU arch, selects kernel at runtime.
 * Run with TERLLAMA_ARCH=scalar|sse42|avx|avx2|avx512|neon to override.
 */
#include "model.h"
#include "loader.h"
#include "kernel_decl.h"
#include "inference.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER (Python helper bridge)
// ═══════════════════════════════════════════════════════════════════════════
static std::vector<int> tokenize_with_python(const std::string& prompt,
                                              const std::string& helper_dir) {
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        pf << prompt;
    }
    std::string cmd = "python3 " + helper_dir + "/tokenize_helper.py";
    int ret = system(cmd.c_str());
    if (ret != 0) { std::cerr << "Tokenization failed\n"; exit(1); }

    std::vector<int> tokens;
    std::ifstream tf(token_file);
    int tid;
    while (tf >> tid) tokens.push_back(tid);
    return tokens;
}

static std::string decode_with_python(const std::vector<int>& tokens,
                                       const std::string& helper_dir) {
    std::string token_file = "/tmp/ternary_decode_in.txt";
    std::string out_file = "/tmp/ternary_decode_out.txt";
    {
        std::ofstream tf(token_file);
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) tf << " ";
            tf << tokens[i];
        }
    }
    std::string cmd = "python3 " + helper_dir + "/decode_helper.py";
    int ret = system(cmd.c_str());
    if (ret != 0) { std::cerr << "Decoding failed\n"; return "?"; }

    std::ifstream of(out_file);
    std::stringstream ss;
    ss << of.rdbuf();
    return ss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " \"prompt\" [max_tokens=40] [temp=0.7]" << std::endl;
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = (argc > 2) ? std::stoi(argv[2]) : 40;
    float temperature = (argc > 3) ? std::stof(argv[3]) : 0.7f;

    // Model/helper dir (same dir as executable, or env override)
    std::string model_dir = std::getenv("TERLLAMA_MODEL_DIR")
        ? std::string(std::getenv("TERLLAMA_MODEL_DIR"))
        : "/media/extra/Symlinks/BitNet/utils/export";
    std::string helper_dir = model_dir;

    srand(42);

    // ─── Detect CPU architecture ─────────────────────────────────────────
    CPUArch arch = detect_cpu_arch();

    // Allow environment override for testing
    const char* arch_override = std::getenv("TERLLAMA_ARCH");
    if (arch_override) {
        std::string ao(arch_override);
        if (ao == "scalar")  arch = CPUArch::X86_64_SCALAR;
        else if (ao == "sse42")  arch = CPUArch::X86_64_SSE42;
        else if (ao == "avx")    arch = CPUArch::X86_64_AVX;
        else if (ao == "avx2")   arch = CPUArch::X86_64_AVX2;
        else if (ao == "avx512") arch = CPUArch::X86_64_AVX512;
        else if (ao == "neon")   arch = CPUArch::ARM64_NEON;
    }

    std::cout << "Terllama — CPU: " << cpu_arch_name(arch)
              << "  |  Model: " << model_dir << std::endl;

    // ─── Load everything ────────────────────────────────────────────────
    std::cout << "Loading config..." << std::endl;
    auto cfg = load_config(model_dir + "/model_extra.bin");

    std::cout << "Loading embedding (" << cfg.vocab_size << "\xC3\x97"
              << cfg.hidden_size << ")..." << std::endl;
    auto embedding = load_embedding(model_dir + "/model_extra.bin", cfg);

    std::cout << "Loading " << cfg.num_hidden_layers << " layer norms..." << std::endl;
    auto layer_norms = load_layer_norms(model_dir + "/model_extra.bin", cfg);

    std::cout << "Loading final norm..." << std::endl;
    auto final_norm = load_final_norm(model_dir + "/model_extra.bin", cfg);

    // Load decomposed linear layers. Prefer I2_S format.
    std::string i2s_path = model_dir + "/model_decomposed_i2s.bin";
    std::string old_path = model_dir + "/model_decomposed.bin";
    std::vector<LayerData> layers;

    {
        std::ifstream test_i2s(i2s_path);
        if (test_i2s.good()) {
            std::cout << "Using I2_S format model..." << std::endl;
            layers = load_decomposed_layers_i2s(i2s_path);
        } else {
            std::cout << "Loading decomposed linear layers (ALS format)..." << std::endl;
            layers = load_decomposed_layers(old_path);
        }
    }
    std::cout << "  Loaded " << layers.size() << " layers." << std::endl;

    std::cout << "Building RoPE cache..." << std::endl;
    auto rope = build_rope_cache(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);

    // ─── Tokenize ───────────────────────────────────────────────────────
    std::cout << "\nTokenizing prompt..." << std::endl;
    auto prompt_tokens = tokenize_with_python(prompt, helper_dir);
    std::cout << "Prompt tokens (" << prompt_tokens.size() << "): ";
    for (int t : prompt_tokens) std::cout << t << " ";
    std::cout << std::endl;

    // ─── Generation ────────────────────────────────────────────────────
    std::cout << "\n=== Generating (ternary weights, " << cpu_arch_name(arch) << ") ===" << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;

    auto [output_tokens, total_ms] = generate(
        prompt_tokens, max_tokens, temperature,
        cfg, embedding, layers, final_norm, layer_norms, rope);

    // ─── Decode ─────────────────────────────────────────────────────────
    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = decode_with_python(all_tokens, helper_dir);
    std::string prompt_decoded = decode_with_python(prompt_tokens, helper_dir);

    // ─── Report ─────────────────────────────────────────────────────────
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Full response:" << std::endl;
    std::cout << decoded << std::endl;

    std::string generated_only;
    if (decoded.size() > prompt_decoded.size()) {
        generated_only = decoded.substr(prompt_decoded.size());
        while (!generated_only.empty() && generated_only[0] == ' ')
            generated_only.erase(0, 1);
    } else {
        size_t pos = decoded.find(prompt_decoded);
        if (pos != std::string::npos)
            generated_only = decoded.substr(pos + prompt_decoded.size());
    }
    std::cout << "\nGenerated text: \"" << generated_only << "\"" << std::endl;

    double total_tokens = (double)(prompt_tokens.size() + output_tokens.size());
    std::cout << "\n── Performance ──" << std::endl;
    std::cout << "  Total time:       " << total_ms << " ms" << std::endl;
    std::cout << "  Prompt tokens:    " << prompt_tokens.size() << std::endl;
    std::cout << "  Generated tokens: " << output_tokens.size() << std::endl;
    std::cout << "  Total tokens:     " << (int)total_tokens << std::endl;
    std::cout << "  ms/token:         " << (total_ms / total_tokens) << std::endl;
    std::cout << "  tokens/sec:       " << (1000.0 * total_tokens / total_ms) << std::endl;
    std::cout << "  Kernel:           " << cpu_arch_name(arch) << std::endl;

    return 0;
}

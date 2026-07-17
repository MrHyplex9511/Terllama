/*
 * commands.cpp — CLI subcommand implementations for Terllama
 *
 * All cmd_* functions migrated from main.cpp.
 * Tokenizer helper uses posix_spawn (no shell) + /proc/self/exe for
 * script directory resolution.
 */
#include "cli/commands.h"
#include "model.h"
#include "loader.h"
#include "kernel_decl.h"
#include "inference.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <pwd.h>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

// Resolve scripts/ directory relative to the running binary via
// /proc/self/exe, not CWD.
static std::string get_helper_dir() {
    char buf[4096];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len <= 0) return "scripts"; // fallback
    buf[len] = '\0';
    std::string exe(buf);
    auto pos = exe.rfind('/');
    if (pos == std::string::npos) return "scripts";
    std::string dir = exe.substr(0, pos) + "/scripts";
    struct stat st;
    if (stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        return dir;
    return "scripts"; // fallback
}

// Run a Python helper script via posix_spawn (no shell).
// Returns the exit code, or -1 on spawn failure.
static int run_python_script(const std::string& script_path) {
    pid_t pid;
    std::string python = "python3";
    const char* argv[] = {"python3", script_path.c_str(), nullptr};

    int ret = posix_spawnp(&pid, python.c_str(), nullptr, nullptr,
                           const_cast<char* const*>(argv), environ);
    if (ret != 0) {
        std::cerr << "Failed to spawn python3 for " << script_path << std::endl;
        return -1;
    }

    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER (Python helper bridge, no shell)
// ═══════════════════════════════════════════════════════════════════════════

static std::vector<int> tokenize_with_python(const std::string& prompt,
                                              const std::string& helper_dir) {
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        pf << prompt;
    }
    int ret = run_python_script(helper_dir + "/tokenize_helper.py");
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
    int ret = run_python_script(helper_dir + "/decode_helper.py");
    if (ret != 0) { std::cerr << "Decoding failed\n"; return "?"; }

    std::ifstream of(out_file);
    std::stringstream ss;
    ss << of.rdbuf();
    return ss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: list
// ═══════════════════════════════════════════════════════════════════════════

int cmd_list() {
    std::string dir = models_dir();
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        std::cout << "No models found (~/.terllama/models/ does not exist)" << std::endl;
        return 0;
    }

    std::ifstream mj(models_json_path());
    if (!mj) {
        std::cout << "No models installed." << std::endl;
        return 0;
    }

    std::cout << "Installed models:" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    std::string content((std::istreambuf_iterator<char>(mj)),
                         std::istreambuf_iterator<char>());

    // Parse JSON array of model entries
    size_t pos = 0;
    int count = 0;
    while (true) {
        auto start = content.find("{\"id\"", pos);
        if (start == std::string::npos) break;
        auto end = content.find("}", start);
        if (end == std::string::npos) break;
        std::string block = content.substr(start, end - start + 1);

        auto extract = [&](const std::string& key) -> std::string {
            std::string search = "\"" + key + "\":\"";
            auto p = block.find(search);
            if (p == std::string::npos) return "";
            p += search.size();
            auto q = block.find("\"", p);
            if (q == std::string::npos) return "";
            return block.substr(p, q - p);
        };
        auto extract_num = [&](const std::string& key) -> long long {
            std::string search = "\"" + key + "\":";
            auto p = block.find(search);
            if (p == std::string::npos) return 0;
            p += search.size();
            auto q = block.find_first_of(",}", p);
            if (q == std::string::npos) return 0;
            return std::stoll(block.substr(p, q - p));
        };

        std::string id     = extract("id");
        std::string fmt    = extract("format");
        long long sz       = extract_num("size");
        std::string ts     = extract("downloaded");

        if (!id.empty()) {
            std::cout << "  " << id
                      << " (" << fmt << ") — " << fmt_size((double)sz);
            if (!ts.empty()) std::cout << "  [" << ts << "]";
            std::cout << std::endl;
            count++;
        }
        pos = end + 1;
    }

    if (count == 0) {
        std::cout << "  (no models found)" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Models directory: " << dir << std::endl;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: show
// ═══════════════════════════════════════════════════════════════════════════

int cmd_show(const std::string& model_id) {
    std::string model_dir = std::getenv("TERLLAMA_MODEL_DIR")
        ? std::string(std::getenv("TERLLAMA_MODEL_DIR"))
        : models_dir() + "/" + model_id;

    std::string extra_path = model_dir + "/model_extra.bin";
    std::string i2s_path  = model_dir + "/model_decomposed_i2s.bin";
    std::string als_path  = model_dir + "/model_decomposed.bin";

    struct stat st;
    std::cout << "Model: " << model_id << std::endl;
    std::cout << "Path:  " << model_dir << std::endl;

    if (stat(extra_path.c_str(), &st) == 0) {
        std::cout << "  Config + embedding: " << fmt_size((double)st.st_size) << std::endl;
    } else {
        std::cout << "  Config: not found" << std::endl;
    }

    if (stat(i2s_path.c_str(), &st) == 0) {
        std::cout << "  Weights (I2_S):     " << fmt_size((double)st.st_size) << std::endl;
    } else if (stat(als_path.c_str(), &st) == 0) {
        std::cout << "  Weights (ALS):      " << fmt_size((double)st.st_size) << std::endl;
    } else {
        std::cout << "  Weights: not found" << std::endl;
    }

    try {
        auto cfg = load_config(extra_path);
        std::cout << "\n  Architecture:" << std::endl;
        std::cout << "    Parameters:       ~" << (cfg.vocab_size * cfg.hidden_size / 1000000) << "M" << std::endl;
        std::cout << "    Hidden size:      " << cfg.hidden_size << std::endl;
        std::cout << "    Layers:           " << cfg.num_hidden_layers << std::endl;
        std::cout << "    Attention heads:  " << cfg.num_attention_heads << std::endl;
        std::cout << "    KV heads:         " << cfg.num_key_value_heads << std::endl;
        std::cout << "    Head dim:         " << cfg.head_dim << std::endl;
        std::cout << "    Vocab size:       " << cfg.vocab_size << std::endl;
        std::cout << "    Max seq len:      " << cfg.max_position_embeddings << std::endl;
        std::cout << "    RMS norm eps:     " << cfg.rms_norm_eps << std::endl;
        std::cout << "    RoPE theta:       " << cfg.rope_theta << std::endl;
    } catch (...) {
        std::cout << "  (could not read config)" << std::endl;
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: rm
// ═══════════════════════════════════════════════════════════════════════════

int cmd_rm(const std::string& model_id) {
    std::string model_dir = models_dir() + "/" + model_id;
    struct stat st;

    if (stat(model_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        std::cerr << "Model not found: " << model_id << std::endl;
        return 1;
    }

    auto rm_file = [](const std::string& path) {
        if (unlink(path.c_str()) == 0)
            std::cout << "  Removed: " << path << std::endl;
    };

    rm_file(model_dir + "/model_decomposed_i2s.bin");
    rm_file(model_dir + "/model_decomposed.bin");
    rm_file(model_dir + "/model_extra.bin");

    rmdir(model_dir.c_str());

    std::string jpath = models_json_path();
    std::ifstream inf(jpath);
    if (inf) {
        std::string content((std::istreambuf_iterator<char>(inf)),
                             std::istreambuf_iterator<char>());
        inf.close();

        std::ofstream of(jpath);
        if (of) {
            std::string result;
            size_t pos = 0;
            bool first = true;
            result = "{\n  \"models\": [\n";
            while (true) {
                auto start = content.find("{\"id\"", pos);
                if (start == std::string::npos) break;
                auto end = content.find("}", start);
                if (end == std::string::npos) break;
                std::string block = content.substr(start, end - start + 1);

                auto p = block.find("\"id\":\"");
                if (p != std::string::npos) {
                    p += 6;
                    auto q = block.find("\"", p);
                    if (q != std::string::npos) {
                        std::string existing_id = block.substr(p, q - p);
                        if (existing_id != model_id) {
                            if (!first) result += ",";
                            first = false;
                            result += block;
                        }
                    }
                }
                pos = end + 1;
            }
            result += "\n  ]\n}\n";
            of << result;
        }
    }

    std::cout << "Model removed: " << model_id << std::endl;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: chat  (internal: non-interactive)
// ═══════════════════════════════════════════════════════════════════════════

static int cmd_chat_simple(const std::string& model_id,
                            const std::string& prompt_text,
                            int max_tokens, float temperature) {
    std::string model_dir = std::getenv("TERLLAMA_MODEL_DIR")
        ? std::string(std::getenv("TERLLAMA_MODEL_DIR"))
        : models_dir() + "/" + model_id;

    std::string helper_dir = get_helper_dir();

    srand(42);
    CPUArch arch = detect_cpu_arch();
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

    std::cout << "Terllama Chat — CPU: " << cpu_arch_name(arch)
              << "  |  Model: " << model_id << std::endl;
    std::cout << std::string(50, '=') << std::endl;

    std::cout << "Loading model..." << std::endl;
    auto loaded = load_model_from(model_dir);
    auto& cfg = loaded.cfg;
    auto& embedding = loaded.embedding;
    auto& layer_norms = loaded.layer_norms;
    auto& final_norm = loaded.final_norm;
    auto& layers = loaded.layers;
    std::cout << "  Loaded " << layers.size() << " layers." << std::endl;

    auto rope = build_rope_cache(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);

    std::cout << "\nPrompt: " << prompt_text << std::endl;
    auto prompt_tokens = tokenize_with_python(prompt_text, helper_dir);

    if (prompt_tokens.empty()) {
        std::cerr << "Tokenization failed" << std::endl;
        return 1;
    }

    auto [output_tokens, total_ms] = generate(
        prompt_tokens, max_tokens, temperature,
        cfg, embedding, layers, final_norm, layer_norms, rope);

    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = decode_with_python(all_tokens, helper_dir);
    std::string prompt_decoded = decode_with_python(prompt_tokens, helper_dir);

    std::cout << "\n=== Response ===" << std::endl;
    std::cout << decoded << std::endl;

    double total_tokens = (double)(prompt_tokens.size() + output_tokens.size());
    std::cout << "\n── Performance ──" << std::endl;
    std::cout << "  Time:       " << total_ms << " ms" << std::endl;
    std::cout << "  Generated:  " << output_tokens.size() << " tokens" << std::endl;
    std::cout << "  Speed:      " << (1000.0 * total_tokens / total_ms) << " tok/s" << std::endl;
    std::cout << "  Kernel:     " << cpu_arch_name(arch) << std::endl;

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: chat (unified dispatcher — parses args, runs interactive or
//                  single-shot)
// ═══════════════════════════════════════════════════════════════════════════

int cmd_chat(int argc, char** argv) {
    std::string model_id;
    std::string prompt_text;
    int max_tokens = 256;
    float temperature = 0.7f;

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "--model" || arg == "-m") && i + 1 < argc)
            model_id = argv[++i];
        else if ((arg == "--prompt" || arg == "-p") && i + 1 < argc)
            prompt_text = argv[++i];
        else if ((arg == "--max-tokens" || arg == "-n") && i + 1 < argc)
            max_tokens = std::stoi(argv[++i]);
        else if ((arg == "--temp" || arg == "-t") && i + 1 < argc)
            temperature = std::stof(argv[++i]);
    }

    if (model_id.empty()) {
        std::cerr << "Usage: " << argv[0] << " chat --model <name> [--prompt \"text\"] [--max-tokens N] [--temp T]" << std::endl;
        return 1;
    }

    // Interactive mode — single-turn loop
    if (prompt_text.empty()) {
        std::cout << "Terllama Chat — " << model_id << std::endl;
        std::cout << "Type your messages. Ctrl+C or empty line to exit." << std::endl;
        std::cout << std::string(50, '-') << std::endl;

        std::string model_dir = std::getenv("TERLLAMA_MODEL_DIR")
            ? std::string(std::getenv("TERLLAMA_MODEL_DIR"))
            : models_dir() + "/" + model_id;
        std::string helper_dir = get_helper_dir();

        srand(42);
        auto loaded = load_model_from(model_dir);
        auto& cfg = loaded.cfg;
        auto& embedding = loaded.embedding;
        auto& layer_norms = loaded.layer_norms;
        auto& final_norm = loaded.final_norm;
        auto& layers = loaded.layers;
        auto rope = build_rope_cache(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);

        std::string line;
        while (true) {
            std::cout << "\nYou: ";
            std::cout.flush();
            if (!std::getline(std::cin, line) || line.empty()) break;

            auto tokens = tokenize_with_python(line, helper_dir);
            if (tokens.empty()) continue;

            auto [out_tokens, ms] = generate(
                tokens, max_tokens, temperature,
                cfg, embedding, layers, final_norm, layer_norms, rope);

            std::vector<int> all = tokens;
            all.insert(all.end(), out_tokens.begin(), out_tokens.end());
            std::string text = decode_with_python(all, helper_dir);

            std::cout << "AI:  " << text << std::endl;
        }
        std::cout << "\nBye!" << std::endl;
        return 0;
    }

    return cmd_chat_simple(model_id, prompt_text, max_tokens, temperature);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: pull (download alias)
// ═══════════════════════════════════════════════════════════════════════════

int cmd_pull(int argc, char** argv) {
    std::vector<const char*> args;
    args.push_back(argv[0]);
    args.push_back("download");
    for (int i = 2; i < argc; i++) {
        args.push_back(argv[i]);
    }

    int new_argc = (int)args.size();
    char** new_argv = new char*[new_argc + 1];
    for (int i = 0; i < new_argc; i++) {
        new_argv[i] = const_cast<char*>(args[i]);
    }
    new_argv[new_argc] = nullptr;

    int ret = downloader_main(new_argc, new_argv);
    delete[] new_argv;
    return ret;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: serve
// ═══════════════════════════════════════════════════════════════════════════

int cmd_serve(int argc, char** argv) {
    return server_main(argc, argv);
}

// ═══════════════════════════════════════════════════════════════════════════
// LEGACY MODE: terllama "prompt" [max_tokens] [temp]
// ═══════════════════════════════════════════════════════════════════════════

int cmd_legacy(const std::string& prompt, int max_tokens, float temperature) {
    std::string model_dir = std::getenv("TERLLAMA_MODEL_DIR")
        ? std::string(std::getenv("TERLLAMA_MODEL_DIR"))
        : ".";
    std::string helper_dir = get_helper_dir();

    srand(42);
    CPUArch arch = detect_cpu_arch();
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

    auto loaded = load_model_from(model_dir);
    auto& cfg = loaded.cfg;
    auto& embedding = loaded.embedding;
    auto& layer_norms = loaded.layer_norms;
    auto& final_norm = loaded.final_norm;
    auto& layers = loaded.layers;

    auto rope = build_rope_cache(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);

    std::cout << "\nTokenizing prompt..." << std::endl;
    auto prompt_tokens = tokenize_with_python(prompt, helper_dir);

    std::cout << "\n=== Generating ===" << std::endl;
    auto [output_tokens, total_ms] = generate(
        prompt_tokens, max_tokens, temperature,
        cfg, embedding, layers, final_norm, layer_norms, rope);

    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = decode_with_python(all_tokens, helper_dir);

    std::cout << "\n" << decoded << std::endl;

    double total_tokens_n = (double)(prompt_tokens.size() + output_tokens.size());
    std::cout << "\n── Performance ──" << std::endl;
    std::cout << "  Time:       " << total_ms << " ms" << std::endl;
    std::cout << "  Generated:  " << output_tokens.size() << " tokens" << std::endl;
    std::cout << "  Speed:      " << (1000.0 * total_tokens_n / total_ms) << " tok/s" << std::endl;

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// USAGE
// ═══════════════════════════════════════════════════════════════════════════

void print_usage(const char* prog) {
    std::cerr << "Terllama — Ternary LLM Inference Engine  (alpha)" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Usage:" << std::endl;
    std::cerr << "  " << prog << " \"prompt\" [max_tokens] [temp]    Run inference (legacy)" << std::endl;
    std::cerr << "  " << prog << " list                            List installed models" << std::endl;
    std::cerr << "  " << prog << " show <model>                    Show model info" << std::endl;
    std::cerr << "  " << prog << " pull <hf-repo> [--fmt i2s|als] Download model from HF" << std::endl;
    std::cerr << "  " << prog << " rm <model>                      Remove a model" << std::endl;
    std::cerr << "  " << prog << " serve [--port N]                Start API server" << std::endl;
    std::cerr << "  " << prog << " chat --model <m> [--prompt p]   CLI chat" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Environment:" << std::endl;
    std::cerr << "  TERLLAMA_MODEL_DIR   model file directory" << std::endl;
    std::cerr << "  TERLLAMA_PORT        server port (default 8375)" << std::endl;
    std::cerr << "  TERLLAMA_ARCH        override CPU arch" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Usage examples:" << std::endl;
    std::cerr << "  " << prog << " pull HuggingFaceTB/SmolLM2-135M --format i2s" << std::endl;
    std::cerr << "  " << prog << " list" << std::endl;
    std::cerr << "  " << prog << " serve --port 8375" << std::endl;
    std::cerr << "  " << prog << " \"Hello, world!\" 100 0.8" << std::endl;
}

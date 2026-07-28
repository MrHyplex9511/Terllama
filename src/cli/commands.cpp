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
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <pwd.h>
#include <spawn.h>
#include <sys/wait.h>

#include "core/tokenizer.h"
#include "core/gigatoken_wrapper.h"

extern char **environ;

// ═══════════════════════════════════════════════════════════════════════════
// SIGNAL HANDLING
// ═══════════════════════════════════════════════════════════════════════════

std::atomic<bool> g_interrupted{false};

extern "C" void handle_signal(int sig) {
    (void)sig;
    g_interrupted = true;
}

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

// Forward declarations
static void add_model_entry(const std::string& model_id,
                             const std::string& format,
                             int64_t size_bytes);

static std::string slugify(const std::string& repo) {
    std::string s = repo;
    for (auto& c : s) {
        if (c == '/') c = '-';
    }
    return s;
}

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

// ═══════════════════════════════════════════════════════════════════════════
// MODEL REGISTRY — shortname → HuggingFace repo resolution
// ═══════════════════════════════════════════════════════════════════════════

struct RegistryEntry {
    std::string hf_repo;
    std::string format;  // "i2s", "als", or "gguf"
    int64_t size_mb;
};

static std::unordered_map<std::string, RegistryEntry> get_registry() {
    return {
        {"tinyllama",    {"TinyLlama/TinyLlama-1.1B-Chat-v1.0", "als",  139}},
        {"smolLM2",      {"HuggingFaceTB/SmolLM2-135M",          "i2s",  54}},
        {"mistral-7b",   {"mistralai/Mistral-7B-v0.3",           "gguf", 4100}},
        {"llama-3.1-8b", {"meta-llama/Llama-3.1-8B",             "gguf", 4800}},
        {"phi-3.5-mini", {"microsoft/Phi-3.5-mini-instruct",     "gguf", 2600}},
        {"gemma-2b",     {"google/gemma-2b-it",                  "gguf", 1400}},
    };
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
        Logger::error(("Failed to spawn python3 for " + script_path).c_str());
        return -1;
    }

    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER (Python helper for encode only; native C++ decode)
// ═══════════════════════════════════════════════════════════════════════════

// Try GigaToken encode first; fall back to Python subprocess.
// model_dir should contain tokenizer.json for GigaToken to load.
static std::vector<int> tokenize_with_helper(const std::string& prompt,
                                              const std::string& helper_dir,
                                              const std::string& model_dir = "") {
    // ── Try GigaToken (fast Rust) ─────────────────────────────────────
    // Thread-safe init via std::call_once
    if (!model_dir.empty()) {
        static GigaTokenWrapper gt;
        static bool gt_ok = false;
        static std::once_flag gt_flag;
        std::call_once(gt_flag, [&]() {
            // Search for .so next to binary, then CWD, then third_party
            char exe_buf[4096];
            ssize_t exe_len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
            std::string exe_dir;
            if (exe_len > 0) {
                exe_buf[exe_len] = '\0';
                exe_dir = std::string(exe_buf);
                auto p = exe_dir.rfind('/');
                if (p != std::string::npos) exe_dir = exe_dir.substr(0, p);
            }
            std::string so_paths = ".:./bin";
            if (!exe_dir.empty()) so_paths = exe_dir + ":" + so_paths;
            gt_ok = gt.load(so_paths);
            if (gt_ok) {
                gt_ok = gt.load_tokenizer(model_dir);
                if (gt_ok) {
                    Logger::info(("GigaToken: loaded tokenizer from " + model_dir).c_str());
                } else {
                    Logger::info(("GigaToken: no tokenizer.json in " + model_dir + ", fallback to Python").c_str());
                }
            } else {
                Logger::info("GigaToken: .so not found, using Python subprocess");
            }
        });
        if (gt_ok && gt.has_tokenizer()) {
            auto ids = gt.encode(prompt);
            if (!ids.empty()) {
                return std::vector<int>(ids.begin(), ids.end());
            }
        }
    }

    // ── Fallback: Python subprocess ───────────────────────────────────
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        pf << prompt;
    }
    int ret = run_python_script(helper_dir + "/tokenize_helper.py");
    if (ret != 0) { Logger::error("Tokenization failed"); return {}; }

    std::vector<int> tokens;
    std::ifstream tf(token_file);
    int tid;
    while (tf >> tid) tokens.push_back(tid);
    return tokens;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: list
// ═══════════════════════════════════════════════════════════════════════════

int cmd_list() {
    std::string dir = models_dir();
    struct stat st;
    if (stat(dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        Logger::info("No models found (~/.terllama/models/ does not exist)");
        return 0;
    }

    std::ifstream mj(models_json_path());
    if (!mj) {
        Logger::info("No models installed.");
        return 0;
    }

    Logger::info("Installed models:");
    Logger::info(std::string(60, '-').c_str());

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
            std::string line = "  " + id + " (" + fmt + ") — " + fmt_size((double)sz);
            if (!ts.empty()) line += "  [" + ts + "]";
            Logger::info(line.c_str());
            count++;
        }
        pos = end + 1;
    }

    if (count == 0) {
        Logger::info("  (no models found)");
    }
    Logger::info("Models directory: {}", dir);
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
    Logger::info("Model: {}", model_id);
    Logger::info("Path:  {}", model_dir);

    if (stat(extra_path.c_str(), &st) == 0) {
        Logger::info("  Config + embedding: {}", fmt_size((double)st.st_size));
    } else {
        Logger::info("  Config: not found");
    }

    if (stat(i2s_path.c_str(), &st) == 0) {
        Logger::info("  Weights (I2_S):     {}", fmt_size((double)st.st_size));
    } else if (stat(als_path.c_str(), &st) == 0) {
        Logger::info("  Weights (ALS):      {}", fmt_size((double)st.st_size));
    } else {
        Logger::info("  Weights: not found");
    }

    try {
        auto cfg = load_config(extra_path);
        Logger::info("  Architecture:");
        Logger::info("    Parameters:      ~{}M", cfg.vocab_size * cfg.hidden_size / 1000000);
        Logger::info("    Hidden size:      {}", cfg.hidden_size);
        Logger::info("    Layers:           {}", cfg.num_hidden_layers);
        Logger::info("    Attention heads:  {}", cfg.num_attention_heads);
        Logger::info("    KV heads:         {}", cfg.num_key_value_heads);
        Logger::info("    Head dim:         {}", cfg.head_dim);
        Logger::info("    Vocab size:       {}", cfg.vocab_size);
        Logger::info("    Max seq len:      {}", cfg.max_position_embeddings);
        Logger::info("    RMS norm eps:     {}", cfg.rms_norm_eps);
        Logger::info("    RoPE theta:       {}", cfg.rope_theta);
    } catch (...) {
        Logger::info("  (could not read config)");
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
        Logger::error("Model not found: {}", model_id);
        return 1;
    }

    auto rm_file = [](const std::string& path) {
        if (unlink(path.c_str()) == 0)
            Logger::info("  Removed: {}", path);
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

    Logger::info("Model removed: {}", model_id);
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

    Logger::info("Terllama Chat — CPU: {}  |  Model: {}", cpu_arch_name(arch), model_id);
    Logger::info(std::string(50, '=').c_str());

    Logger::info("Loading model...");
    auto loaded = load_model_from(model_dir);
    auto& cfg = loaded.cfg;
    auto& embedding = loaded.embedding;
    auto& layer_norms = loaded.layer_norms;
    auto& final_norm = loaded.final_norm;
    auto& layers = loaded.layers;
    Logger::info("  Loaded {} layers.", layers.size());
    Logger::info("  Final norm (first 5): {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
                 final_norm[0], final_norm[1], final_norm[2], final_norm[3], final_norm[4]);
    {
        int last_start = std::max(0, (int)final_norm.size() - 5);
        Logger::info("  Final norm (last 5):  {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}",
            final_norm[last_start], final_norm[last_start+1], final_norm[last_start+2],
            final_norm[last_start+3], final_norm[last_start+4]);
    }

    auto rope = build_rope_cache(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);

    Logger::info("Prompt: {}", prompt_text);
    auto prompt_tokens = tokenize_with_helper(prompt_text, helper_dir, model_dir);

    if (prompt_tokens.empty()) {
        Logger::error("Tokenization failed");
        return 1;
    }

    auto [output_tokens, total_ms] = generate(
        prompt_tokens, max_tokens, temperature,
        cfg, embedding, layers, final_norm, layer_norms, rope);

    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = loaded.tokenizer.decode(all_tokens);
    std::string prompt_decoded = loaded.tokenizer.decode(prompt_tokens);

    Logger::info("=== Response ===");
    Logger::info(decoded.c_str());

    double total_tokens = (double)(prompt_tokens.size() + output_tokens.size());
    Logger::info("── Performance ──");
    Logger::info("  Time:       {} ms", total_ms);
    Logger::info("  Generated:  {} tokens", output_tokens.size());
    Logger::info("  Speed:      {} tok/s", (1000.0 * total_tokens / total_ms));
    Logger::info("  Kernel:     {}", cpu_arch_name(arch));
    Logger::info("\033[32m> Generated {} tokens in {:.1f}s ({:.1f} tok/sec)\033[0m",
        output_tokens.size(), total_ms / 1000.0,
        1000.0 * total_tokens / total_ms);

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
        Logger::error("Usage: {} chat --model <name> [--prompt \"text\"] [--max-tokens N] [--temp T]", argv[0]);
        return 1;
    }

    // Interactive mode — single-turn loop
    if (prompt_text.empty()) {
        Logger::info("Terllama Chat — {}", model_id);
        Logger::info("Type your messages. Ctrl+C or empty line to exit.");
        Logger::info(std::string(50, '-').c_str());

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
        while (!g_interrupted) {
            Logger::info("You: ");
            if (!std::getline(std::cin, line) || line.empty()) break;

            auto tokens = tokenize_with_helper(line, helper_dir, model_dir);
            if (tokens.empty()) continue;

            auto [out_tokens, ms] = generate(
                tokens, max_tokens, temperature,
                cfg, embedding, layers, final_norm, layer_norms, rope);

            std::vector<int> all = tokens;
            all.insert(all.end(), out_tokens.begin(), out_tokens.end());
            std::string text = loaded.tokenizer.decode(all);

            Logger::info("AI:  {}", text);
            double itok = (double)(tokens.size() + out_tokens.size());
            Logger::info("\033[32m> {:.1f}s ({:.1f} tok/sec)\033[0m",
                ms / 1000.0, 1000.0 * itok / ms);
        }
        Logger::info("Bye!");
        return 0;
    }

    return cmd_chat_simple(model_id, prompt_text, max_tokens, temperature);
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: pull (download alias)
// ═══════════════════════════════════════════════════════════════════════════

int cmd_pull(int argc, char** argv) {
    if (argc < 3) {
        Logger::error("Usage: {} pull <model> [--fmt i2s|als|gguf]", argv[0]);
        return 1;
    }

    std::string model_ref = argv[2];

    // Check registry for shortname resolution
    auto registry = get_registry();
    std::string hf_repo = model_ref;
    std::string auto_fmt;
    int64_t size_mb = 0;

    auto it = registry.find(model_ref);
    if (it != registry.end()) {
        hf_repo = it->second.hf_repo;
        auto_fmt = it->second.format;
        size_mb = it->second.size_mb;
        Logger::info("Resolved '%s' -> %s  (%s, %lld MB)",
                     model_ref.c_str(), hf_repo.c_str(),
                     auto_fmt.c_str(), (long long)size_mb);
    }

    // Build new argv for downloader_main: terllama download <hf_repo> [--fmt fmt]
    std::vector<const char*> args;
    args.push_back(argv[0]);
    args.push_back("download");
    args.push_back(hf_repo.c_str());

    // Check if user explicitly passed --fmt/--format
    bool has_explicit_fmt = false;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--format" || a == "--fmt") {
            has_explicit_fmt = true;
            break;
        }
    }

    if (!auto_fmt.empty() && !has_explicit_fmt) {
        args.push_back("--format");
        args.push_back(auto_fmt.c_str());
    }

    // Pass remaining args through (skip model_ref since resolved)
    for (int i = 3; i < argc; i++) {
        args.push_back(argv[i]);
    }

    int new_argc = (int)args.size();
    char** new_argv = new char*[new_argc + 1];
    for (int i = 0; i < new_argc; i++) {
        new_argv[i] = const_cast<char*>(args[i]);
    }
    new_argv[new_argc] = nullptr;

    if (size_mb > 0) {
        Logger::info("Downloading %s (%lld MB)...",
                     hf_repo.c_str(), (long long)size_mb);
        Logger::info("This may take a few minutes depending on model size and connection speed.");
    }

    int ret = downloader_main(new_argc, new_argv);
    delete[] new_argv;

    if (ret == 0) {
        Logger::info("Download complete!");
        // Compute model directory size for registry
        int64_t total_size = 0;
        std::string mdir = models_dir() + "/" + slugify(hf_repo);
        DIR* d = opendir(mdir.c_str());
        if (d) {
            struct dirent* e;
            while ((e = readdir(d)) != nullptr) {
                if (e->d_type == DT_REG) {
                    std::string fpath = mdir + "/" + e->d_name;
                    struct stat st;
                    if (stat(fpath.c_str(), &st) == 0)
                        total_size += st.st_size;
                }
            }
            closedir(d);
        }
        add_model_entry(slugify(hf_repo), auto_fmt.empty() ? "gguf" : auto_fmt, total_size);
    }
    return ret;
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL REGISTRY PERSISTENCE — write entry to ~/.terllama/models.json
// ═══════════════════════════════════════════════════════════════════════════

static void add_model_entry(const std::string& model_id,
                             const std::string& format,
                             int64_t size_bytes) {
    std::string jpath = models_json_path();
    std::string timestamp;
    {
        time_t now = time(nullptr);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", localtime(&now));
        timestamp = buf;
    }

    // Read existing entries
    std::vector<std::string> entries;
    std::ifstream inf(jpath);
    if (inf) {
        std::string content((std::istreambuf_iterator<char>(inf)),
                             std::istreambuf_iterator<char>());
        // Extract existing entries as raw JSON blocks
        size_t pos = 0;
        while (true) {
            auto start = content.find("{\"id\"", pos);
            if (start == std::string::npos) break;
            auto end = content.find("}", start);
            if (end == std::string::npos) break;
            entries.push_back(content.substr(start, end - start + 1));
            pos = end + 1;
        }
    }

    // Check if entry already exists; if so, remove it
    auto entry_exists = [&](const std::string& block) -> bool {
        auto p = block.find("\"id\":\"");
        if (p == std::string::npos) return false;
        p += 6;
        auto q = block.find("\"", p);
        if (q == std::string::npos) return false;
        return block.substr(p, q - p) == model_id;
    };
    entries.erase(std::remove_if(entries.begin(), entries.end(), entry_exists),
                  entries.end());

    // Add new entry
    std::string new_entry = "{\"id\":\"" + model_id
        + "\",\"format\":\"" + format
        + "\",\"size\":" + std::to_string(size_bytes)
        + ",\"downloaded\":\"" + timestamp + "\"}";
    entries.push_back(new_entry);

    // Write back
    std::ofstream of(jpath);
    if (!of) {
        Logger::warn("Cannot write {}", jpath);
        return;
    }
    of << "{\n  \"models\": [\n";
    for (size_t i = 0; i < entries.size(); i++) {
        if (i > 0) of << ",\n";
        of << "    " << entries[i];
    }
    of << "\n  ]\n}\n";
    Logger::info("Model entry saved to {}", jpath);
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

    Logger::info("Terllama — CPU: {}  |  Model: {}", cpu_arch_name(arch), model_dir);

    auto loaded = load_model_from(model_dir);
    auto& cfg = loaded.cfg;
    auto& embedding = loaded.embedding;
    auto& layer_norms = loaded.layer_norms;
    auto& final_norm = loaded.final_norm;
    auto& layers = loaded.layers;

    auto rope = build_rope_cache(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);

    Logger::info("Tokenizing prompt...");
    auto prompt_tokens = tokenize_with_helper(prompt, helper_dir, model_dir);

    Logger::info("=== Generating ===");
    auto [output_tokens, total_ms] = generate(
        prompt_tokens, max_tokens, temperature,
        cfg, embedding, layers, final_norm, layer_norms, rope);

    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = loaded.tokenizer.decode(all_tokens);

    Logger::info(decoded.c_str());

    double total_tokens_n = (double)(prompt_tokens.size() + output_tokens.size());
    Logger::info("── Performance ──");
    Logger::info("  Time:       {} ms", total_ms);
    Logger::info("  Generated:  {} tokens", output_tokens.size());
    Logger::info("  Speed:      {} tok/s", (1000.0 * total_tokens_n / total_ms));

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: bench
// ═══════════════════════════════════════════════════════════════════════════

int cmd_bench() {
    // Find default model
    std::string models_path = models_dir();
    // Scan for first available model directory
    std::string model_dir;
    DIR* dir = opendir(models_path.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            if (entry->d_type == DT_DIR && entry->d_name[0] != '.') {
                model_dir = models_path + "/" + entry->d_name;
                break;
            }
        }
        closedir(dir);
    }

    if (model_dir.empty()) {
        Logger::error("No models found. Pull one first: terllama pull tinyllama");
        Logger::error("Or set TERLLAMA_MODEL_DIR");
        return 1;
    }

    Logger::info("Benchmarking model: {}", model_dir);

    srand(42);
    CPUArch arch = detect_cpu_arch();
    auto loaded = load_model_from(model_dir);
    auto rope = build_rope_cache(loaded.cfg.max_position_embeddings,
                                 loaded.cfg.head_dim, loaded.cfg.rope_theta);

    // Fixed benchmark prompt
    std::string bench_prompt = "The future of AI is";
    auto prompt_tokens = tokenize_with_helper(bench_prompt, get_helper_dir(), model_dir);

    // Warmup run
    Logger::info("Warmup...");
    generate(prompt_tokens, 50, 0.7f,
             loaded.cfg, loaded.embedding, loaded.layers,
             loaded.final_norm, loaded.layer_norms, rope);

    // Benchmark: 3 runs
    const int NUM_RUNS = 3;
    struct BenchResult {
        double ms;
        int tokens;
    };
    std::vector<BenchResult> results;

    for (int r = 0; r < NUM_RUNS; r++) {
        Logger::info("Run {} / {}...", r + 1, NUM_RUNS);
        auto [tokens, ms] = generate(prompt_tokens, 100, 0.7f,
            loaded.cfg, loaded.embedding, loaded.layers,
            loaded.final_norm, loaded.layer_norms, rope);
        results.push_back(BenchResult{ms, (int)(prompt_tokens.size() + tokens.size())});
    }

    // Print benchmark table
    Logger::info("");
    Logger::info("═══════════════════════════════════════════");
    Logger::info("  Terllama Benchmark");
    Logger::info("═══════════════════════════════════════════");
    Logger::info("  Model:      {}", model_dir);
    Logger::info("  CPU Arch:   {}", cpu_arch_name(arch));
    Logger::info("  Prompt:     \"{}\" ({} tokens)", bench_prompt, prompt_tokens.size());
    Logger::info("");
    Logger::info("  Run  |  Time (ms)  |  Tokens  |  Speed (tok/s)");
    Logger::info("  ─────┼────────────┼──────────┼───────────────");
    double avg_speed = 0;
    for (int r = 0; r < NUM_RUNS; r++) {
        double speed = 1000.0 * results[r].tokens / results[r].ms;
        avg_speed += speed;
        Logger::info("  {}    |  {:.0f}       |  {}      |  {:.1f}",
            r + 1, results[r].ms, results[r].tokens, speed);
    }
    avg_speed /= NUM_RUNS;
    Logger::info("  ─────┼────────────┼──────────┼───────────────");
    Logger::info("  Avg   |            |          |  \033[32m{:.1f} tok/s\033[0m", avg_speed);
    Logger::info("");

    // Save to benchmarks.json
    std::string bench_path = home_dir() + "/.terllama/benchmarks.json";
    std::ofstream bf(bench_path);
    if (bf) {
        bf << "{\n";
        bf << "  \"model\": \"" << model_dir << "\",\n";
        bf << "  \"arch\": \"" << cpu_arch_name(arch) << "\",\n";
        bf << "  \"prompt_tokens\": " << prompt_tokens.size() << ",\n";
        bf << "  \"avg_speed_tok_s\": " << avg_speed << ",\n";
        bf << "  \"runs\": [\n";
        for (int r = 0; r < NUM_RUNS; r++) {
            if (r > 0) bf << ",\n";
            bf << "    { \"run\": " << (r+1)
               << ", \"ms\": " << results[r].ms
               << ", \"tokens\": " << results[r].tokens << " }";
        }
        bf << "\n  ]\n}\n";
        bf.close();
    }
    Logger::info("Results saved to {}", bench_path);

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

static bool has_any_model() {
    std::string mdir = models_dir();
    struct stat st;
    if (stat(mdir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
        return false;
    DIR* dir = opendir(mdir.c_str());
    if (!dir) return false;
    bool found = false;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_DIR && entry->d_name[0] != '.') {
            found = true;
            break;
        }
    }
    closedir(dir);
    return found;
}

// ═══════════════════════════════════════════════════════════════════════════
// SUBCOMMAND: bench tokenizer
// ═══════════════════════════════════════════════════════════════════════════

// Generate a repeatable test string of ~N bytes for benchmarking.
static std::string make_bench_text(size_t target_bytes) {
    const char* words[] = {
        "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog. ",
        "Hello ", "world! ", "This ", "is ", "a ", "test ", "of ", "the ", "tokenizer. ",
        "Artificial ", "intelligence ", "and ", "machine ", "learning ", "are ", "transforming ",
        "every ", "industry. ", "Neural ", "networks ", "process ", "vast ", "amounts ",
        "of ", "textual ", "data. ", "Transformers ", "are ", "the ", "backbone ",
        "of ", "modern ", "NLP. ", "Tokenization ", "is ", "the ", "first ", "step. ",
        "高性能 ", "AI ", "模型 ", "需要 ", "高效的 ", "分词器。",
        "Émoticônes ", "et ", "caractères ", "spéciaux: ", "é ", "ü ", "ñ ", "ç. ",
        "数字 ", "123 ", "456 ", "7890 ", "和 ", "符号 ", "!@#$% ", "^&*() ",
    };
    const int n = sizeof(words) / sizeof(words[0]);
    std::string result;
    result.reserve(target_bytes + 256);
    for (int i = 0; result.size() < target_bytes; i++) {
        result += words[i % n];
    }
    return result;
}

// Benchmark tokenizer throughput: GigaToken (Rust) vs Python subprocess.
int cmd_bench_tokenizer(int argc, char** argv) {
    std::string model_dir;
    std::string bench_text;

    // Parse: terllama bench tokenizer <model> [--text "..."]
    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--text" && i + 1 < argc) {
            bench_text = argv[++i];
        } else if (model_dir.empty()) {
            model_dir = arg;
        }
    }

    // Resolve model directory
    if (model_dir.empty()) {
        model_dir = std::getenv("TERLLAMA_MODEL_DIR")
            ? std::string(std::getenv("TERLLAMA_MODEL_DIR"))
            : models_dir();
        // If models_dir doesn't contain a model, scan for first one
        std::string first = model_dir;
        DIR* dir = opendir(first.c_str());
        if (dir) {
            struct dirent* entry;
            bool found_model = false;
            while ((entry = readdir(dir)) && !found_model) {
                if (entry->d_type == DT_DIR && entry->d_name[0] != '.') {
                    model_dir = first + "/" + entry->d_name;
                    found_model = true;
                }
            }
            closedir(dir);
        }
    }

    struct stat st;
    if (stat(model_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        Logger::error(("Model directory not found: " + model_dir).c_str());
        Logger::error(("Usage: " + std::string(argv[0]) + " bench tokenizer <model> [--text \"...\"]").c_str());
        return 1;
    }

    // Generate benchmark text if not provided
    if (bench_text.empty()) {
        bench_text = make_bench_text(1024 * 1024);  // ~1 MB
    }

    size_t text_bytes = bench_text.size();
    Logger::info("═══════════════════════════════════════════");
    Logger::info("   Tokenizer Benchmark");
    Logger::info(("   Model: " + model_dir).c_str());
    {
        char buf[128];
        snprintf(buf, sizeof(buf), "   Text:  %zu bytes (%.1f MB)",
                 text_bytes, (double)text_bytes / (1024.0 * 1024.0));
        Logger::info(buf);
    }
    Logger::info("═══════════════════════════════════════════");

    // ── 1. GigaToken ────────────────────────────────────────
    Logger::info("");
    Logger::info("── GigaToken (Rust) ──");

    // Hoisted stats for comparison section
    double gt_avg_sec = 0;
    double gt_mb_per_sec = 0;
    double gt_mtok_per_sec = 0;
    std::vector<int> gt_token_counts;
    bool gt_has_data = false;

    GigaTokenWrapper gt;
    auto t0 = std::chrono::high_resolution_clock::now();
    bool gt_loaded = gt.load(".:./bin");
    if (!gt_loaded) {
        Logger::info(("  Library load: FAILED (" + gt.error() + ")").c_str());
        Logger::info("  (Use Python fallback only)");
    } else {
        bool tok_loaded = gt.load_tokenizer(model_dir);
        if (!tok_loaded) {
            Logger::info(("  Tokenizer load: FAILED (" + gt.error() + ")").c_str());
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf), "  Tokenizer:  loaded (vocab_size=%d)", gt.vocab_size());
            Logger::info(buf);
            auto t1 = std::chrono::high_resolution_clock::now();

            // Warmup
            auto warmup = gt.encode("Warmup sentence for cache effects.");
            (void)warmup;

            // Benchmark: 5 runs
            const int NUM_RUNS = 5;
            double total_sec = 0;
            size_t total_tokens = 0;
            for (int r = 0; r < NUM_RUNS; r++) {
                auto start = std::chrono::high_resolution_clock::now();
                auto ids = gt.encode(bench_text);
                auto end = std::chrono::high_resolution_clock::now();
                double sec = std::chrono::duration<double>(end - start).count();
                total_sec += sec;
                total_tokens += ids.size();
                gt_token_counts.push_back((int)ids.size());
            }

            gt_avg_sec = total_sec / NUM_RUNS;
            double avg_tokens = (double)total_tokens / NUM_RUNS;
            gt_mb_per_sec = (double)text_bytes / gt_avg_sec / (1024.0 * 1024.0);
            gt_mtok_per_sec = avg_tokens / gt_avg_sec / 1e6;
            gt_has_data = true;

            {
                char buf[256];
                snprintf(buf, sizeof(buf), "  Runs:       %d (avg of %d)", NUM_RUNS, NUM_RUNS);
                Logger::info(buf);
            }
            {
                char buf[128];
                snprintf(buf, sizeof(buf), "  Time:       %.4fs avg", gt_avg_sec);
                Logger::info(buf);
            }
            {
                char buf[128];
                snprintf(buf, sizeof(buf), "  Throughput: %.2f MB/s (%.2f GB/s)",
                         gt_mb_per_sec, gt_mb_per_sec / 1024.0);
                Logger::info(buf);
            }
            {
                char buf[128];
                snprintf(buf, sizeof(buf), "  Tokens:     %.0f avg (%.2f Mtok/s)",
                         avg_tokens, gt_mtok_per_sec);
                Logger::info(buf);
            }
            {
                int c0 = gt_token_counts.size() > 0 ? gt_token_counts[0] : 0;
                int c1 = gt_token_counts.size() > 1 ? gt_token_counts[1] : 0;
                int c2 = gt_token_counts.size() > 2 ? gt_token_counts[2] : 0;
                char buf[128];
                snprintf(buf, sizeof(buf), "  Token IDs:  %d %d %d", c0, c1, c2);
                Logger::info(buf);
            }
        }
    }

    // ── 2. Python subprocess ────────────────────────────────
    Logger::info("");
    Logger::info("── Legacy (Python subprocess) ──");

    // Write text to temp file (same paths as tokenize_helper.py)
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        pf << bench_text;
    }
    std::string helper_dir = get_helper_dir();

    // Warmup
    {
        std::ofstream pf(prompt_file);
        pf << "Warmup sentence";
    }
    run_python_script(helper_dir + "/tokenize_helper.py");

    // Write real text
    {
        std::ofstream pf(prompt_file);
        pf << bench_text;
    }

    const int NUM_PY_RUNS = 3;  // Python is slow, fewer runs
    double py_total_sec = 0;
    size_t py_total_tokens = 0;
    std::vector<int> py_token_counts;
    for (int r = 0; r < NUM_PY_RUNS; r++) {
        auto start = std::chrono::high_resolution_clock::now();
        int ret = run_python_script(helper_dir + "/tokenize_helper.py");
        auto end = std::chrono::high_resolution_clock::now();
        if (ret != 0) {
            char buf[128];
            snprintf(buf, sizeof(buf), "  Python tokenization failed (exit=%d)", ret);
            Logger::error(buf);
            break;
        }
        double sec = std::chrono::duration<double>(end - start).count();
        py_total_sec += sec;

        std::vector<int> tokens;
        std::ifstream tf(token_file);
        int tid;
        while (tf >> tid) tokens.push_back(tid);
        py_total_tokens += tokens.size();
        py_token_counts.push_back((int)tokens.size());
    }

    if (!py_token_counts.empty()) {
        double py_avg_sec = py_total_sec / NUM_PY_RUNS;
        double py_avg_tokens = (double)py_total_tokens / NUM_PY_RUNS;
        double py_mb_per_sec = (double)text_bytes / py_avg_sec / (1024.0 * 1024.0);
        double py_mtok_per_sec = py_avg_tokens / py_avg_sec / 1e6;

        {
            char buf[256];
            snprintf(buf, sizeof(buf), "  Runs:       %d (avg of %d)", NUM_PY_RUNS, NUM_PY_RUNS);
            Logger::info(buf);
        }
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "  Time:       %.4fs avg", py_avg_sec);
            Logger::info(buf);
        }
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "  Throughput: %.2f MB/s (%.4f GB/s)",
                     py_mb_per_sec, py_mb_per_sec / 1024.0);
            Logger::info(buf);
        }
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "  Tokens:     %.0f avg (%.2f Mtok/s)",
                     py_avg_tokens, py_mtok_per_sec);
            Logger::info(buf);
        }

        // ── Comparison ──────────────────────────────────────
        Logger::info("");
        Logger::info("═══════════════════════════════════════════");
        Logger::info("   Comparison");
        Logger::info("");

        if (gt_has_data && gt_token_counts.size() > 0) {
            double speedup = py_avg_sec / gt_avg_sec;
            {
                char buf[128];
                snprintf(buf, sizeof(buf), "   Speedup:        %.1fx", speedup);
                Logger::info(buf);
            }

            // Token count validation
            int gt_n = gt_token_counts[0];
            int py_n = py_token_counts[0];
            if (gt_n == py_n) {
                char buf[128];
                snprintf(buf, sizeof(buf), "   Token match:    Both = %d", gt_n);
                Logger::info(buf);
            } else {
                double pct = 100.0 * std::abs(gt_n - py_n) / std::max(gt_n, py_n);
                char buf[128];
                snprintf(buf, sizeof(buf), "   Token match:    Giga=%d vs Py=%d  (diff %.2f%%)",
                         gt_n, py_n, pct);
                Logger::info(buf);
            }
        }

        Logger::info("");
        if (gt_has_data) {
            char buf[128];
            snprintf(buf, sizeof(buf), "   GigaToken:      %.1f MB/s  (%.1f Mtok/s)",
                     gt_mb_per_sec, gt_mtok_per_sec);
            Logger::info(buf);
        }
        {
            char buf[128];
            snprintf(buf, sizeof(buf), "   Legacy (Py):    %.1f MB/s  (%.1f Mtok/s)",
                     py_mb_per_sec, py_mtok_per_sec);
            Logger::info(buf);
        }
        Logger::info("═══════════════════════════════════════════");
    }

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// USAGE
// ═══════════════════════════════════════════════════════════════════════════

void print_usage(const char* prog) {
    // ─── First-run wizard: interactive download prompt ───
    if (!has_any_model()) {
        fprintf(stderr, "\nNo models found! Download TinyLlama-1.1B (139 MB) to test? [Y/n] ");
        fflush(stderr);
        std::string response;
        std::getline(std::cin, response);
        if (response.empty() || response == "Y" || response == "y" || response == "yes") {
            fprintf(stderr, "Downloading TinyLlama...\n");
            const char* pull_argv[] = {prog, "pull", "tinyllama"};
            int pull_argc = 3;
            cmd_pull(pull_argc, const_cast<char**>(pull_argv));
            fprintf(stderr, "\nDownload complete! Run '%s chat --model tinyllama' to start chatting.\n\n", prog);
        }
        fprintf(stderr, "\n");
    }

    Logger::error("Terllama v%s - CPU-first ternary LLM inference engine", TERLLAMA_VERSION);
    Logger::error("");
    Logger::error("Usage:");
    Logger::error("  {} \"prompt\" [max_tokens] [temp]    Run inference (legacy)", prog);
    Logger::error("  {} list                            List installed models", prog);
    Logger::error("  {} show <model>                    Show model info", prog);
    Logger::error("  {} pull <hf-repo> [--fmt i2s|als] Download model from HF", prog);
    Logger::error("  {} rm <model>                      Remove a model", prog);
    Logger::error("  {} serve [--port N] [--keep-alive SEC] [--memory-limit MB]  Start API server", prog);
    Logger::error("  {} chat --model <m> [--prompt p]   CLI chat", prog);
    Logger::error("  {} mote-build <input> <output> --experts K --topk k  Convert dense → MoTE", prog);
    Logger::error("  {} mote-list <path>                List MoTE layers", prog);
    Logger::error(("  " + std::string(prog) + " bench tokenizer <model> [--text \"...\"]  Benchmark tokenizer").c_str());
    Logger::error("");
    Logger::error("Environment:");
    Logger::error("  TERLLAMA_MODEL_DIR   model file directory");
    Logger::error("  TERLLAMA_PORT        server port (default 8375)");
    Logger::error("  TERLLAMA_ARCH        override CPU arch");
    Logger::error("");
    Logger::error("Usage examples:");
    Logger::error("  {} pull HuggingFaceTB/SmolLM2-135M --format i2s", prog);
    Logger::error("  {} list", prog);
    Logger::error("  {} serve --port 8375", prog);
    Logger::error("  {} \"Hello, world!\" 100 0.8", prog);
}

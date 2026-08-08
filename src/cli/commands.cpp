/*
 * commands.cpp — CLI subcommand implementations for Terllama
 *
 * All cmd_* functions migrated from main.cpp.
 * Tokenizer uses GigaToken (.so + tokenizer.json); the engine no longer
 * spawns a Python subprocess.
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
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <pwd.h>

#include "core/tokenizer.h"
#include "core/gigatoken_wrapper.h"

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

// ═══════════════════════════════════════════════════════════════════════════
// PATH SAFETY — model ids / HF repos must never escape models_dir()
// ═══════════════════════════════════════════════════════════════════════════

// Reject model ids that could traverse out of the models directory:
// empty, "..", or containing '/' or '\'. Returns "" on invalid input.
static std::string sanitize_model_id(const std::string& id) {
    if (id.empty()) return "";
    if (id.find("..") != std::string::npos) return "";
    if (id.find('/') != std::string::npos) return "";
    if (id.find('\\') != std::string::npos) return "";
    return id;
}

// Build "<models_dir>/<model_id>" and verify the canonicalized path stays
// inside models_dir(). Returns "" when the id is invalid or the resolved
// path escapes the models directory (defense in depth behind sanitize).
static std::string model_path_for_id(const std::string& model_id) {
    std::string safe = sanitize_model_id(model_id);
    if (safe.empty()) return "";
    std::string base = models_dir();
    std::string joined = base + "/" + safe;
    std::error_code ec;
    std::filesystem::path joined_canon = std::filesystem::weakly_canonical(joined, ec);
    std::filesystem::path base_canon = std::filesystem::weakly_canonical(base, ec);
    if (!ec) {
        std::string j = joined_canon.string();
        std::string b = base_canon.string();
        if (j != b && j.rfind(b + "/", 0) != 0) {
            Logger::warn("Model id '{}' resolves outside the models directory; ignoring", model_id);
            return "";
        }
    }
    return joined;
}

// Validate a HuggingFace repo id (owner/name). Rejects path traversal and
// anything other than [A-Za-z0-9_.-] with a single '/' separating owner
// and repo. Returns false when the repo is not a safe plain identifier.
static bool is_valid_hf_repo(const std::string& repo) {
    if (repo.empty()) return false;
    if (repo.find("..") != std::string::npos) return false;
    size_t slash = repo.find('/');
    if (slash == std::string::npos) return false;                     // require owner/repo
    if (repo.find('/', slash + 1) != std::string::npos) return false; // exactly one '/'
    for (size_t i = 0; i < repo.size(); i++) {
        if (i == slash) continue;
        char c = repo[i];
        bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
               || (c >= '0' && c <= '9') || c == '_' || c == '.' || c == '-';
        if (!ok) return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// STATE FILES — ~/.terllama models.json / benchmarks.json
// ═══════════════════════════════════════════════════════════════════════════

// Serialize read-modify-write of the registry file across concurrent
// pull/rm invocations.
static std::mutex g_registry_mutex;

// Write `content` to `path` atomically with mode 0600: write to a temp file
// in the same directory, fsync, then rename() over the target. This avoids a
// partially-written state file after a crash and keeps model metadata private
// to the user (default umask would otherwise make it world-readable).
// Returns false (with a logged warning) on any failure.
static bool write_state_file(const std::string& path, const std::string& content) {
    std::string tmp = path + ".tmp." + std::to_string((long)getpid());
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) {
        Logger::warn("Cannot open state file {} for writing: {}", tmp, std::strerror(errno));
        return false;
    }
    size_t written = 0;
    while (written < content.size()) {
        ssize_t n = ::write(fd, content.data() + written, content.size() - written);
        if (n < 0) {
            if (errno == EINTR) continue;
            Logger::warn("Failed writing {}: {}", tmp, std::strerror(errno));
            ::close(fd);
            ::unlink(tmp.c_str());
            return false;
        }
        written += (size_t)n;
    }
    if (::fsync(fd) != 0) {
        Logger::warn("fsync failed for {}: {}", tmp, std::strerror(errno));
    }
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        Logger::warn("rename {} -> {} failed: {}", tmp, path, std::strerror(errno));
        ::unlink(tmp.c_str());
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL REGISTRY — shortname → HuggingFace repo resolution
// ═══════════════════════════════════════════════════════════════════════════

struct RegistryEntry {
    std::string hf_repo;
    std::string format;  // "als" or "gguf"
    int64_t size_mb;
};

static std::unordered_map<std::string, RegistryEntry> get_registry() {
    return {
        {"tinyllama",    {"TinyLlama/TinyLlama-1.1B-Chat-v1.0", "als",  139}},
        {"smolLM2",      {"HuggingFaceTB/SmolLM2-135M",          "als",  54}},
        {"mistral-7b",   {"mistralai/Mistral-7B-v0.3",           "gguf", 4100}},
        {"llama-3.1-8b", {"meta-llama/Llama-3.1-8B",             "gguf", 4800}},
        {"phi-3.5-mini", {"microsoft/Phi-3.5-mini-instruct",     "gguf", 2600}},
        {"gemma-2b",     {"google/gemma-2b-it",                  "gguf", 1400}},
    };
}

// Legacy Python helper (run_python_script) removed — the engine no longer
// spawns python3. Any runtime path that needs tokenization must use
// GigaToken or fail cleanly.

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER (GigaToken encode/decode; no Python fallback)
// ═══════════════════════════════════════════════════════════════════════════

// Shared GigaToken instance for the CLI (encode + decode).
static GigaTokenWrapper g_cli_gigatoken;
static bool             g_cli_gigatoken_ok = false;
static std::once_flag   g_cli_gigatoken_flag;

// Lazy-load libgigatoken_rs.so + HF tokenizer.json from the model dir.
// Thread-safe via std::call_once.
static void ensure_gigatoken(const std::string& model_dir) {
    std::call_once(g_cli_gigatoken_flag, [&]() {
        if (model_dir.empty()) return;
        // Resolve the .so relative to the executable ONLY — never from CWD
        // or ./bin (dlopen hijack prevention, issue #6).
        char exe_buf[4096];
        ssize_t exe_len = readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
        std::string so_paths;
        if (exe_len > 0) {
            exe_buf[exe_len] = '\0';
            std::string exe_dir(exe_buf);
            auto p = exe_dir.rfind('/');
            if (p != std::string::npos) so_paths = exe_dir.substr(0, p);
        }
        if (so_paths.empty()) {
            Logger::info("GigaToken: cannot resolve exe dir — .so search disabled");
            return;
        }
        g_cli_gigatoken_ok = g_cli_gigatoken.load(so_paths);
        if (g_cli_gigatoken_ok) {
            g_cli_gigatoken_ok = g_cli_gigatoken.load_tokenizer(model_dir);
            if (g_cli_gigatoken_ok) {
                Logger::info(("GigaToken: loaded tokenizer from " + model_dir).c_str());
            } else {
                Logger::info(("GigaToken: no tokenizer.json in " + model_dir + ", tokenizer unavailable").c_str());
            }
        } else {
            Logger::info("GigaToken: .so not found, tokenizer unavailable");
        }
    });
}

// Try GigaToken encode. There is no Python fallback — if GigaToken cannot
// load (no .so / no tokenizer.json), fail cleanly with a clear error.
// model_dir should contain tokenizer.json for GigaToken to load.
static std::vector<int> tokenize_with_helper(const std::string& prompt,
                                              const std::string& model_dir = "") {
    ensure_gigatoken(model_dir);
    if (g_cli_gigatoken_ok && g_cli_gigatoken.has_tokenizer()) {
        auto ids = g_cli_gigatoken.encode(prompt);
        return std::vector<int>(ids.begin(), ids.end());
    }

    // No Python fallback: the engine must not require Python.
    Logger::error("Tokenizer unavailable: no GigaToken .so / tokenizer.json and native encode is not supported. Install the bundled libgigatoken_rs.so or provide tokenizer.json.");
    return {};
}

// Decode with fallback: native (llama) → GigaToken (byte-level BPE).
// No Python fallback — if neither can decode, fail cleanly.
static std::string decode_with_fallback(const Tokenizer& tokenizer,
                                         const std::vector<int>& token_ids,
                                         const std::string& model_dir = "") {
    // Fast path: native tokenizer (llama style)
    {
        std::string native = tokenizer.decode(token_ids);
        if (!native.empty() && native != "?") return native;
    }

    // GigaToken path
    ensure_gigatoken(model_dir);
    if (g_cli_gigatoken_ok && g_cli_gigatoken.has_tokenizer()) {
        std::vector<uint32_t> ids(token_ids.begin(), token_ids.end());
        std::string text = g_cli_gigatoken.decode(ids);
        if (!text.empty()) return text;
    }

    // No Python fallback. Log once per process to avoid spamming streaming
    // loops; return empty so callers fail gracefully.
    static std::once_flag warned;
    std::call_once(warned, [] {
        Logger::error("Tokenizer unavailable: no native vocab and no GigaToken .so / tokenizer.json. Install the bundled libgigatoken_rs.so or provide tokenizer.json.");
    });
    return "";
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
    std::string model_dir;
    const char* env_dir = std::getenv("TERLLAMA_MODEL_DIR");
    if (env_dir) {
        model_dir = std::string(env_dir);
    } else {
        model_dir = model_path_for_id(model_id);
        if (model_dir.empty()) {
            Logger::error("Invalid model id: {}", model_id);
            return 1;
        }
    }

    std::string extra_path = model_dir + "/model_extra.bin";
    std::string als_path  = model_dir + "/model_decomposed.bin";

    struct stat st;
    Logger::info("Model: {}", model_id);
    Logger::info("Path:  {}", model_dir);

    if (stat(extra_path.c_str(), &st) == 0) {
        Logger::info("  Config + embedding: {}", fmt_size((double)st.st_size));
    } else {
        Logger::info("  Config: not found");
    }

    if (stat(als_path.c_str(), &st) == 0) {
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
    std::string model_dir = model_path_for_id(model_id);
    if (model_dir.empty()) {
        Logger::error("Invalid model id: {}", model_id);
        return 1;
    }
    struct stat st;

    if (stat(model_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        Logger::error("Model not found: {}", model_id);
        return 1;
    }

    auto rm_file = [](const std::string& path) {
        if (unlink(path.c_str()) == 0)
            Logger::info("  Removed: {}", path);
    };

    rm_file(model_dir + "/model_decomposed.bin");
    rm_file(model_dir + "/model_extra.bin");

    rmdir(model_dir.c_str());

    std::string jpath = models_json_path();
    // Serialize concurrent pull/rm, then rewrite the registry atomically
    // (temp + fsync + rename, mode 0600) — never leave a partial file.
    std::lock_guard<std::mutex> lock(g_registry_mutex);
    std::ifstream inf(jpath);
    if (inf) {
        std::string content((std::istreambuf_iterator<char>(inf)),
                             std::istreambuf_iterator<char>());
        inf.close();

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
        if (!write_state_file(jpath, result)) {
            Logger::warn("Failed to update registry after removing '{}'", model_id);
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
    std::string model_dir;
    const char* env_dir = std::getenv("TERLLAMA_MODEL_DIR");
    if (env_dir) {
        model_dir = std::string(env_dir);
    } else {
        model_dir = model_path_for_id(model_id);
        if (model_dir.empty()) {
            Logger::error("Invalid model id: {}", model_id);
            return 1;
        }
    }

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
    auto prompt_tokens = tokenize_with_helper(prompt_text, model_dir);

    if (prompt_tokens.empty()) {
        Logger::error("Tokenization failed");
        return 1;
    }

    auto [output_tokens, total_ms] = generate(
        prompt_tokens, max_tokens, temperature,
        cfg, embedding, layers, final_norm, layer_norms, rope);

    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = decode_with_fallback(loaded.tokenizer, all_tokens, model_dir);
    std::string prompt_decoded = decode_with_fallback(loaded.tokenizer, prompt_tokens, model_dir);

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

        std::string model_dir;
        const char* env_dir = std::getenv("TERLLAMA_MODEL_DIR");
        if (env_dir) {
            model_dir = std::string(env_dir);
        } else {
            model_dir = model_path_for_id(model_id);
            if (model_dir.empty()) {
                Logger::error("Invalid model id: {}", model_id);
                return 1;
            }
        }

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

            auto tokens = tokenize_with_helper(line, model_dir);
            if (tokens.empty()) continue;

            auto [out_tokens, ms] = generate(
                tokens, max_tokens, temperature,
                cfg, embedding, layers, final_norm, layer_norms, rope);

            std::vector<int> all = tokens;
            all.insert(all.end(), out_tokens.begin(), out_tokens.end());
            std::string text = decode_with_fallback(loaded.tokenizer, all, model_dir);

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
        Logger::error("Usage: {} pull <model> [--fmt als|gguf]", argv[0]);
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

    // Reject unsafe repo ids before they reach slugify/downloader: only
    // [A-Za-z0-9_.-] with a single '/' separating owner/repo is allowed,
    // so the slugified out dir can never escape models_dir().
    if (!is_valid_hf_repo(hf_repo)) {
        Logger::error("Invalid model repo: '{}'. Expected owner/name using only [A-Za-z0-9_.-] (e.g. owner/repo).", hf_repo);
        return 1;
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
        // Compute model directory size for registry. Re-verify the slugified
        // out dir stays under models_dir() (input was validated above; this
        // is defense in depth). Empty => invalid; opendir below just fails.
        int64_t total_size = 0;
        std::string mdir = model_path_for_id(slugify(hf_repo));
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
    // Serialize concurrent pull/rm; the whole read-modify-write must hold
    // the lock so two writers cannot lose each other's entries.
    std::lock_guard<std::mutex> lock(g_registry_mutex);
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

    // Write back atomically (temp + fsync + rename) with mode 0600.
    std::string content = "{\n  \"models\": [\n";
    for (size_t i = 0; i < entries.size(); i++) {
        if (i > 0) content += ",\n";
        content += "    " + entries[i];
    }
    content += "\n  ]\n}\n";
    if (!write_state_file(jpath, content)) {
        Logger::warn("Failed to save model entry for '{}'", model_id);
        return;
    }
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
    auto prompt_tokens = tokenize_with_helper(prompt, model_dir);

    if (prompt_tokens.empty()) {
        Logger::error("Tokenization failed — aborting before generation");
        return 1;
    }

    Logger::info("=== Generating ===");
    auto [output_tokens, total_ms] = generate(
        prompt_tokens, max_tokens, temperature,
        cfg, embedding, layers, final_norm, layer_norms, rope);

    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = decode_with_fallback(loaded.tokenizer, all_tokens, model_dir);

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
    auto prompt_tokens = tokenize_with_helper(bench_prompt, model_dir);

    if (prompt_tokens.empty()) {
        Logger::error("Tokenization failed — cannot benchmark without a tokenizer");
        return 1;
    }

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

    // Save to benchmarks.json (atomic temp+fsync+rename, mode 0600 like
    // the model registry — benchmark results are user state files too).
    std::string bench_path = home_dir() + "/.terllama/benchmarks.json";
    std::string bench_content = "{\n";
    bench_content += "  \"model\": \"" + model_dir + "\",\n";
    bench_content += std::string("  \"arch\": \"") + cpu_arch_name(arch) + "\",\n";
    bench_content += "  \"prompt_tokens\": " + std::to_string(prompt_tokens.size()) + ",\n";
    bench_content += "  \"avg_speed_tok_s\": " + std::to_string(avg_speed) + ",\n";
    bench_content += "  \"runs\": [\n";
    for (int r = 0; r < NUM_RUNS; r++) {
        if (r > 0) bench_content += ",\n";
        bench_content += "    { \"run\": " + std::to_string(r + 1)
                       + ", \"ms\": " + std::to_string(results[r].ms)
                       + ", \"tokens\": " + std::to_string(results[r].tokens) + " }";
    }
    bench_content += "\n  ]\n}\n";
    if (write_state_file(bench_path, bench_content)) {
        Logger::info("Results saved to {}", bench_path);
    } else {
        Logger::warn("Could not save benchmark results to {}", bench_path);
    }

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

// Benchmark GigaToken tokenizer throughput. (Legacy Python-subprocess leg
// removed: the engine no longer requires Python.)
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

    double gt_avg_sec = 0;
    double gt_mb_per_sec = 0;
    double gt_mtok_per_sec = 0;
    std::vector<int> gt_token_counts;
    bool gt_ready = false;

    GigaTokenWrapper gt;
    bool gt_loaded = gt.load(".:./bin");
    if (!gt_loaded) {
        Logger::info(("  Library load: FAILED (" + gt.error() + ")").c_str());
    } else {
        bool tok_loaded = gt.load_tokenizer(model_dir);
        if (!tok_loaded) {
            Logger::info(("  Tokenizer load: FAILED (" + gt.error() + ")").c_str());
        } else {
            char buf[128];
            snprintf(buf, sizeof(buf), "  Tokenizer:  loaded (vocab_size=%d)", gt.vocab_size());
            Logger::info(buf);

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
            gt_ready = true;

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

    // ── 2. Legacy Python subprocess ────────────────────────
    // Removed: the engine no longer requires Python and never spawns python3.
    // GigaToken (above) is the only tokenizer path, so there is nothing to
    // benchmark against. scripts/tokenize_helper.py is kept only as a dev tool.
    Logger::info("");
    Logger::info("── Legacy (Python subprocess) ──");
    Logger::info("  Removed — engine no longer spawns python3. GigaToken is the only encode path.");
    if (!gt_ready) {
        Logger::info("  GigaToken unavailable — nothing to benchmark.");
        Logger::info("  Install the bundled libgigatoken_rs.so or provide tokenizer.json");
        Logger::info("  in the model directory to benchmark tokenization.");
    } else {
        char buf[160];
        snprintf(buf, sizeof(buf), "  GigaToken:      %.1f MB/s  (%.1f Mtok/s)",
                 gt_mb_per_sec, gt_mtok_per_sec);
        Logger::info(buf);
    }
    Logger::info("═══════════════════════════════════════════");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// USAGE
// ═══════════════════════════════════════════════════════════════════════════

void print_usage(const char* prog) {
    // ─── First-run wizard: interactive download prompt ───
    if (!has_any_model()) {
        fprintf(stderr, "\nNo models found. Download TinyLlama-1.1B (139 MB) to test? [y/N] ");
        fflush(stderr);
        std::string response;
        std::getline(std::cin, response);
        // Explicit consent only: bare Enter (empty) or anything other than
        // y/yes must NOT trigger a download.
        if (response == "Y" || response == "y" || response == "yes") {
            fprintf(stderr, "Downloading TinyLlama...\n");
            const char* pull_argv[] = {prog, "pull", "tinyllama"};
            int pull_argc = 3;
            cmd_pull(pull_argc, const_cast<char**>(pull_argv));
            fprintf(stderr, "\nDownload complete! Run '%s chat --model tinyllama' to start chatting.\n\n", prog);
        } else {
            fprintf(stderr, "Skipping download. You can install it later with '%s pull tinyllama'.\n", prog);
        }
        fprintf(stderr, "\n");
    }

    Logger::error("Terllama v%s - CPU-first ternary LLM inference engine", TERLLAMA_VERSION);
    Logger::error("");
    Logger::error("Usage:");
    Logger::error("  {} \"prompt\" [max_tokens] [temp]    Run inference (legacy)", prog);
    Logger::error("  {} list                            List installed models", prog);
    Logger::error("  {} show <model>                    Show model info", prog);
    Logger::error("  {} pull <hf-repo> [--fmt als|gguf] Download model from HF", prog);
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
    Logger::error("  {} pull HuggingFaceTB/SmolLM2-135M --format als", prog);
    Logger::error("  {} list", prog);
    Logger::error("  {} serve --port 8375", prog);
    Logger::error("  {} \"Hello, world!\" 100 0.8", prog);
}

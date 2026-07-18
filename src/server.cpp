/*
 * server.cpp — HTTP server for Terllama (OpenAI-compatible API)
 *
 * Endpoints:
 *   GET  /v1/models              List available models
 *   POST /v1/chat/completions    Chat completions (streaming + non-streaming)
 *   POST /v1/completions         Text completions (streaming + non-streaming)
 *   GET  /health                 Health check
 *   GET  /                       Serve web UI or API listing
 *
 * Build:
 *   g++ -std=c++17 -O3 -fopenmp -I. -Ithird_party \
 *       src/server.cpp src/server/handlers.cpp src/dispatcher.cpp src/kernel_scalar.cpp src/kernel_avx2.cpp \
 *       -o terllama-server -lm -fopenmp -lpthread
 *
 * Usage:
 *   ./terllama-server [port]
 *   TERLLAMA_MODEL_DIR=/path/to/model ./terllama-server
 *   TERLLAMA_PORT=8080 ./terllama-server
 */

#include "server/handlers.h"
#include "kernel_decl.h"
#include <json.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <climits>
#include <unistd.h>
#include <libgen.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include "core/logger.h"

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL MODEL STATE (defined here, accessed extern from handlers)
// ═══════════════════════════════════════════════════════════════════════════

ServerModelState g_model;
std::mutex       g_model_mutex;
std::string      g_api_key;
size_t           g_memory_limit = 0;
std::atomic<int> g_active_requests{0};
std::atomic<long long> g_last_request_time{0};

// ═══════════════════════════════════════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════════════════════════════════════

bool init_server(const std::string& model_dir) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (g_model.loaded) return true;

    g_model.model_dir  = model_dir;
    // Resolve scripts/ relative to this binary (co-located in repo)
    g_model.helper_dir = []{
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
        if (len != -1) { buf[len]='\0'; return std::string(dirname(buf))+"/scripts"; }
        return std::string("scripts");
    }();
    g_model.arch       = detect_cpu_arch();

    // Allow environment override for testing
    const char* arch_override = std::getenv("TERLLAMA_ARCH");
    if (arch_override) {
        std::string ao(arch_override);
        if (ao == "scalar")       g_model.arch = CPUArch::X86_64_SCALAR;
        else if (ao == "sse42")   g_model.arch = CPUArch::X86_64_SSE42;
        else if (ao == "avx")     g_model.arch = CPUArch::X86_64_AVX;
        else if (ao == "avx2")    g_model.arch = CPUArch::X86_64_AVX2;
        else if (ao == "avx512")  g_model.arch = CPUArch::X86_64_AVX512;
        else if (ao == "neon")    g_model.arch = CPUArch::ARM64_NEON;
    }

    Logger::info("Terllama — CPU: {}  |  Model: {}", cpu_arch_name(g_model.arch), model_dir);

    try {
        Logger::info("Loading model (auto-detect GGUF vs .bin)...");
        auto loaded = load_model_from(model_dir);
        g_model.cfg = loaded.cfg;
        g_model.embedding = loaded.embedding;
        g_model.layer_norms = loaded.layer_norms;
        g_model.final_norm = loaded.final_norm;
        g_model.layers = loaded.layers;
        Logger::info("  Loaded {} layers.", g_model.layers.size());

        Logger::info("Building RoPE cache...");
        g_model.rope = build_rope_cache(
            g_model.cfg.max_position_embeddings,
            g_model.cfg.head_dim,
            g_model.cfg.rope_theta);

        g_model.loaded = true;
        Logger::info("Server initialized.");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Failed to load model: {}", e.what());
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SERVER MAIN (callable from terllama serve)
// ═══════════════════════════════════════════════════════════════════════════

int server_main(int argc, char** argv) {
    srand((unsigned int)time(nullptr));

    // Model directory — --model <slug/path>, env var, or default to .
    std::string model_dir = ".";
    for (int i = 1; i < argc - 1; i++) {
        std::string a = argv[i];
        if (a == "--model" || a == "-m") { model_dir = argv[i + 1]; break; }
    }
    if (model_dir == "." || model_dir.find('/') == std::string::npos) {
        const char* env_m = std::getenv("TERLLAMA_MODEL_DIR");
        if (env_m) { model_dir = env_m; }
        else if (model_dir != ".") {
            // Resolve slug to ~/.terllama/models/<slug>
            model_dir = std::string(getenv("HOME") ? getenv("HOME") : "/root")
                      + "/.terllama/models/" + model_dir;
        }
    }

    // Port — handle --port N, -p N, positional, or env
    int port = 8375;
    for (int i = 1; i < argc - 1; i++) {
        std::string a = argv[i];
        if (a == "--port" || a == "-p") {
            port = std::stoi(argv[i + 1]);
            break;
        }
    }
    if (port == 8375) {
        // Check positional: skip subcommand name if present
        for (int i = 1; i < argc; i++) {
            std::string a = argv[i];
            if (a == "serve" || a == "server") continue;
            if (a[0] != '-') { port = std::stoi(a); break; }
        }
    }
    if (const char* env_port = std::getenv("TERLLAMA_PORT")) {
        port = std::stoi(env_port);
    }

    // API key — --api-key <key> or env
    for (int i = 1; i < argc - 1; i++) {
        std::string a = argv[i];
        if (a == "--api-key") {
            g_api_key = argv[i + 1];
            break;
        }
    }
    if (g_api_key.empty()) {
        const char* env_key = std::getenv("TERLLAMA_API_KEY");
        if (env_key) g_api_key = env_key;
    }

    std::string keep_alive_str = "5m";
    for (int i = 1; i < argc - 1; i++) {
        std::string a = argv[i];
        if (a == "--keep-alive") {
            keep_alive_str = argv[i + 1];
            break;
        }
    }

    long long keep_alive_ms = 5 * 60 * 1000; // default 5m
    if (!keep_alive_str.empty()) {
        try {
            if (keep_alive_str.back() == 'm') {
                keep_alive_ms = std::stoll(keep_alive_str.substr(0, keep_alive_str.size() - 1)) * 60 * 1000;
            } else if (keep_alive_str.back() == 's') {
                keep_alive_ms = std::stoll(keep_alive_str.substr(0, keep_alive_str.size() - 1)) * 1000;
            } else {
                keep_alive_ms = std::stoll(keep_alive_str) * 1000;
            }
        } catch (...) {
            Logger::warn("Invalid --keep-alive format: {}, default to 5m", keep_alive_str);
            keep_alive_ms = 5 * 60 * 1000;
        }
    }

    // Parse memory limit
    for (int i = 1; i < argc - 1; i++) {
        std::string a = argv[i];
        if (a == "--memory-limit") {
            std::string limit_str = argv[i + 1];
            try {
                char unit = limit_str.back();
                if (unit == 'G' || unit == 'g') {
                    g_memory_limit = std::stoull(limit_str.substr(0, limit_str.size() - 1)) * 1024 * 1024 * 1024;
                } else if (unit == 'M' || unit == 'm') {
                    g_memory_limit = std::stoull(limit_str.substr(0, limit_str.size() - 1)) * 1024 * 1024;
                } else if (unit == 'K' || unit == 'k') {
                    g_memory_limit = std::stoull(limit_str.substr(0, limit_str.size() - 1)) * 1024;
                } else {
                    g_memory_limit = std::stoull(limit_str);
                }
                Logger::info("Memory limit set to: {} bytes", g_memory_limit);
            } catch (...) {
                Logger::warn("Invalid --memory-limit format: {}, disabling limit", limit_str);
                g_memory_limit = 0;
            }
            break;
        }
    }

    std::string web_dir = model_dir + "/web";

    // ─── Init model ─────────────────────────────────────────────────────
    Logger::info("Initializing Terllama server...");
    if (!init_server(model_dir)) {
        Logger::error("FATAL: failed to load model from {}", model_dir);
        return 1;
    }

    // Start keep-alive watchdog thread
    if (keep_alive_ms > 0) {
        g_last_request_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        
        std::thread watchdog([keep_alive_ms]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::seconds(5));
                if (g_model.loaded) {
                    long long now = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                    long long last = g_last_request_time.load();
                    if (now - last > keep_alive_ms && g_active_requests.load() == 0) {
                        Logger::info("No request for {} ms. Unloading model to save memory...", keep_alive_ms);
                        std::lock_guard<std::mutex> lock(g_model_mutex);
                        g_model.embedding.clear();
                        g_model.embedding.shrink_to_fit();
                        g_model.layers.clear();
                        g_model.layers.shrink_to_fit();
                        g_model.final_norm.clear();
                        g_model.final_norm.shrink_to_fit();
                        g_model.layer_norms.clear();
                        g_model.layer_norms.shrink_to_fit();
                        g_model.rope.sin.clear();
                        g_model.rope.sin.shrink_to_fit();
                        g_model.rope.cos.clear();
                        g_model.rope.cos.shrink_to_fit();
                        g_model.loaded = false;
                        Logger::info("Model unloaded.");
                    }
                }
            }
        });
        watchdog.detach();
    }

    // ─── HTTP server ────────────────────────────────────────────────────
    httplib::Server svr;

    // Routes
    svr.Options(".*", handle_options);

    svr.Get ("/v1/models",           handle_models);
    svr.Post("/v1/chat/completions", handle_chat_completions);
    svr.Post("/v1/completions",      handle_completions);
    svr.Get ("/health",              handle_health);

    // Static file serving for web UI
    {
        std::ifstream test_web(web_dir + "/index.html");
        if (test_web.good()) {
            svr.set_base_dir(web_dir, "/");
            Logger::info("Web UI at http://0.0.0.0:{}/", port);
        } else {
            svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
                (void)req;
                add_cors_headers(res);
                send_json(res, json{
                    {"service",   "Terllama Inference Server"},
                    {"version",   "1.0.0"},
                    {"endpoints", json::array({
                        "GET  /v1/models",
                        "POST /v1/chat/completions",
                        "POST /v1/completions",
                        "GET  /health"
                    })}
                }.dump());
            });
            Logger::info("No web/index.html — serving API listing at /");
        }
    }

    // Error handler
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        add_cors_headers(res);
        if (res.status == 404) {
            send_error(res, "Not found: " + req.path, 404, "not_found");
        }
    });

    // Exception handler
    svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res,
                                  std::exception_ptr ep) {
        (void)req;
        add_cors_headers(res);
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            send_error(res, std::string("Internal error: ") + e.what(), 500);
        } catch (...) {
            send_error(res, "Unknown internal error", 500);
        }
    });

    svr.set_payload_max_length(10 * 1024 * 1024); // 10 MB max
    svr.set_read_timeout(300, 0);
    svr.set_write_timeout(300, 0);

    // ─── Banner ─────────────────────────────────────────────────────────
    Logger::info("");
    Logger::info("╔══════════════════════════════════════════════╗");
    Logger::info("║        Terllama Inference Server            ║");
    Logger::info("╠══════════════════════════════════════════════╣");
    {
        std::string cpu_line = "║  CPU:    ";
        cpu_line += cpu_arch_name(g_model.arch);
        int pad = 35 - (int)strlen(cpu_arch_name(g_model.arch));
        cpu_line.append(pad, ' ');
        cpu_line += "║";
        Logger::info(cpu_line.c_str());
    }
    Logger::info("║  Layers: {}  Hidden: {}  Heads: {}        ║",
                 g_model.cfg.num_hidden_layers, g_model.cfg.hidden_size,
                 g_model.cfg.num_attention_heads);
    {
        std::string port_line = "║  Port:   ";
        port_line += std::to_string(port);
        int pad = (port < 10000 ? 1 : 0) + (port < 1000 ? 1 : 0);
        port_line.append(36 - pad, ' ');
        port_line += "║";
        Logger::info(port_line.c_str());
    }
    {
        std::string api_line = "║  API:    http://0.0.0.0:";
        api_line += std::to_string(port);
        api_line += "/v1/models";
        int pad = 18 - (port > 9999 ? 5 : port > 999 ? 4 : 3);
        api_line.append(pad, ' ');
        api_line += "║";
        Logger::info(api_line.c_str());
    }
    Logger::info("╚══════════════════════════════════════════════╝");

    svr.listen("0.0.0.0", port);
    return 0;
}

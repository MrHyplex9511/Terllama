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

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL MODEL STATE (defined here, accessed extern from handlers)
// ═══════════════════════════════════════════════════════════════════════════

ServerModelState g_model;
std::mutex       g_model_mutex;
std::string      g_api_key;

// ═══════════════════════════════════════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════════════════════════════════════

static bool init_server(const std::string& model_dir) {
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

    std::cout << "Terllama — CPU: " << cpu_arch_name(g_model.arch)
              << "  |  Model: " << model_dir << std::endl;

    try {
        std::cout << "Loading model (auto-detect GGUF vs .bin)..." << std::endl;
        auto loaded = load_model_from(model_dir);
        g_model.cfg = loaded.cfg;
        g_model.embedding = loaded.embedding;
        g_model.layer_norms = loaded.layer_norms;
        g_model.final_norm = loaded.final_norm;
        g_model.layers = loaded.layers;
        std::cout << "  Loaded " << g_model.layers.size() << " layers."
                  << std::endl;

        std::cout << "Building RoPE cache..." << std::endl;
        g_model.rope = build_rope_cache(
            g_model.cfg.max_position_embeddings,
            g_model.cfg.head_dim,
            g_model.cfg.rope_theta);

        g_model.loaded = true;
        std::cout << "Server initialized." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
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

    std::string web_dir = model_dir + "/web";

    // ─── Init model ─────────────────────────────────────────────────────
    std::cout << "Initializing Terllama server..." << std::endl;
    if (!init_server(model_dir)) {
        std::cerr << "FATAL: failed to load model from " << model_dir << std::endl;
        return 1;
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
            std::cout << "Web UI at http://0.0.0.0:" << port << "/" << std::endl;
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
            std::cout << "No web/index.html — serving API listing at /" << std::endl;
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
    std::cout << "\n"
              << "╔══════════════════════════════════════════════╗\n"
              << "║        Terllama Inference Server            ║\n"
              << "╠══════════════════════════════════════════════╣\n"
              << "║  CPU:    " << cpu_arch_name(g_model.arch);
    {
        int pad = 35 - (int)strlen(cpu_arch_name(g_model.arch));
        for (int i = 0; i < pad; i++) std::cout << ' ';
        std::cout << "║\n";
    }
    std::cout << "║  Layers: " << g_model.cfg.num_hidden_layers
              << "  Hidden: " << g_model.cfg.hidden_size
              << "  Heads: " << g_model.cfg.num_attention_heads
              << "        ║\n"
              << "║  Port:   " << port;
    {
        int pad = (port < 10000 ? 1 : 0) + (port < 1000 ? 1 : 0);
        for (int i = 0; i < 36 - pad; i++) std::cout << ' ';
        std::cout << "║\n";
    }
    std::cout << "║  API:    http://0.0.0.0:" << port << "/v1/models";
    {
        int pad = 18 - (port > 9999 ? 5 : port > 999 ? 4 : 3);
        for (int i = 0; i < pad; i++) std::cout << ' ';
        std::cout << "║\n";
    }
    std::cout << "╚══════════════════════════════════════════════╝\n"
              << std::endl;

    svr.listen("0.0.0.0", port);
    return 0;
}

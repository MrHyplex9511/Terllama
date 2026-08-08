/*
 * worker_main.cpp — Terllama worker CLI entry point.
 *
 * Usage:
 *   terllama-worker --listen 127.0.0.1:9100 [--model PATH --shard START,END]
 *
 * With --model/--shard the worker loads its slice at startup; without them it
 * starts empty and waits for a POST /load from the coordinator.
 */
#include <sys/stat.h>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "protocol.h"
#include "worker.h"

namespace {

tldist::Worker* g_worker = nullptr;

void on_signal(int sig) {
    const char* name = (sig == SIGINT) ? "SIGINT" : "SIGTERM";
    Logger::info("terllama-worker: {} received, shutting down", name);
    if (g_worker) g_worker->stop();
}

void usage(FILE* out) {
    fprintf(out,
        "usage: terllama-worker [--listen HOST:PORT] [--model PATH --shard START,END]\n"
        "\n"
        "  --listen HOST:PORT   address to serve on (default 127.0.0.1:9100)\n"
        "  --model PATH         model directory (model_extra.bin + model_decomposed.bin,\n"
        "                       or a .gguf file/dir). If omitted, wait for POST /load.\n"
        "  --shard START,END    half-open transformer layer range [START, END) to own.\n"
        "                       Requires --model. END defaults to the model's layer count.\n"
        "  -h, --help           show this help\n");
}

// Derive the total transformer layer count for a model path without loading
// a shard: cheap config read for .bin models; full (startup-only) load for GGUF.
int derive_n_layers(const std::string& model_path) {
    struct stat st;
    std::string extra = model_path + "/model_extra.bin";
    if (stat(extra.c_str(), &st) == 0) {
        return load_config(extra).num_hidden_layers;  // loader.h (global ns)
    }
    if (model_path_for(model_path).empty()) {
        throw std::runtime_error("no model files found in " + model_path);
    }
    LoadedModel full = load_model_from(model_path);
    return full.cfg.num_hidden_layers;
}

}  // namespace

int main(int argc, char** argv) {
    std::string listen = "127.0.0.1:9100";
    std::string model_path;
    int start = 0;
    int end = -1;  // -1 = default to full model

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--listen" && i + 1 < argc) {
            listen = argv[++i];
        } else if (a == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (a == "--shard" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t comma = s.find(',');
            if (comma == std::string::npos) {
                fprintf(stderr, "terllama-worker: --shard expects START,END (got '%s')\n", s.c_str());
                return 1;
            }
            start = std::atoi(s.substr(0, comma).c_str());
            end = std::atoi(s.substr(comma + 1).c_str());
            if (start < 0 || end <= start) {
                fprintf(stderr, "terllama-worker: invalid shard range '%s'\n", s.c_str());
                return 1;
            }
        } else if (a == "-h" || a == "--help") {
            usage(stdout);
            return 0;
        } else {
            fprintf(stderr, "terllama-worker: unknown argument '%s'\n", a.c_str());
            usage(stderr);
            return 1;
        }
    }
    if (end >= 0 && model_path.empty()) {
        fprintf(stderr, "terllama-worker: --shard requires --model\n");
        return 1;
    }

    tldist::Worker worker(listen);
    g_worker = &worker;

    if (!model_path.empty()) {
        try {
            int n_layers = derive_n_layers(model_path);
            if (end < 0) end = n_layers;  // no --shard: own the whole model
            if (start >= n_layers || end > n_layers || end <= start) {
                fprintf(stderr,
                        "terllama-worker: shard [%d,%d) out of range (model has %d layers)\n",
                        start, end, n_layers);
                return 1;
            }
            tldist::ShardSpec spec;
            spec.device_rank = 0;
            spec.world_size = 1;
            spec.start_layer = start;
            spec.end_layer = end;
            spec.n_layers = n_layers;
            spec.is_first = (start == 0);
            spec.is_last = (end == n_layers);

            worker.load_model(model_path, spec);
            Logger::info("terllama-worker ready, layers [%d,%d), is_first=%d is_last=%d",
                         start, end, spec.is_first, spec.is_last);
        } catch (const std::exception& e) {
            Logger::error("terllama-worker: model load failed: {}", e.what());
            return 1;
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    return worker.run();
}

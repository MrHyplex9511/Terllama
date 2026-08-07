// ═══════════════════════════════════════════════════════════════════════════
// Terllama — HuggingFace model puller (native download + ALS convert)
// ═══════════════════════════════════════════════════════════════════════════
//
// The pull used to shell out to python3 (pip install + export script). It now
// runs entirely in-process: src/convert/export.cpp downloads config.json,
// tokenizer files and the safetensors checkpoint via libcurl, decomposes the
// weights with the native ALS pipeline, and writes model_decomposed.bin +
// model_extra.bin. Zero Python, zero subprocesses.

#include "convert/export.h"
#include "core/logger.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <atomic>
#include <chrono>

static std::string slugify(const std::string& repo) {
    std::string s = repo;
    for (auto& c : s) {
        if (c == '/') c = '-';
    }
    return s;
}

static void print_usage(const char* prog) {
    Logger::error("Usage: {} pull <hf_repo> [--format als|gguf]", prog);
    Logger::error("");
    Logger::error("Download a model from HuggingFace and convert to Terllama format.");
    Logger::error("");
    Logger::error("Arguments:");
    Logger::error("  <hf_repo>    HuggingFace repo (e.g. HuggingFaceTB/SmolLM2-135M)");
    Logger::error("  --format     'als' (default) or 'gguf'");
    Logger::error("  --outdir     output directory (default: ~/.terllama/models/<repo>)");
    Logger::error("");
    Logger::error("Models stored in ~/.terllama/models/<repo-name>/");
    Logger::error("Tracked in ~/.terllama/models.json");
}

int downloader_main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    std::string hf_repo;
    std::string format = "als";
    std::string outdir_override;  // optional --outdir <path>

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--format" || arg == "--fmt") {
            if (i + 1 >= argc) {
                Logger::error("Error: --format requires an argument");
                return 1;
            }
            format = argv[++i];
            if (format != "als" && format != "gguf") {
                Logger::error("Error: unknown format '{}' (use 'als')", format);
                return 1;
            }
        } else if (arg == "--outdir" || arg == "-o") {
            if (i + 1 >= argc) {
                Logger::error("Error: --outdir requires an argument");
                return 1;
            }
            outdir_override = argv[++i];
        } else if (hf_repo.empty()) {
            hf_repo = arg;
        } else {
            Logger::error("Error: unexpected argument '{}'", arg);
            return 1;
        }
    }

    if (hf_repo.empty()) {
        Logger::error("Error: missing HuggingFace repo");
        print_usage(argv[0]);
        return 1;
    }

    std::string model_slug = slugify(hf_repo);
    std::string out_dir = outdir_override;
    if (out_dir.empty()) {
        out_dir = std::string(getenv("HOME") ? getenv("HOME") : "/root")
                + "/.terllama/models/" + model_slug;
    }

    Logger::info("Downloading {} from HuggingFace...", hf_repo);
    Logger::info("Converting to {} format...", format);
    Logger::info("Output: {}", out_dir);

    // Spinner on stderr while the native converter downloads + decomposes;
    // [PROGRESS] lines (parseable by the desktop UI) go to stdout from
    // convert_model.
    std::atomic<bool> running{true};
    std::thread spinner([&]() {
        const char* frames = "|/-\\";
        int i = 0;
        while (running) {
            fprintf(stderr, "\r  Converting... %c", frames[i++ % 4]);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        fprintf(stderr, "\r%*s\r", 40, "");
    });

    const int ret = terllama::convert_model(hf_repo, out_dir, 12, format);

    running = false;
    spinner.join();

    if (ret != 0) {
        Logger::error("Conversion failed (exit code {})", ret);
        return 1;
    }

    Logger::info("Model downloaded to: {}", out_dir);
    return 0;
}

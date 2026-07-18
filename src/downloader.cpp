// ═══════════════════════════════════════════════════════════════════════════
// Terllama — HuggingFace model puller (download + convert via Python export)
// ═══════════════════════════════════════════════════════════════════════════
//
// The actual download + ternary conversion is done by the Python script
// scripts/export_ternary_model_bitnet.py. This C++ stub finds the script
// relative to the binary path and invokes it with the right args.

#include "core/logger.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <libgen.h>   // dirname
#include <unistd.h>   // readlink, access
#include <limits.h>   // PATH_MAX
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include <spawn.h>
#include <sys/wait.h>

extern char **environ;

static std::string get_bin_dir(const char* argv0) {
    // Try /proc/self/exe first (Linux)
    char buf[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len != -1) {
        buf[len] = '\0';
        char* d = dirname(buf);
        return std::string(d);
    }
    // Fallback: argv[0]
    char* d = dirname(const_cast<char*>(argv0));
    return std::string(d);
}

static std::string slugify(const std::string& repo) {
    std::string s = repo;
    for (auto& c : s) {
        if (c == '/') c = '-';
    }
    return s;
}

static void print_usage(const char* prog) {
    Logger::error("Usage: {} pull <hf_repo> [--format i2s|als]", prog);
    Logger::error("");
    Logger::error("Download a model from HuggingFace and convert to Terllama format.");
    Logger::error("");
    Logger::error("Arguments:");
    Logger::error("  <hf_repo>    HuggingFace repo (e.g. HuggingFaceTB/SmolLM2-135M)");
    Logger::error("  --format     'i2s' (default) or 'als'");
    Logger::error("");
    Logger::error("Models stored in ~/.terllama/models/<repo-name>/");
    Logger::error("Tracked in ~/.terllama/models.json");
}

int downloader_main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    std::string hf_repo;
    std::string format = "i2s";

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--format" || arg == "--fmt") {
            if (i + 1 >= argc) {
                Logger::error("Error: --format requires an argument");
                return 1;
            }
            format = argv[++i];
            if (format != "i2s" && format != "als" && format != "gguf") {
                Logger::error("Error: unknown format '{}'", format);
                return 1;
            }
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

    // Find scripts dir relative to binary
    std::string bin_dir = get_bin_dir(argv[0]);
    std::string script_path = bin_dir + "/../scripts/export_ternary_model_bitnet.py";

    // Check if script exists
    if (access(script_path.c_str(), F_OK) != 0) {
        // Try relative to CWD
        script_path = "scripts/export_ternary_model_bitnet.py";
        if (access(script_path.c_str(), F_OK) != 0) {
            Logger::error("Error: can't find export script at scripts/export_ternary_model_bitnet.py");
            return 1;
        }
    }

    std::string model_slug = slugify(hf_repo);
    std::string out_dir = std::string(getenv("HOME") ? getenv("HOME") : "/root")
                        + "/.terllama/models/" + model_slug;

    Logger::info("Downloading {} from HuggingFace...", hf_repo);
    Logger::info("Converting to {} format...", format);
    Logger::info("Output: {}", out_dir);

    // Run pip install via posix_spawnp (no shell)
    auto run_spawn = [](const std::string& prog, const std::vector<std::string>& args) -> int {
        pid_t pid;
        std::vector<const char*> argv;
        argv.push_back(prog.c_str());
        for (const auto& a : args) argv.push_back(a.c_str());
        argv.push_back(nullptr);

        int ret = posix_spawnp(&pid, prog.c_str(), nullptr, nullptr,
                               const_cast<char* const*>(argv.data()), environ);
        if (ret != 0) return -1;
        int status;
        waitpid(pid, &status, 0);
        if (WIFEXITED(status)) return WEXITSTATUS(status);
        return -1;
    };

    // Install dependencies first
    Logger::info("  Installing Python dependencies...");
    int pip_ret = run_spawn("pip", {"install", "transformers", "torch", "-q"});
    if (pip_ret != 0) {
        Logger::error("pip install failed (exit {}), continuing anyway...", pip_ret);
    }

    // Run export script with spinner
    std::atomic<bool> downloading{true};
    std::thread spinner([&]() {
        const char* frames = "|/-\\";
        int i = 0;
        while (downloading) {
            fprintf(stderr, "\r  Converting... %c", frames[i++ % 4]);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        fprintf(stderr, "\r%*s\r", 40, "");
    });

    int ret = run_spawn("python3", {
        script_path,
        "--model", hf_repo,
        "--outdir", out_dir,
        "--format", format
    });

    downloading = false;
    spinner.join();

    if (ret != 0) {
        Logger::error("Export failed (exit code %d)", ret);
        return 1;
    }

    Logger::info("Model downloaded to: %s", out_dir.c_str());
    return 0;
}

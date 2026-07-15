// ═══════════════════════════════════════════════════════════════════════════
// Terllama — HuggingFace model puller (download + convert via Python export)
// ═══════════════════════════════════════════════════════════════════════════
//
// The actual download + ternary conversion is done by the Python script
// scripts/export_ternary_model_bitnet.py. This C++ stub finds the script
// relative to the binary path and invokes it with the right args.

#include <iostream>
#include <string>
#include <cstdlib>
#include <cstring>
#include <libgen.h>   // dirname
#include <unistd.h>   // readlink, access
#include <limits.h>   // PATH_MAX
#include <vector>

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
    std::cerr << "Usage: " << prog << " pull <hf_repo> [--format i2s|als]" << std::endl;
    std::cerr << "\nDownload a model from HuggingFace and convert to Terllama format." << std::endl;
    std::cerr << "\nArguments:" << std::endl;
    std::cerr << "  <hf_repo>    HuggingFace repo (e.g. HuggingFaceTB/SmolLM2-135M)" << std::endl;
    std::cerr << "  --format     'i2s' (default) or 'als'" << std::endl;
    std::cerr << "\nModels stored in ~/.terllama/models/<repo-name>/" << std::endl;
    std::cerr << "Tracked in ~/.terllama/models.json" << std::endl;
}

int downloader_main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    std::string hf_repo;
    std::string format = "i2s";

    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--format" || arg == "--fmt") {
            if (i + 1 >= argc) {
                std::cerr << "Error: --format requires an argument" << std::endl;
                return 1;
            }
            format = argv[++i];
            if (format != "i2s" && format != "als") {
                std::cerr << "Error: unknown format '" << format << "'" << std::endl;
                return 1;
            }
        } else if (hf_repo.empty()) {
            hf_repo = arg;
        } else {
            std::cerr << "Error: unexpected argument '" << arg << "'" << std::endl;
            return 1;
        }
    }

    if (hf_repo.empty()) {
        std::cerr << "Error: missing HuggingFace repo" << std::endl;
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
            std::cerr << "Error: can't find export script at scripts/export_ternary_model_bitnet.py" << std::endl;
            return 1;
        }
    }

    std::string model_slug = slugify(hf_repo);
    std::string out_dir = std::string(getenv("HOME") ? getenv("HOME") : "/root")
                        + "/.terllama/models/" + model_slug;

    std::cout << "Downloading " << hf_repo << " from HuggingFace..." << std::endl;
    std::cout << "Converting to " << format << " format..." << std::endl;
    std::cout << "Output: " << out_dir << std::endl;
    std::cout << std::endl;

    // Build command
    std::string cmd = "pip install transformers torch -q 2>/dev/null && "
                    + std::string("python3 \"") + script_path + "\""
                    + " --model \"" + hf_repo + "\""
                    + " --outdir \"" + out_dir + "\""
                    + " --format " + format;

    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "Export failed (exit code " << ret << ")" << std::endl;
        return 1;
    }

    std::cout << "\nModel downloaded to: " << out_dir << std::endl;
    return 0;
}

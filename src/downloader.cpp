/*
 * downloader.cpp — HuggingFace model downloader for Terllama
 *
 * Downloads model files from HuggingFace using wget/curl.
 * Progress bar, resume support, SHA256 verification.
 *
 * Usage (via terllama CLI):
 *   terllama pull HuggingFaceTB/SmolLM2-135M --format i2s
 *   terllama pull HuggingFaceTB/SmolLM2-135M --format als
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>
#include <pwd.h>

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

static std::string home_dir() {
    const char* h = getenv("HOME");
    if (h) return h;
    struct passwd* pw = getpwuid(getuid());
    return pw ? pw->pw_dir : "/root";
}

std::string get_model_dir(const std::string& hf_repo) {
    std::string s = hf_repo;
    for (auto& c : s)
        if (c == '/' || c == '\\' || c == ' ') c = '-';
    return home_dir() + "/.terllama/models/" + s;
}

static bool ensure_dir(const std::string& path) {
    size_t pos = 0;
    std::string p;
    while ((pos = path.find('/', pos)) != std::string::npos) {
        p = path.substr(0, pos++);
        if (p.empty()) continue;
        mkdir(p.c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static long long file_size(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return -1;
    return st.st_size;
}

static std::string fmt_size(double bytes) {
    char buf[32];
    if (bytes < 1024.0)
        snprintf(buf, sizeof(buf), "%.0f B", bytes);
    else if (bytes < 1024.0 * 1024.0)
        snprintf(buf, sizeof(buf), "%.1f KB", bytes / 1024.0);
    else
        snprintf(buf, sizeof(buf), "%.1f MB", bytes / (1024.0 * 1024.0));
    return buf;
}

static std::string timestamp() {
    time_t now = time(nullptr);
    struct tm* tm = gmtime(&now);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", tm);
    return buf;
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON TRACKING (~/.terllama/models.json)
// ═══════════════════════════════════════════════════════════════════════════

static std::string json_path() {
    return home_dir() + "/.terllama/models.json";
}

static std::string json_escape_str(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

struct ModelEntry {
    std::string id, format, path, downloaded;
    long long size{0};
};

static bool update_models_json(const ModelEntry& entry) {
    std::string jpath = json_path();
    std::string dir = home_dir() + "/.terllama";
    ensure_dir(dir);

    std::vector<ModelEntry> entries;
    std::ifstream inf(jpath);
    if (inf) {
        std::string content((std::istreambuf_iterator<char>(inf)),
                             std::istreambuf_iterator<char>());
        size_t pos = 0;
        while (true) {
            auto start = content.find("{\"id\"", pos);
            if (start == std::string::npos) break;
            auto end = content.find("}", start);
            if (end == std::string::npos) break;
            std::string block = content.substr(start, end - start + 1);

            auto grab = [&](const std::string& key) -> std::string {
                auto p = block.find("\"" + key + "\":\"");
                if (p == std::string::npos) return "";
                p += key.size() + 5;
                auto q = block.find("\"", p);
                return (q == std::string::npos) ? "" : block.substr(p, q - p);
            };
            auto grab_num = [&](const std::string& key) -> long long {
                auto p = block.find("\"" + key + "\":");
                if (p == std::string::npos) return 0;
                p += key.size() + 3;
                auto q = block.find_first_of(",}", p);
                return (q == std::string::npos) ? 0 : std::stoll(block.substr(p, q - p));
            };

            bool dup = (grab("id") == entry.id && grab("format") == entry.format);
            if (!dup)
                entries.push_back({grab("id"), grab("format"), grab("path"),
                                   grab("downloaded"), grab_num("size")});
            pos = end + 1;
        }
    }
    entries.push_back(entry);

    std::ofstream of(jpath);
    if (!of) return false;
    of << "{\n  \"models\": [\n";
    for (size_t i = 0; i < entries.size(); i++) {
        auto& e = entries[i];
        of << "    {\n";
        of << "      \"id\": \""   << json_escape_str(e.id)        << "\",\n";
        of << "      \"format\": \"" << json_escape_str(e.format)    << "\",\n";
        of << "      \"path\": \""  << json_escape_str(e.path)      << "\",\n";
        of << "      \"size\": "    << e.size                       << ",\n";
        of << "      \"downloaded\": \"" << json_escape_str(e.downloaded) << "\"\n";
        of << "    }";
        if (i < entries.size() - 1) of << ",";
        of << "\n";
    }
    of << "  ]\n}\n";
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// DOWNLOAD FILE (using wget or curl CLI)
// ═══════════════════════════════════════════════════════════════════════════

bool download_file(const std::string& url, const std::string& dest_path) {
    std::string dir = dest_path;
    auto slash = dir.rfind('/');
    if (slash != std::string::npos) {
        dir = dir.substr(0, slash);
        if (!ensure_dir(dir)) {
            std::cerr << "Error: cannot create directory " << dir << std::endl;
            return false;
        }
    }

    // Prefer wget for built-in progress bar and resume
    // Use wget if available
    std::string cmd;
    bool use_wget = (system("which wget >/dev/null 2>&1") == 0);

    if (use_wget) {
        // wget with progress bar, resume (-c), timeout, retries
        cmd = "wget -c --progress=bar:force --timeout=30 --tries=3";
        cmd += " -O \"" + dest_path + ".part\" \"" + url + "\"";
    } else {
        // curl fallback with progress
        long long partial = file_size(dest_path + ".part");
        cmd = "curl -L --connect-timeout 30 --max-time 7200";
        if (partial > 0) {
            cmd += " -C " + std::to_string(partial);
        }
        cmd += " -o \"" + dest_path + ".part\"";
        cmd += " \"" + url + "\"";
    }

    int ret = system(cmd.c_str());
    if (ret != 0) {
        std::cerr << "\nDownload failed (exit code " << ret << ")" << std::endl;
        return false;
    }

    // Rename .part → final
    if (rename((dest_path + ".part").c_str(), dest_path.c_str()) != 0) {
        std::cerr << "Error: cannot rename " << dest_path << ".part" << std::endl;
        return false;
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// SHA256 VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════

bool verify_sha256(const std::string& path, const std::string& expected_sha256) {
    if (expected_sha256.empty()) return true;

    std::string cmd = "sha256sum \"" + path + "\" 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return true;

    char buf[128] = {0};
    if (!fgets(buf, sizeof(buf), pipe)) { pclose(pipe); return true; }
    pclose(pipe);

    std::string actual;
    for (int i = 0; buf[i] && i < 64; i++) actual += buf[i];

    std::string exp = expected_sha256;
    for (auto& c : exp) c = tolower(c);
    for (auto& c : actual) c = tolower(c);

    if (actual != exp) {
        std::cerr << "SHA256 MISMATCH" << std::endl;
        std::cerr << "  Expected: " << exp << std::endl;
        std::cerr << "  Actual:   " << actual << std::endl;
        return false;
    }
    std::cerr << "SHA256 OK" << std::endl;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// DOWNLOAD MODEL FROM HUGGINGFACE
// ═══════════════════════════════════════════════════════════════════════════

std::string download_model(const std::string& hf_repo, const std::string& format) {
    std::string filename;
    if (format == "i2s")
        filename = "model_decomposed_i2s.bin";
    else if (format == "als")
        filename = "model_decomposed.bin";
    else {
        std::cerr << "Error: unknown format '" << format << "' (use 'i2s' or 'als')" << std::endl;
        return "";
    }

    std::string model_dir = get_model_dir(hf_repo);
    std::string dest_path = model_dir + "/" + filename;

    struct stat st;
    if (stat(dest_path.c_str(), &st) == 0 && st.st_size > 0) {
        std::cerr << "Model already exists: " << dest_path
                  << " (" << fmt_size((double)st.st_size) << ")" << std::endl;
        return dest_path;
    }

    std::string url = "https://huggingface.co/" + hf_repo
                    + "/resolve/main/" + filename;

    std::cerr << "Downloading:" << std::endl;
    std::cerr << "  Repo:  " << hf_repo << std::endl;
    std::cerr << "  File:  " << filename << std::endl;
    std::cerr << "  URL:   " << url << std::endl;
    std::cerr << "  To:    " << dest_path << std::endl;

    // Also download model_extra.bin (needed for config, embedding, norms)
    std::string extra_url = "https://huggingface.co/" + hf_repo
                          + "/resolve/main/model_extra.bin";
    std::string extra_path = model_dir + "/model_extra.bin";

    if (stat(extra_path.c_str(), &st) != 0 || st.st_size <= 0) {
        std::cerr << "\n-- model_extra.bin --" << std::endl;
        if (!download_file(extra_url, extra_path)) {
            std::cerr << "Warning: failed to download model_extra.bin" << std::endl;
            std::cerr << "  (required for config/embedding/norms)" << std::endl;
        } else {
            std::cerr << "  Done: " << extra_path << std::endl;
        }
    }

    std::cerr << "\n-- " << filename << " --" << std::endl;
    if (!download_file(url, dest_path)) {
        std::cerr << "Download failed" << std::endl;
        return "";
    }

    long long actual_size = file_size(dest_path);
    std::cerr << "  Done: " << dest_path << " (" << fmt_size((double)actual_size) << ")" << std::endl;

    // Update tracking
    update_models_json({hf_repo, format, dest_path, timestamp(), actual_size});
    return dest_path;
}

// ═══════════════════════════════════════════════════════════════════════════
// CLI MAIN
// ═══════════════════════════════════════════════════════════════════════════

static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " download <hf_repo> [--format i2s|als]" << std::endl;
    std::cerr << "\nDownload a model from HuggingFace." << std::endl;
    std::cerr << "\nArguments:" << std::endl;
    std::cerr << "  <hf_repo>    HuggingFace repo (e.g. HuggingFaceTB/SmolLM2-135M)" << std::endl;
    std::cerr << "  --format     'i2s' (default) or 'als'" << std::endl;
    std::cerr << "\nModels stored in ~/.terllama/models/<repo-name>/" << std::endl;
    std::cerr << "Tracked in ~/.terllama/models.json" << std::endl;
}

int downloader_main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    std::string cmd = argv[2];
    if (cmd != "download") { print_usage(argv[0]); return 1; }

    std::string hf_repo;
    std::string format = "i2s";

    for (int i = 3; i < argc; i++) {
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

    std::string result = download_model(hf_repo, format);
    if (result.empty()) {
        std::cerr << "Download failed." << std::endl;
        return 1;
    }

    std::cout << "Model downloaded to: " << result << std::endl;
    return 0;
}

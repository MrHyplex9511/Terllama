/*
 * hf_download.cpp — Native HuggingFace file download via libcurl.
 *
 * GET https://huggingface.co/{repo_id}/resolve/main/{filename}
 * Follows CDN redirects, verifies TLS (no self-signed), 300s timeout,
 * fails on non-2xx. Atomic write: temp file + rename.
 */
#include "convert/hf_download.h"

#include "core/logger.h"
#include <json.hpp>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <unistd.h>

#if defined(TERLLAMA_HAVE_CURL)
#include <curl/curl.h>
#endif

using json = nlohmann::json;

namespace terllama {

// ── URL encoding (RFC 3986 unreserved chars pass through) ──────────────────
std::string hf_url_encode(const std::string& s) {
    static const char* hex = "0123456789ABCDEF";
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
            c == '~' || c == '/') {
            out += static_cast<char>(c);
        } else {
            out += '%';
            out += hex[c >> 4];
            out += hex[c & 0x0F];
        }
    }
    return out;
}

#if !defined(TERLLAMA_HAVE_CURL)
// Stubs: this TU was built without libcurl (CMake: CURL_FOUND false).
// Downloads fail with a clear error; callers should fall back elsewhere.

int hf_download_file(const std::string&, const std::string&, const std::string&,
                     const std::string&) {
    Logger::error("hf_download: built without libcurl (TERLLAMA_HAVE_CURL "
                  "undefined) — cannot download{}", "");
    return 1;
}

int hf_download_sharded(const std::string&, const std::string&,
                        const std::string&, std::vector<std::string>&) {
    Logger::error("hf_download_sharded: built without libcurl "
                  "(TERLLAMA_HAVE_CURL undefined){}", "");
    return 1;
}

#else // TERLLAMA_HAVE_CURL

namespace {

size_t write_to_file(char* ptr, size_t size, size_t nmemb, void* userdata) {
    FILE* f = static_cast<FILE*>(userdata);
    return fwrite(ptr, size, nmemb, f);
}

} // namespace

int hf_download_file(const std::string& repo_id,
                     const std::string& filename,
                     const std::string& dest_dir,
                     const std::string& hf_token) {
    if (repo_id.empty() || filename.empty() || dest_dir.empty()) {
        Logger::error("hf_download: empty repo_id/filename/dest_dir "
                      "(repo='{}', file='{}', dest='{}')", repo_id, filename,
                      dest_dir);
        return 1;
    }

    // dest_dir must exist.
    struct stat st;
    if (stat(dest_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        Logger::error("hf_download: dest_dir '{}' does not exist", dest_dir);
        return 1;
    }

    const std::string url =
        "https://huggingface.co/" + hf_url_encode(repo_id) + "/resolve/main/" +
        hf_url_encode(filename);

    // Reject path components that could escape dest_dir (safety).
    if (filename.find("..") != std::string::npos ||
        filename.find('/') != std::string::npos) {
        Logger::error("hf_download: filename '{}' must be a bare file name",
                      filename);
        return 1;
    }

    const std::string final_path = dest_dir + "/" + filename;
    const std::string tmp_path =
        dest_dir + "/." + filename + ".tmp." + std::to_string(static_cast<long>(getpid()));

    FILE* f = fopen(tmp_path.c_str(), "wb");
    if (!f) {
        Logger::error("hf_download: cannot open temp file '{}'", tmp_path);
        return 1;
    }

    CURL* curl = curl_easy_init();
    if (!curl) {
        fclose(f);
        unlink(tmp_path.c_str());
        Logger::error("hf_download: curl_easy_init failed{}", " (OOM)");
        return 1;
    }

    long http_code = 0;
    CURLcode rc = CURLE_OK;
    bool ok = false;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT,
                     "terllama/0.1 (native hf downloader)");
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_to_file);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
    // TLS: verify peer + host against system CAs; do NOT accept self-signed.
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 1L);
    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 2L);
    if (!hf_token.empty()) {
        const std::string auth = "Authorization: Bearer " + hf_token;
        struct curl_slist* headers = nullptr;
        headers = curl_slist_append(headers, auth.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        rc = curl_easy_perform(curl);
        curl_slist_free_all(headers);
    } else {
        rc = curl_easy_perform(curl);
    }

    if (rc == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
        if (http_code >= 200 && http_code < 300) {
            ok = true;
        } else {
            Logger::error("hf_download: HTTP {} for {}", http_code, url);
        }
    } else {
        Logger::error("hf_download: curl error {} for {}",
                      curl_easy_strerror(rc), url);
    }

    fclose(f);
    curl_easy_cleanup(curl);

    if (ok) {
        if (rename(tmp_path.c_str(), final_path.c_str()) != 0) {
            Logger::error("hf_download: rename '{}' -> '{}' failed: {}",
                          tmp_path, final_path, strerror(errno));
            unlink(tmp_path.c_str());
            return 1;
        }
        struct stat fst;
        long size = (stat(final_path.c_str(), &fst) == 0) ? (long)fst.st_size : -1;
        Logger::info("hf_download: '{}' -> {} ({} bytes)", url, final_path,
                     size);
        return 0;
    }

    unlink(tmp_path.c_str());
    return 1;
}

// ── Sharded model support ──────────────────────────────────────────────────

bool should_use_index(const std::string& config_json) {
    if (config_json.empty()) return false;
    try {
        json cfg = json::parse(config_json);
        // True if any string value in the config (at any depth) equals the
        // index file name — e.g. "model_index": "model.safetensors.index.json"
        // as written by transformers for sharded repos.
        std::function<bool(const json&)> walk = [&](const json& node) -> bool {
            if (node.is_object()) {
                for (auto it = node.begin(); it != node.end(); ++it) {
                    if (walk(it.value())) return true;
                }
            } else if (node.is_array()) {
                for (const auto& el : node) {
                    if (walk(el)) return true;
                }
            } else if (node.is_string() &&
                       node.get<std::string>() ==
                           "model.safetensors.index.json") {
                return true;
            }
            return false;
        };
        return walk(cfg);
    } catch (const std::exception& e) {
        Logger::warn("should_use_index: config parse failed: {}", e.what());
        return false;
    }
}

int hf_download_sharded(const std::string& repo_id,
                        const std::string& dest_dir,
                        const std::string& hf_token,
                        std::vector<std::string>& out_files) {
    out_files.clear();

    const std::string index_name = "model.safetensors.index.json";
    if (hf_download_file(repo_id, index_name, dest_dir, hf_token) != 0) {
        Logger::error("hf_download_sharded: failed to fetch index '{}'",
                      index_name);
        return 1;
    }

    // Read the index JSON.
    const std::string index_path = dest_dir + "/" + index_name;
    FILE* f = fopen(index_path.c_str(), "rb");
    if (!f) {
        Logger::error("hf_download_sharded: cannot open index '{}'",
                      index_path);
        return 1;
    }
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len < 0) {
        fclose(f);
        return 1;
    }
    std::string body(static_cast<size_t>(len), '\0');
    size_t got = fread(&body[0], 1, body.size(), f);
    fclose(f);
    if (got != body.size()) {
        Logger::error("hf_download_sharded: short read on index (need {}, got {})", body.size(), got);
        return 1;
    }

    json idx;
    try {
        idx = json::parse(body);
    } catch (const std::exception& e) {
        Logger::error("hf_download_sharded: index JSON parse failed: {}",
                      e.what());
        return 1;
    }
    if (!idx.contains("weight_map") || !idx["weight_map"].is_object()) {
        Logger::error("hf_download_sharded: index has no weight_map{}", "");
        return 1;
    }

    // Collect unique shard file names (preserving first-seen order).
    std::vector<std::string> shards;
    for (auto it = idx["weight_map"].begin(); it != idx["weight_map"].end(); ++it) {
        if (!it.value().is_string()) continue;
        const std::string fname = it.value().get<std::string>();
        bool seen = false;
        for (const auto& s : shards) {
            if (s == fname) { seen = true; break; }
        }
        if (!seen) shards.push_back(fname);
    }

    if (shards.empty()) {
        Logger::error("hf_download_sharded: weight_map lists no shards{}", "");
        return 1;
    }

    for (const auto& shard : shards) {
        if (hf_download_file(repo_id, shard, dest_dir, hf_token) != 0) {
            Logger::error("hf_download_sharded: failed shard '{}'", shard);
            return 1;
        }
        out_files.push_back(dest_dir + "/" + shard);
    }
    Logger::info("hf_download_sharded: {} shards downloaded for {}",
                 shards.size(), repo_id);
    return 0;
}

#endif // TERLLAMA_HAVE_CURL

} // namespace terllama

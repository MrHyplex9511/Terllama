/*
 * worker.cpp — Terllama distributed-inference worker implementation.
 */
#include "worker.h"

#include <sys/stat.h>

#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <vector>

#include <json.hpp>

namespace tldist {

namespace {

// Linux: read MemAvailable from /proc/meminfo (kB -> bytes). Fallback 0.
int64_t read_ram_available() {
#if defined(__linux__)
    std::ifstream f("/proc/meminfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 13, "MemAvailable:") == 0) {
            long long kb = std::atoll(line.c_str() + 13);
            return kb * 1024;
        }
    }
#else
    (void)0;
#endif
    return 0;
}

// Sum of model_extra.bin + model_decomposed.bin sizes; falls back to the
// GGUF file when no .bin files exist (GGUF models).
int64_t model_size_bytes(const std::string& model_dir) {
    int64_t total = 0;
    struct stat st;
    std::string extra = model_dir + "/model_extra.bin";
    std::string decomp = model_dir + "/model_decomposed.bin";
    if (stat(extra.c_str(), &st) == 0) total += (int64_t)st.st_size;
    if (stat(decomp.c_str(), &st) == 0) total += (int64_t)st.st_size;
    if (total == 0) {
        std::string gguf = model_path_for(model_dir);
        if (!gguf.empty() && stat(gguf.c_str(), &st) == 0) total = (int64_t)st.st_size;
    }
    return total;
}

}  // namespace

// ─── Lifecycle ─────────────────────────────────────────────────────────────

Worker::Worker(std::string listen_addr) : listen_addr_(std::move(listen_addr)) {
    svr_.set_read_timeout(300, 0);
    svr_.set_write_timeout(300, 0);
    svr_.set_payload_max_length(512LL * 1024 * 1024);
}

void Worker::stop() { svr_.stop(); }

void Worker::load_model(const std::string& model_path, const ShardSpec& spec) {
    // Load outside the lock (can take seconds), swap under it.
    auto m = std::make_shared<ShardModel>(load_model_shard(model_path, spec));
    std::lock_guard<std::mutex> lk(mu_);
    loaded_ = std::move(m);
    model_dir_ = model_path;
    Logger::info("terllama-worker: model loaded from {}", model_path);
}

int Worker::run() {
    register_routes();
    auto pos = listen_addr_.rfind(':');
    std::string host = (pos == std::string::npos) ? listen_addr_ : listen_addr_.substr(0, pos);
    std::string port_str = (pos == std::string::npos) ? "9100" : listen_addr_.substr(pos + 1);
    int port = std::atoi(port_str.c_str());
    if (port <= 0 || port > 65535) {
        Logger::error("bad listen port: {}", port_str);
        return 1;
    }
    Logger::info("terllama-worker listening on {}:{}", host, port);
    if (!svr_.listen(host.c_str(), port)) {
        Logger::error("listen failed on {}:{}", host, port);
        return 1;
    }
    return 0;
}

// ─── Routes ────────────────────────────────────────────────────────────────

void Worker::register_routes() {
    svr_.Get(kHealthPath, [this](const httplib::Request&, httplib::Response& res) {
        handle_health(res);
    });
    svr_.Post(kLoadPath, [this](const httplib::Request& req, httplib::Response& res) {
        handle_load(req, res);
    });
    svr_.Post(kForwardPath, [this](const httplib::Request& req, httplib::Response& res) {
        handle_forward(req, res);
    });
    svr_.Post(kResetPath, [this](const httplib::Request&, httplib::Response& res) {
        handle_reset(res);
    });
    svr_.set_exception_handler([](const httplib::Request&, httplib::Response& res,
                                  std::exception_ptr ep) {
        std::string msg = "unhandled exception";
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            msg = e.what();
        }
        res.status = 500;
        nlohmann::json j = {{"ok", false}, {"error", msg}};
        res.set_content(j.dump(), "application/json");
    });
}

void Worker::handle_health(httplib::Response& res) {
    HealthInfo h;
    std::lock_guard<std::mutex> lk(mu_);
    h.ok = true;
    h.model_loaded = (loaded_ != nullptr);
    h.model_dir = model_dir_;
    if (loaded_) h.shard = loaded_->spec;
    h.ram_available_bytes = read_ram_available();
    h.model_size_bytes = model_size_bytes(model_dir_);
    nlohmann::json j;
    to_json(j, h);
    res.set_content(j.dump(), "application/json");
}

void Worker::handle_load(const httplib::Request& req, httplib::Response& res) {
    LoadRequest lr;
    try {
        auto j = nlohmann::json::parse(req.body);
        from_json(j, lr);
    } catch (const std::exception& e) {
        res.status = 400;
        nlohmann::json j = {{"ok", false}, {"error", std::string("bad load request: ") + e.what()}};
        res.set_content(j.dump(), "application/json");
        return;
    }
    try {
        load_model(lr.model_path, lr.shard);
    } catch (const std::exception& e) {
        res.status = 500;
        nlohmann::json j = {{"ok", false}, {"error", std::string("load failed: ") + e.what()}};
        res.set_content(j.dump(), "application/json");
        return;
    }
    nlohmann::json j = {{"ok", true}, {"error", ""}};
    res.set_content(j.dump(), "application/json");
}

void Worker::handle_forward(const httplib::Request& req, httplib::Response& res) {
    int seq_pos = 0;
    uint32_t input_kind = 0;
    std::vector<float> data;
    if (!unpack_forward_request(reinterpret_cast<const uint8_t*>(req.body.data()),
                                req.body.size(), seq_pos, input_kind, data)) {
        res.status = 400;
        nlohmann::json j = {{"ok", false}, {"error", "malformed forward request"}};
        res.set_content(j.dump(), "application/json");
        return;
    }

    std::vector<float> out;
    try {
        std::lock_guard<std::mutex> lk(mu_);
        if (!loaded_) throw std::runtime_error("no model loaded");
        ShardModel& m = *loaded_;
        const ModelConfig& cfg = m.cfg;
        const ShardSpec& spec = m.spec;

        std::vector<float> hidden((size_t)cfg.hidden_size);
        if (spec.is_first && input_kind == kInputToken) {
            if (data.empty()) throw std::runtime_error("token input missing");
            int token = (int)data[0];
            if (token < 0 || token >= cfg.vocab_size)
                throw std::runtime_error("token id out of range");
            std::copy(&m.embedding[(size_t)token * cfg.hidden_size],
                      &m.embedding[(size_t)(token + 1) * cfg.hidden_size],
                      hidden.data());
        } else if (input_kind == kInputHidden) {
            if (data.size() != (size_t)cfg.hidden_size)
                throw std::runtime_error("hidden state size mismatch");
            std::copy(data.begin(), data.end(), hidden.begin());
        } else {
            res.status = 400;
            nlohmann::json j = {{"ok", false},
                                {"error", "input_kind not valid for this shard"}};
            res.set_content(j.dump(), "application/json");
            return;
        }

        for (int i = spec.start_layer; i < spec.end_layer; i++) {
            transformer_block(hidden.data(), seq_pos, i, cfg, m.layers,
                              m.layer_norms[i], m.rope, m.kv);
        }

        if (spec.is_last) {
            rms_norm(hidden.data(), m.final_norm.data(), cfg.hidden_size,
                     cfg.rms_norm_eps);
            int idx = find_layer_index(m.layers, "lm_head");
            out.assign((size_t)cfg.vocab_size, 0.0f);
            ternary_linear_dispatch(m.layers[idx], hidden.data(), out.data());
        } else {
            out = std::move(hidden);
        }
    } catch (const std::exception& e) {
        res.status = 500;
        nlohmann::json j = {{"ok", false},
                            {"error", std::string("forward failed: ") + e.what()}};
        res.set_content(j.dump(), "application/json");
        return;
    }

    auto payload = pack_forward_response(out);
    res.set_content(reinterpret_cast<const char*>(payload.data()), payload.size(),
                    "application/octet-stream");
}

void Worker::handle_reset(httplib::Response& res) {
    std::lock_guard<std::mutex> lk(mu_);
    if (loaded_) {
        const ModelConfig& cfg = loaded_->cfg;
        loaded_->kv = KVCache(cfg.max_position_embeddings, cfg.num_hidden_layers,
                              cfg.num_key_value_heads, cfg.head_dim,
                              cfg.hidden_size);
    }
    nlohmann::json j = {{"ok", true}};
    res.set_content(j.dump(), "application/json");
}

}  // namespace tldist

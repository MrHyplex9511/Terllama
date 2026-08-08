/*
 * coordinator.cpp — Terllama distributed inference coordinator.
 *
 * Orchestrates pipeline-parallel token generation across N workers that each
 * expose the Terllama distributed HTTP RPC (distributed/protocol.h), and
 * serves an OpenAI-compatible API (/v1/models, /v1/completions,
 * /v1/chat/completions) with optional SSE streaming.
 */
#include "coordinator.h"

#include "inference.h"
#include "loader.h"
#include "core/logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <csignal>
#include <cstdlib>
#include <dirent.h>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════════════
// Signal / shutdown plumbing (async-safe flag + watchdog thread)
// ═══════════════════════════════════════════════════════════════════════════

static std::atomic<bool> g_stop_requested{false};
static httplib::Server* g_server_ptr = nullptr;

extern "C" void cluster_sigint_handler(int) {
    g_stop_requested.store(true, std::memory_order_relaxed);
}

// ═══════════════════════════════════════════════════════════════════════════
// Small static helpers
// ═══════════════════════════════════════════════════════════════════════════

// Chat message (local mirror of handlers.h Message — avoids linking handlers).
struct ChatMessage {
    std::string role;
    std::string content;
};

namespace {

std::string normalize_url(const std::string& url) {
    if (url.empty())
        throw std::runtime_error("empty worker URL");
    if (url.rfind("http://", 0) == 0 || url.rfind("https://", 0) == 0)
        return url;
    return "http://" + url;
}

std::string basename_of(const std::string& path) {
    std::string p = path;
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    auto slash = p.rfind('/');
    std::string name = (slash == std::string::npos) ? p : p.substr(slash + 1);
    if (name.empty()) name = "terllama";
    return name;
}

// Local estimate of model size: sum of regular file sizes under model_path.
int64_t estimate_model_size_bytes(const std::string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode))
        return (int64_t)st.st_size;
    int64_t total = 0;
    DIR* d = opendir(path.c_str());
    if (!d) return 0;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
        if (e->d_name[0] == '.') continue;
        std::string full = path + "/" + e->d_name;
        if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode))
            total += (int64_t)st.st_size;
    }
    closedir(d);
    return total;
}

void cluster_cors(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin", "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
}

void cluster_send_json(httplib::Response& res, const std::string& body, int status = 200) {
    res.status = status;
    res.set_content(body, "application/json");
}

void cluster_send_error(httplib::Response& res, const std::string& message,
                        int status, const std::string& type) {
    cluster_send_json(res,
        json{{"error", json{{"message", message}, {"type", type}}}}.dump(),
        status);
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════════════

Coordinator::Coordinator(std::vector<std::string> worker_urls,
                         std::string model_path, std::string api_port)
    : model_path_(std::move(model_path)),
      api_port_(std::move(api_port)),
      model_name_(basename_of(model_path_)) {
    if (worker_urls.empty())
        throw std::runtime_error("Coordinator: no workers configured");
    workers_.reserve(worker_urls.size());
    for (const auto& u : worker_urls) {
        ClusterWorker w;
        w.url = normalize_url(u);
        w.client = std::make_shared<httplib::Client>(w.url);
        w.client->set_connection_timeout(5, 0);
        w.client->set_read_timeout(300, 0);
        w.client->set_write_timeout(300, 0);
        workers_.push_back(std::move(w));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Worker comms
// ═══════════════════════════════════════════════════════════════════════════

tldist::HealthInfo Coordinator::get_health(size_t idx) {
    auto res = workers_[idx].client->Get(tldist::kHealthPath);
    if (!res)
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " health request failed: " +
                                 httplib::to_string(res.error()));
    if (res->status != 200)
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " health HTTP " + std::to_string(res->status));
    tldist::HealthInfo h;
    try {
        json j = json::parse(res->body);
        h.ok = j.value("ok", false);
        h.model_loaded = j.value("model_loaded", false);
        h.model_dir = j.value("model_dir", std::string());
        h.ram_available_bytes = j.value("ram_available_bytes", (int64_t)0);
        h.model_size_bytes = j.value("model_size_bytes", (int64_t)0);
        h.error = j.value("error", std::string());
        if (j.contains("shard"))
            h.shard = j.at("shard").get<tldist::ShardSpec>();
    } catch (const std::exception& e) {
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " bad health JSON: " + e.what());
    }
    return h;
}

void Coordinator::post_load(size_t idx, const tldist::LoadRequest& req) {
    // to_json(ShardSpec) is picked up by nlohmann via ADL.
    json j{{"model_path", req.model_path}, {"shard", req.shard}};
    auto res = workers_[idx].client->Post(tldist::kLoadPath, j.dump(),
                                          "application/json");
    if (!res)
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " load request failed: " +
                                 httplib::to_string(res.error()));
    if (res->status != 200)
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " load HTTP " + std::to_string(res->status));
    bool ok = false;
    try {
        ok = json::parse(res->body).value("ok", false);
    } catch (...) { ok = false; }
    if (!ok)
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " refused to load model: " + res->body);
}

void Coordinator::post_reset(size_t idx) {
    auto res = workers_[idx].client->Post(tldist::kResetPath, "", "application/json");
    if (!res)
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " reset request failed: " +
                                 httplib::to_string(res.error()));
    if (res->status != 200)
        throw std::runtime_error("worker " + std::to_string(idx) +
                                 " reset HTTP " + std::to_string(res->status));
}

// ═══════════════════════════════════════════════════════════════════════════
// Pipeline
// ═══════════════════════════════════════════════════════════════════════════

std::vector<float> Coordinator::pipeline(int seq_pos,
                                         const std::vector<float>& input,
                                         uint32_t input_kind) {
    std::vector<float> cur = input;
    for (size_t w = 0; w < workers_.size(); w++) {
        std::vector<uint8_t> body =
            tldist::pack_forward_request(seq_pos, input_kind, cur);
        auto res = workers_[w].client->Post(
            tldist::kForwardPath, (const char*)body.data(), body.size(),
            "application/octet-stream");
        if (!res)
            throw std::runtime_error("worker " + std::to_string(w) +
                                     " forward failed: " +
                                     httplib::to_string(res.error()));
        if (res->status != 200)
            throw std::runtime_error("worker " + std::to_string(w) +
                                     " forward HTTP " +
                                     std::to_string(res->status));
        std::vector<float> out;
        if (!tldist::unpack_forward_response(
                reinterpret_cast<const uint8_t*>(res->body.data()),
                res->body.size(), out))
            throw std::runtime_error("worker " + std::to_string(w) +
                                     " bad forward response (magic/size)");
        cur = std::move(out);
        input_kind = tldist::kInputHidden;  // ranks > 0 take hidden state
    }
    return cur;  // hidden state (middle) or logits (last worker)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tokenizer
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> Coordinator::tokenize(const std::string& text) const {
    if (gigatoken_ && gigatoken_->has_tokenizer()) {
        auto ids = gigatoken_->encode(text);
        return std::vector<int>(ids.begin(), ids.end());
    }
    throw std::runtime_error(
        "Tokenizer unavailable: no GigaToken tokenizer.json loaded");
}

// Mirror of decode_with_fallback (src/server/handlers.cpp): native llama
// decode first, then GigaToken; never a Python subprocess.
std::string Coordinator::decode(const std::vector<int>& ids) const {
    std::string native = tokenizer_.decode(ids);
    if (!native.empty() && native != "?") return native;
    if (gigatoken_ && gigatoken_->has_tokenizer()) {
        std::vector<uint32_t> uids(ids.begin(), ids.end());
        std::string text = gigatoken_->decode(uids);
        if (!text.empty()) return text;
    }
    return "";
}

std::string Coordinator::make_id(const char* prefix) {
    static std::atomic<uint64_t> counter{0};
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    uint64_t rand_part = rng();
    uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);
    char buf[64];
    snprintf(buf, sizeof(buf), "%s-%llx%llx", prefix,
             static_cast<unsigned long long>(rand_part),
             static_cast<unsigned long long>(seq));
    return buf;
}

long Coordinator::now_ts() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ═══════════════════════════════════════════════════════════════════════════
// Generation core
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> Coordinator::generate(const std::vector<int>& prompt_tokens,
                                       int max_tokens, float temperature,
                                       const std::vector<int>& prev_tokens,
                                       std::function<bool(int)> on_token) {
    // v1: single stream at a time.
    std::lock_guard<std::mutex> lock(gen_mutex_);

    if (prompt_tokens.empty() || max_tokens <= 0) return {};

    const int n_prompt = (int)prompt_tokens.size();
    const int max_seq = cfg_.max_position_embeddings;
    if (n_prompt >= max_seq) return {};           // no room for a generated token
    if (max_tokens > max_seq - n_prompt) max_tokens = max_seq - n_prompt;
    if (max_tokens <= 0) return {};

    // Fresh request: clear KV on every worker.
    for (size_t i = 0; i < workers_.size(); i++) post_reset(i);

    // Prefill: walk the pipeline once per prompt token (same seq_pos per
    // worker per step). Logits are discarded.
    for (int pos = 0; pos < n_prompt; pos++) {
        pipeline(pos, {float(prompt_tokens[(size_t)pos])}, tldist::kInputToken);
    }

    // Repeat-penalty context: tail of prev_tokens if given, else none.
    std::vector<int> recent;
    if (!prev_tokens.empty()) {
        for (int j = std::max(0, (int)prev_tokens.size() - 8);
             j < (int)prev_tokens.size(); j++)
            recent.push_back(prev_tokens[(size_t)j]);
    }

    const int eos = cfg_.eos_token_id;
    int next_token = prompt_tokens.back();

    // Autoregressive decode (mirrors generate_stream in src/core/inference.cpp).
    std::vector<int> output_tokens;
    for (int i = 0; i < max_tokens; i++) {
        int pos = n_prompt + i;  // global KV position for every worker
        std::vector<float> logits =
            pipeline(pos, {float(next_token)}, tldist::kInputToken);

        if (temperature < 0.01f)
            next_token = sample_argmax(logits.data(), cfg_.vocab_size);
        else
            next_token = sample_multinomial(logits.data(), cfg_.vocab_size,
                                            temperature, recent, 1.1f);

        output_tokens.push_back(next_token);
        recent.push_back(next_token);
        if ((int)recent.size() > 8) recent.erase(recent.begin());

        if (on_token && !on_token(next_token)) break;
        if (next_token == eos) break;
    }
    return output_tokens;
}

// ═══════════════════════════════════════════════════════════════════════════
// start(): health → config → shards → load → tokenizer → serve API
// ═══════════════════════════════════════════════════════════════════════════

bool Coordinator::start() {
    try {
        const size_t n = workers_.size();

        // 1. Health-check every worker; collect RAM + any pre-loaded shard.
        std::vector<int64_t> rams(n, 0);
        std::vector<tldist::ShardSpec> pre(n);
        bool all_preloaded = true;
        for (size_t i = 0; i < n; i++) {
            tldist::HealthInfo h = get_health(i);
            if (!h.ok)
                throw std::runtime_error("worker " + std::to_string(i) +
                                         " unhealthy: " + h.error);
            rams[i] = h.ram_available_bytes;
            pre[i] = h.shard;
            if (!h.model_loaded || h.shard.end_layer <= h.shard.start_layer)
                all_preloaded = false;
        }
        // Fallback: local model-size estimate when a worker reports no RAM.
        const int64_t local_model_size = estimate_model_size_bytes(model_path_);
        for (size_t i = 0; i < n; i++)
            if (rams[i] <= 0) rams[i] = local_model_size;

        // 2. Load the LOCAL config only (workers load their own weights).
        struct stat st;
        std::string extra = model_path_ + "/model_extra.bin";
        if (stat(extra.c_str(), &st) == 0) {
            cfg_ = load_config(extra);
        } else {
            // GGUF file (direct path or inside dir) — full load is expensive
            // but only the config is kept.
            cfg_ = load_model_from(model_path_).cfg;
        }

        // 3. Partition layers across workers. When every worker already loaded
        // a shard (e.g. spawned pre-sharded by the terllama-node daemon, which
        // partitioned by ADVERTISED RAM), adopt that plan verbatim — do NOT
        // recompute from live RAM, which double-loads weights and can abort
        // once the machine's real free memory is low. Standalone usage (bare
        // workers, no --shard) falls back to computing + assigning shards.
        std::vector<tldist::ShardSpec> shards;
        bool valid_plan = all_preloaded;
        if (valid_plan) {
            int expect = 0;
            for (size_t i = 0; i < n; i++) {
                if (pre[i].start_layer != expect) { valid_plan = false; break; }
                expect = pre[i].end_layer;
            }
            if (expect != cfg_.num_hidden_layers) valid_plan = false;
        }
        if (valid_plan) {
            shards = pre;
            Logger::info("terllama-cluster: adopting {} pre-loaded shards (node-daemon plan)",
                         n);
        } else {
            shards = tldist::compute_shards(
                cfg_.num_hidden_layers, (int)n, rams, local_model_size);
        }

        // 4. Assign shards to workers. Pre-loaded ones keep their slice (skip
        //    the redundant POST /load — re-loading 500MB+ is wasteful); only
        //    the standalone path pushes a load request.
        for (size_t i = 0; i < n; i++) {
            if (!valid_plan) {
                tldist::LoadRequest lr;
                lr.model_path = model_path_;
                lr.shard = shards[i];
                post_load(i, lr);
            }
            workers_[i].shard = shards[i];
        }

        // 5. Tokenizer: GigaToken (HF tokenizer.json) + native fallback.
        gigatoken_ = std::make_shared<GigaTokenWrapper>();
        {
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
                gigatoken_.reset();
            } else if (gigatoken_->load(so_paths)) {
                if (gigatoken_->load_tokenizer(model_path_)) {
                    Logger::info("GigaToken: loaded HF tokenizer from {} (vocab={})",
                                 model_path_ + "/tokenizer.json",
                                 gigatoken_->vocab_size());
                } else {
                    Logger::info("GigaToken: no tokenizer.json in model dir — native decode only");
                    gigatoken_.reset();
                }
            } else {
                Logger::info("GigaToken: .so not found — native decode only");
                gigatoken_.reset();
            }
        }
        std::string tok_json = model_path_ + "/tokenizer.json";
        if (stat(tok_json.c_str(), &st) == 0) {
            tokenizer_.load_from_tokenizer_json(tok_json);
            Logger::info("Native tokenizer loaded from {}", tok_json);
        }

        // 6. Serve the OpenAI-compatible API.
        httplib::Server svr;

        svr.Options(".*", [](const httplib::Request&, httplib::Response& res) {
            cluster_cors(res);
            res.status = 204;
        });

        // ── GET /v1/models ───────────────────────────────────────────────
        svr.Get("/v1/models", [this](const httplib::Request&, httplib::Response& res) {
            cluster_cors(res);
            json data = {
                {"id", model_name_},
                {"object", "model"},
                {"owned_by", "terllama-cluster"},
            };
            cluster_send_json(res,
                json{{"object", "list"}, {"data", {data}}}.dump());
        });

        // ── GET /health ──────────────────────────────────────────────────
        svr.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
            cluster_cors(res);
            cluster_send_json(res, json{
                {"status",  "ok"},
                {"workers", (int)workers_.size()},
                {"model",   model_name_},
                {"layers",  cfg_.num_hidden_layers},
            }.dump());
        });

        // ── POST /v1/completions ─────────────────────────────────────────
        svr.Post("/v1/completions", [this](const httplib::Request& req,
                                           httplib::Response& res) {
            cluster_cors(res);
            json body;
            try {
                body = json::parse(req.body);
            } catch (...) {
                return cluster_send_error(res, "Invalid JSON body", 400, "invalid_request");
            }
            std::string prompt;
            bool stream = false;
            float temperature = 0.7f;
            int max_tokens = 256;
            try {
                prompt = body.value("prompt", std::string());
                stream = body.value("stream", false);
                temperature = body.value("temperature", 0.7f);
                max_tokens = body.value("max_tokens", 256);
            } catch (const std::exception&) {
                return cluster_send_error(res, "Internal server error", 500, "internal_error");
            }
            if (prompt.empty())
                return cluster_send_error(res, "No prompt provided", 400, "invalid_request");
            if (max_tokens <= 0)
                return cluster_send_error(res, "max_tokens must be a positive integer", 400, "invalid_request");
            if (max_tokens > 4096) max_tokens = 4096;

            std::vector<int> prompt_tokens;
            try {
                prompt_tokens = tokenize(prompt);
            } catch (const std::exception& e) {
                return cluster_send_error(res, e.what(), 500, "tokenization_error");
            }
            if (prompt_tokens.empty())
                return cluster_send_error(res, "Tokenization failed", 500, "tokenization_error");
            int context_budget = cfg_.max_position_embeddings - (int)prompt_tokens.size();
            if (context_budget <= 0)
                return cluster_send_error(res, "Prompt too long: exceeds model context", 400, "invalid_request");
            if (max_tokens > context_budget) max_tokens = std::max(1, context_budget);
            if (temperature < 0.01f) temperature = 0.0f;

            const std::string id = make_id("cmpl");
            const long created = now_ts();

            if (stream) {
                res.status = 200;
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");
                res.set_header("Access-Control-Allow-Origin", "*");
                res.set_chunked_content_provider("text/event-stream",
                    [this, prompt_tokens, temperature, max_tokens, id, created](
                        size_t offset, httplib::DataSink& sink) -> bool {
                        if (offset > 0) return false;
                        std::vector<int> buf;
                        try {
                            generate(prompt_tokens, max_tokens, temperature, {},
                                [&](int tok) {
                                    buf.push_back(tok);
                                    bool is_eos = (tok == cfg_.eos_token_id);
                                    if (buf.size() >= 16 || is_eos) {
                                        std::string text = decode(buf);
                                        if (!text.empty()) {
                                            json choice = {
                                                {"index", 0},
                                                {"text", text},
                                                {"logprobs", nullptr},
                                                {"finish_reason", nullptr},
                                            };
                                            json chunk = {
                                                {"id", id},
                                                {"object", "text_completion"},
                                                {"created", std::to_string(created)},
                                                {"model", model_name_},
                                                {"choices", {choice}},
                                            };
                                            std::string sse = "data: " + chunk.dump() + "\n\n";
                                            if (!sink.write(sse.data(), sse.size()))
                                                return false;
                                        }
                                        buf.clear();
                                    }
                                    return true;
                                });
                        } catch (const std::exception& e) {
                            Logger::error("Streaming completion failed: {}", e.what());
                        }
                        if (!buf.empty()) {
                            std::string text = decode(buf);
                            if (!text.empty()) {
                                json choice = {
                                    {"index", 0}, {"text", text},
                                    {"logprobs", nullptr}, {"finish_reason", nullptr},
                                };
                                json chunk = {
                                    {"id", id}, {"object", "text_completion"},
                                    {"created", std::to_string(created)},
                                    {"model", model_name_}, {"choices", {choice}},
                                };
                                std::string sse = "data: " + chunk.dump() + "\n\n";
                                sink.write(sse.data(), sse.size());
                            }
                            buf.clear();
                        }
                        sink.write("data: [DONE]\n\n", 16);
                        sink.done();
                        return true;
                    });
                return;
            }

            // Non-streaming.
            std::vector<int> output_tokens;
            try {
                output_tokens = generate(prompt_tokens, max_tokens, temperature, {}, nullptr);
            } catch (const std::exception& e) {
                Logger::error("Completion failed: {}", e.what());
                return cluster_send_error(res,
                    std::string("worker failed: ") + e.what(), 502, "cluster_error");
            }

            std::vector<int> all_tokens = prompt_tokens;
            all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
            std::string decoded = decode(all_tokens);
            std::string prompt_decoded = decode(prompt_tokens);
            std::string generated_text;
            if (decoded.size() > prompt_decoded.size()) {
                generated_text = decoded.substr(prompt_decoded.size());
                while (!generated_text.empty() && generated_text[0] == ' ')
                    generated_text.erase(0, 1);
            } else if (!decoded.empty()) {
                generated_text = decoded;
            }

            const int eos = cfg_.eos_token_id;
            std::string finish_reason = output_tokens.empty() ? "length"
                : (output_tokens.back() == eos ? "stop" : "length");

            json choice = {
                {"index", 0},
                {"text", generated_text},
                {"logprobs", nullptr},
                {"finish_reason", finish_reason},
            };
            json usage = {
                {"prompt_tokens", (int)prompt_tokens.size()},
                {"completion_tokens", (int)output_tokens.size()},
                {"total_tokens", (int)prompt_tokens.size() + (int)output_tokens.size()},
            };
            cluster_send_json(res, json{
                {"id", id},
                {"object", "text_completion"},
                {"created", std::to_string(created)},
                {"model", model_name_},
                {"choices", {choice}},
                {"usage", usage},
            }.dump());
        });

        // ── POST /v1/chat/completions ────────────────────────────────────
        svr.Post("/v1/chat/completions", [this](const httplib::Request& req,
                                                httplib::Response& res) {
            cluster_cors(res);
            json body;
            try {
                body = json::parse(req.body);
            } catch (...) {
                return cluster_send_error(res, "Invalid JSON body", 400, "invalid_request");
            }
            bool stream = false;
            float temperature = 0.7f;
            int max_tokens = 256;
            std::vector<ChatMessage> messages;
            try {
                stream = body.value("stream", false);
                temperature = body.value("temperature", 0.7f);
                max_tokens = body.value("max_tokens", 256);
                if (body.contains("messages") && body["messages"].is_array()) {
                    for (const auto& m : body["messages"]) {
                        ChatMessage msg;
                        msg.role = m.value("role", std::string());
                        msg.content = m.value("content", std::string());
                        if (!msg.role.empty()) messages.push_back(std::move(msg));
                    }
                }
            } catch (const std::exception&) {
                return cluster_send_error(res, "Internal server error", 500, "internal_error");
            }
            if (messages.empty())
                return cluster_send_error(res, "No messages provided", 400, "invalid_request");
            if (max_tokens <= 0)
                return cluster_send_error(res, "max_tokens must be a positive integer", 400, "invalid_request");
            if (max_tokens > 4096) max_tokens = 4096;

            // v1 prompt build: system first, then user, joined by "\n\n".
            std::string system_part, user_part;
            for (const auto& m : messages) {
                if (m.role == "system")
                    system_part += (system_part.empty() ? "" : "\n\n") + m.content;
                else if (m.role == "user")
                    user_part += (user_part.empty() ? "" : "\n\n") + m.content;
            }
            std::string prompt;
            if (!system_part.empty()) prompt = system_part;
            if (!user_part.empty())
                prompt = prompt.empty() ? user_part : prompt + "\n\n" + user_part;

            std::vector<int> prompt_tokens;
            try {
                prompt_tokens = tokenize(prompt);
            } catch (const std::exception& e) {
                return cluster_send_error(res, e.what(), 500, "tokenization_error");
            }
            if (prompt_tokens.empty())
                return cluster_send_error(res, "Tokenization failed", 500, "tokenization_error");
            int context_budget = cfg_.max_position_embeddings - (int)prompt_tokens.size();
            if (context_budget <= 0)
                return cluster_send_error(res, "Prompt too long: exceeds model context", 400, "invalid_request");
            if (max_tokens > context_budget) max_tokens = std::max(1, context_budget);
            if (temperature < 0.01f) temperature = 0.0f;

            const std::string id = make_id("chatcmpl");
            const long created = now_ts();

            if (stream) {
                res.status = 200;
                res.set_header("Content-Type", "text/event-stream");
                res.set_header("Cache-Control", "no-cache");
                res.set_header("Connection", "keep-alive");
                res.set_header("Access-Control-Allow-Origin", "*");
                res.set_chunked_content_provider("text/event-stream",
                    [this, prompt_tokens, temperature, max_tokens, id, created](
                        size_t offset, httplib::DataSink& sink) -> bool {
                        if (offset > 0) return false;
                        std::vector<int> buf;
                        try {
                            generate(prompt_tokens, max_tokens, temperature, {},
                                [&](int tok) {
                                    buf.push_back(tok);
                                    bool is_eos = (tok == cfg_.eos_token_id);
                                    if (buf.size() >= 16 || is_eos) {
                                        std::string text = decode(buf);
                                        if (!text.empty()) {
                                            json delta = {
                                                {"role", "assistant"},
                                                {"content", text},
                                            };
                                            json chunk = {
                                                {"id", id},
                                                {"object", "chat.completion.chunk"},
                                                {"created", std::to_string(created)},
                                                {"model", model_name_},
                                                {"choices", {{{"index", 0}, {"delta", delta}}}},
                                            };
                                            std::string sse = "data: " + chunk.dump() + "\n\n";
                                            if (!sink.write(sse.data(), sse.size()))
                                                return false;
                                        }
                                        buf.clear();
                                    }
                                    return true;
                                });
                        } catch (const std::exception& e) {
                            Logger::error("Streaming chat failed: {}", e.what());
                        }
                        if (!buf.empty()) {
                            std::string text = decode(buf);
                            if (!text.empty()) {
                                json delta = {{"role", "assistant"}, {"content", text}};
                                json chunk = {
                                    {"id", id}, {"object", "chat.completion.chunk"},
                                    {"created", std::to_string(created)},
                                    {"model", model_name_},
                                    {"choices", {{{"index", 0}, {"delta", delta}}}},
                                };
                                std::string sse = "data: " + chunk.dump() + "\n\n";
                                sink.write(sse.data(), sse.size());
                            }
                            buf.clear();
                        }
                        sink.write("data: [DONE]\n\n", 16);
                        sink.done();
                        return true;
                    });
                return;
            }

            // Non-streaming.
            std::vector<int> output_tokens;
            try {
                output_tokens = generate(prompt_tokens, max_tokens, temperature, {}, nullptr);
            } catch (const std::exception& e) {
                Logger::error("Chat completion failed: {}", e.what());
                return cluster_send_error(res,
                    std::string("worker failed: ") + e.what(), 502, "cluster_error");
            }

            std::vector<int> all_tokens = prompt_tokens;
            all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
            std::string decoded = decode(all_tokens);
            std::string prompt_decoded = decode(prompt_tokens);
            std::string generated_text;
            if (decoded.size() > prompt_decoded.size()) {
                generated_text = decoded.substr(prompt_decoded.size());
                while (!generated_text.empty() && generated_text[0] == ' ')
                    generated_text.erase(0, 1);
            } else if (!decoded.empty()) {
                generated_text = decoded;
            }

            const int eos = cfg_.eos_token_id;
            std::string finish_reason = output_tokens.empty() ? "length"
                : (output_tokens.back() == eos ? "stop" : "length");

            json message = {
                {"role", "assistant"},
                {"content", generated_text},
            };
            json choice = {
                {"index", 0},
                {"message", message},
                {"finish_reason", finish_reason},
                {"logprobs", nullptr},
            };
            json usage = {
                {"prompt_tokens", (int)prompt_tokens.size()},
                {"completion_tokens", (int)output_tokens.size()},
                {"total_tokens", (int)prompt_tokens.size() + (int)output_tokens.size()},
            };
            cluster_send_json(res, json{
                {"id", id},
                {"object", "chat.completion"},
                {"created", std::to_string(created)},
                {"model", model_name_},
                {"choices", {choice}},
                {"usage", usage},
            }.dump());
        });

        svr.set_payload_max_length(10 * 1024 * 1024);  // 10 MB
        svr.set_read_timeout(300, 0);
        svr.set_write_timeout(300, 0);

        // SIGINT → stop the listen loop gracefully.
        std::signal(SIGINT, cluster_sigint_handler);
        g_server_ptr = &svr;
        std::thread watchdog([] {
            while (!g_stop_requested.load(std::memory_order_relaxed))
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (g_server_ptr) g_server_ptr->stop();
        });
        watchdog.detach();

        // Readiness banner.
        Logger::info("Terllama cluster ready: {} workers, model {}, layers split {}",
                     workers_.size(), model_name_, cfg_.num_hidden_layers);
        std::string split;
        for (size_t i = 0; i < workers_.size(); i++) {
            if (!split.empty()) split += ",";
            split += "[" + std::to_string(workers_[i].shard.start_layer) + "," +
                     std::to_string(workers_[i].shard.end_layer) + ")";
        }
        Logger::info("  Shards: {}", split);
        Logger::info("  API: http://0.0.0.0:{}/v1/models", api_port_);

        int port = 8375;
        try { port = std::stoi(api_port_); } catch (...) {}
        svr.listen("0.0.0.0", port);
        Logger::info("Cluster stopped.");
        return true;

    } catch (const std::exception& e) {
        Logger::error("Coordinator start failed: {}", e.what());
        return false;
    }
}

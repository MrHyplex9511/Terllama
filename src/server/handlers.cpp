/*
 * handlers.cpp — HTTP route handler implementations for Terllama server
 *
 * Uses nlohmann/json for JSON construction. Tokenization/decode use native
 * C++ + GigaToken; the engine never spawns a Python subprocess.
 * API response shapes are byte-identical to the original manual JSON output.
 */
#include "server/handlers.h"
#include "inference.h"
#include "core/tokenizer.h"

#include <json.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <atomic>
#include <random>
#include <chrono>
#include <thread>
#include <memory>
#include <deque>
#include <map>
#include <unistd.h>
#include <csignal>

// Thread-local batch decode buffer (16-token batches for streaming)
static thread_local std::vector<int> tls_decode_buffer;

// Signal flag (defined in commands.cpp)
extern std::atomic<bool> g_interrupted;

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════════════════
// RESPONSE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

void add_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.set_header("Access-Control-Max-Age",       "86400");
}

std::string error_body(const std::string& message,
                       const std::string& type)
{
    return json{
        {"error", json{
            {"message", message},
            {"type",    type}
        }}
    }.dump();
}

void send_json(httplib::Response& res, const std::string& body,
               int status)
{
    res.status = status;
    res.set_content(body, "application/json");
}

void send_error(httplib::Response& res, const std::string& message,
                int status, const std::string& type)
{
    send_json(res, error_body(message, type), status);
}

// ═══════════════════════════════════════════════════════════════════════════
// RATE LIMITING (sliding window per source IP)
// ═══════════════════════════════════════════════════════════════════════════

static constexpr size_t kRateLimitMax = 20;               // requests per window per IP
static constexpr std::chrono::seconds kRateLimitWindow{60};

static std::mutex g_rl_mutex;
static std::map<std::string,
                std::deque<std::chrono::steady_clock::time_point>> g_rl_history;

// Returns true if the request is within the per-IP sliding-window budget.
static bool rate_limit_allow(const httplib::Request& req) {
    const std::string ip = req.remote_addr;
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(g_rl_mutex);
    auto& deq = g_rl_history[ip];
    while (!deq.empty() && now - deq.front() > kRateLimitWindow)
        deq.pop_front();
    if (deq.size() >= kRateLimitMax) return false;
    deq.push_back(now);
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// max_tokens VALIDATION
// ═══════════════════════════════════════════════════════════════════════════

static constexpr int kMaxTokensCeiling = 4096;

// Clamps max_tokens to [1, kMaxTokensCeiling]. Sends 400 and returns false
// when the client asks for <= 0 tokens.
static bool sanitize_max_tokens(int& max_tokens, httplib::Response& res) {
    if (max_tokens <= 0) {
        send_error(res, "max_tokens must be a positive integer", 400, "invalid_request");
        return false;
    }
    if (max_tokens > kMaxTokensCeiling) max_tokens = kMaxTokensCeiling;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// API KEY AUTH CHECK
// ═══════════════════════════════════════════════════════════════════════════

// Returns true if request is authorized (or no key set).
// Sends 401 and returns false on mismatch.
bool check_api_key(const httplib::Request& req,
                   httplib::Response& res)
{
    if (g_api_key.empty()) return true;
    const auto& auth = req.get_header_value("Authorization");
    if (auth.size() > 7 && auth.substr(0, 7) == "Bearer " &&
        auth.substr(7) == g_api_key)
        return true;
    send_json(res, json{
        {"error", json{
            {"message", "Unauthorized"},
            {"type",    "auth_error"}
        }}
    }.dump(), 401);
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER (GigaToken encode; native C++ decode)
// ═══════════════════════════════════════════════════════════════════════════
// The encode path (tokenize) goes through the GigaToken wrapper (tokenizer.json
// loaded at model init). The decode path is called per-token in streaming and
// uses native C++ (Tokenizer from GGUF metadata) with a GigaToken fallback.
// There is NO Python subprocess fallback: if neither path is available the
// caller gets an empty result / error and fails cleanly.

std::vector<int> tokenize_with_helper(const std::string& prompt)
{
    // GigaToken encode (tokenizer.json loaded at model init)
    if (g_model.gigatoken && g_model.gigatoken->has_tokenizer()) {
        auto ids = g_model.gigatoken->encode(prompt);
        return std::vector<int>(ids.begin(), ids.end());
    }

    // No Python fallback: the engine must not require Python. The caller
    // (HTTP handler) turns this into a 500 error response.
    Logger::error("Tokenizer unavailable: no GigaToken .so / tokenizer.json and native encode is not supported. Install the bundled libgigatoken_rs.so or provide tokenizer.json.");
    return {};
}

// ── Decode fallback chain ─────────────────────────────────────────────────
// 1. Native Tokenizer (SentencePiece/"llama" vocab from GGUF or JSON) — fast
//    C++ path. Byte-level BPE ("gpt2") tokenizers can't be decoded natively;
//    native decode() returns "?" for those.
// 2. GigaToken wrapper (loaded from tokenizer.json in the model dir).
// No Python decode helper — if neither path works, return empty and log once.
std::string decode_with_fallback(const Tokenizer& tokenizer,
                                 const std::shared_ptr<GigaTokenWrapper>& gigatoken,
                                 const std::vector<int>& token_ids)
{
    // Fast path: native tokenizer (llama style)
    {
        std::string native = tokenizer.decode(token_ids);
        if (!native.empty() && native != "?") return native;
    }

    // GigaToken path
    if (gigatoken && gigatoken->has_tokenizer()) {
        std::vector<uint32_t> ids(token_ids.begin(), token_ids.end());
        std::string text = gigatoken->decode(ids);
        if (!text.empty()) return text;
    }

    // No Python fallback. Log once per process (this is called per token
    // batch in streaming) and return empty so the stream skips gracefully.
    static std::once_flag warned;
    std::call_once(warned, [] {
        Logger::error("Tokenizer unavailable: no native vocab and no GigaToken .so / tokenizer.json. Install the bundled libgigatoken_rs.so or provide tokenizer.json.");
    });
    return "";
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLETION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

std::string build_chat_prompt(const std::vector<Message>& messages) {
    std::string prompt;
    bool first_user = true;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            prompt = "[INST] " + msg.content + " [/INST]\n" + prompt;
        } else if (msg.role == "user") {
            if (!first_user) prompt += " ";
            prompt += "[INST] " + msg.content + " [/INST]";
            first_user = false;
        } else if (msg.role == "assistant") {
            prompt += " " + msg.content;
        }
    }
    return prompt;
}

std::string make_id(const char* prefix) {
    static std::atomic<uint64_t> counter{0};
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    uint64_t rand_part = rng();
    uint64_t seq       = counter.fetch_add(1, std::memory_order_relaxed);
    char buf[64];
    snprintf(buf, sizeof(buf), "%s-%llx%llx", prefix,
             static_cast<unsigned long long>(rand_part),
             static_cast<unsigned long long>(seq));
    return buf;
}

long now_ts() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ═══════════════════════════════════════════════════════════════════════════
// SNAPSHOT HELPER (thread-safe model state copy for concurrent requests)
// ═══════════════════════════════════════════════════════════════════════════

struct ModelSnap {
    ModelConfig cfg;
    std::vector<float> embedding;
    std::vector<LayerData> layers;
    std::vector<float> final_norm;
    std::vector<NormWeights> layer_norms;
    RoPECache rope;
    Tokenizer tokenizer;
    std::shared_ptr<GigaTokenWrapper> gigatoken;
};

#include <sys/stat.h>

struct RequestGuard {
    // The request slot is reserved atomically by try_reserve_request();
    // this guard only releases it on completion (destructor), including on
    // exception unwind.
    ~RequestGuard() {
        g_active_requests--;
        g_last_request_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

// Touch last request time — called at start of every public handler to
// prevent keep-alive watchdog from shutting down during active use.
static void touch_last_request() {
    g_last_request_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

size_t get_model_size_bytes() {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (!g_model.loaded) {
        size_t total = 0;
        struct stat st;
        if (g_model.model_dir.empty()) return 0;
        std::vector<std::string> files = {
            g_model.model_dir + "/model_extra.bin",
            g_model.model_dir + "/model_decomposed.bin",
            g_model.model_dir + "/model.gguf"
        };
        for (const auto& path : files) {
            if (stat(path.c_str(), &st) == 0) {
                total += st.st_size;
            }
        }
        return total;
    }
    size_t total = g_model.embedding.size() * sizeof(float);
    total += g_model.final_norm.size() * sizeof(float);
    for (const auto& w : g_model.layer_norms) {
        total += w.input_layernorm.size() * sizeof(float);
        total += w.post_attention_layernorm.size() * sizeof(float);
    }
    for (const auto& l : g_model.layers) {
        total += l.raw_weights.size() * sizeof(float);
        for (const auto& t : l.terms) {
            total += t.combined.size() * sizeof(uint32_t);
        }
        for (const auto& set : l.block_terms) {
            for (const auto& b : set) {
                total += b.packed.size() * sizeof(uint8_t);
            }
        }
    }
    total += g_model.rope.sin.size() * sizeof(float);
    total += g_model.rope.cos.size() * sizeof(float);
    return total;
}

size_t get_kv_cache_size_bytes() {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (!g_model.loaded) {
        return 128 * 1024 * 1024;
    }
    const auto& cfg = g_model.cfg;
    return (size_t)cfg.num_hidden_layers * cfg.max_position_embeddings * cfg.num_key_value_heads * cfg.head_dim * sizeof(float) * 2;
}

// ── Memory-limit admission (check-and-reserve atomically) ─────────────────
// g_active_requests is incremented here under g_admission_mutex so the
// projected-memory check and the reservation are atomic — no TOCTOU between
// the pre-check and RequestGuard's increment. RequestGuard only releases the
// slot on completion. Returns false (no reservation) when the request would
// exceed the memory limit.
static std::mutex g_admission_mutex;

static bool try_reserve_request() {
    std::lock_guard<std::mutex> lock(g_admission_mutex);
    if (g_memory_limit > 0) {
        size_t model_mem = get_model_size_bytes();
        size_t kv_mem    = get_kv_cache_size_bytes();
        int    active    = g_active_requests.load();
        size_t projected_mem = model_mem + (size_t)(active + 1) * kv_mem;
        if (projected_mem > g_memory_limit) {
            Logger::warn("Request rejected: projected memory ({} MB) exceeds limit ({} MB)",
                projected_mem / (1024 * 1024), g_memory_limit / (1024 * 1024));
            return false;
        }
    }
    g_active_requests++;
    return true;
}

// ── Model snapshot cache ─────────────────────────────────────────────────
// Snapshot copies are multi-GB; instead of deep-copying per request, cache a
// single snapshot and rebuild it only when the model is (re)loaded or
// unloaded (see invalidate_model_snapshot(), called from server.cpp).
// Concurrent requests share the cached snapshot via shared_ptr.
static std::mutex g_snapshot_mutex;
static std::shared_ptr<ModelSnap> g_snapshot_cache;

std::shared_ptr<ModelSnap> get_model_snapshot() {
    {
        std::lock_guard<std::mutex> lock(g_snapshot_mutex);
        if (g_snapshot_cache) return g_snapshot_cache;
    }
    // Build a fresh copy outside the cache lock to avoid lock-order inversion
    // with g_model_mutex (held by init_server / watchdog unload). A redundant
    // build can happen under a load race; the loser's copy is dropped below.
    auto fresh = std::make_shared<ModelSnap>();
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        *fresh = {
            g_model.cfg,
            g_model.embedding,
            g_model.layers,
            g_model.final_norm,
            g_model.layer_norms,
            g_model.rope,
            g_model.tokenizer,
            g_model.gigatoken
        };
    }
    std::lock_guard<std::mutex> lock(g_snapshot_mutex);
    if (!g_snapshot_cache) g_snapshot_cache = fresh;
    return g_snapshot_cache;
}

// Invalidates the cached snapshot so the next request rebuilds it. Called
// from server.cpp on model (re)load and on watchdog unload so the cached
// multi-GB copy is not retained after the model is freed.
void invalidate_model_snapshot() {
    std::lock_guard<std::mutex> lock(g_snapshot_mutex);
    g_snapshot_cache.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

// ── GET /v1/models ──────────────────────────────────────────────────────

void handle_models(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    add_cors_headers(res);
    touch_last_request();
    if (!check_api_key(req, res)) return;

    json entry = {
        {"id",       "default"},
        {"object",   "model"},
        {"created",  std::to_string(now_ts())},
        {"owned_by", "terllama"}
    };

    send_json(res, json{
        {"object", "list"},
        {"data",   {entry}}
    }.dump());
}

// ── POST /v1/chat/completions ───────────────────────────────────────────

void handle_chat_completions(const httplib::Request& req,
                             httplib::Response& res)
{
    add_cors_headers(res);
    touch_last_request();
    if (!check_api_key(req, res)) return;

    if (!rate_limit_allow(req)) {
        send_error(res, "Too many requests", 429, "rate_limited");
        return;
    }

    if (!g_model.loaded) {
        if (!g_model.model_dir.empty()) {
            Logger::info("Auto-reloading model from {}...", g_model.model_dir);
            if (!init_server(g_model.model_dir)) {
                send_error(res, "Failed to auto-reload model", 500, "model_load_failed");
                return;
            }
        } else {
            send_error(res, "Model not loaded and no directory configured", 503, "model_not_loaded");
            return;
        }
    }

    // Reserve the request slot atomically against the memory limit (see
    // try_reserve_request). RequestGuard releases it on completion.
    if (!try_reserve_request()) {
        send_error(res, "Service Unavailable: Request would exceed memory limit", 503, "service_unavailable");
        return;
    }
    auto req_guard = std::make_shared<RequestGuard>();

    // Parse request body via nlohmann/json
    json req_body;
    try {
        req_body = json::parse(req.body);
    } catch (...) {
        send_error(res, "Invalid JSON body", 400, "invalid_request");
        return;
    }

    std::string model;
    bool        stream     = false;
    float       temperature = 0.7f;
    int         max_tokens  = 256;
    std::vector<Message> messages;

    // Parameter extraction can throw (e.g. a string where a number is
    // expected). Wrap it so malformed fields yield the generic error rather
    // than leaking exception text to the client.
    try {
        model       = req_body.value("model", std::string("default"));
        stream      = req_body.value("stream", false);
        temperature = req_body.value("temperature", 0.7f);
        max_tokens  = req_body.value("max_tokens", 256);

        // Parse messages array
        if (req_body.contains("messages") && req_body["messages"].is_array()) {
            for (const auto& m : req_body["messages"]) {
                Message msg;
                msg.role    = m.value("role",    std::string());
                msg.content = m.value("content", std::string());
                if (!msg.role.empty()) messages.push_back(std::move(msg));
            }
        }
    } catch (const std::exception& e) {
        Logger::error("Bad request parameter: {}", e.what());
        send_error(res, "Internal server error", 500, "internal_error");
        return;
    }
    if (messages.empty()) {
        send_error(res, "No messages provided", 400, "invalid_request");
        return;
    }

    if (!sanitize_max_tokens(max_tokens, res)) return;

    std::string prompt = build_chat_prompt(messages);

    // Tokenize (also capture the model context size under the same lock)
    std::vector<int> prompt_tokens;
    int max_position_embeddings = 0;
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        prompt_tokens = tokenize_with_helper(prompt);
        max_position_embeddings = g_model.cfg.max_position_embeddings;
    }
    if (prompt_tokens.empty()) {
        send_error(res, "Tokenization failed", 500, "tokenization_error");
        return;
    }

    // Defense-in-depth for the KVCache overflow: reject prompts longer than
    // the model context, and never generate more tokens than fit in it.
    if ((size_t)max_position_embeddings < prompt_tokens.size()) {
        send_error(res, "Prompt too long: exceeds model context", 400, "invalid_request");
        return;
    }
    int context_budget = max_position_embeddings - (int)prompt_tokens.size();
    if (max_tokens > context_budget) max_tokens = std::max(1, context_budget);

    if (temperature < 0.01f) temperature = 0.0f;

    auto snap = get_model_snapshot();

    if (stream) {
        // ── Streaming (SSE via chunked transfer) ────────────────────────
        std::string id      = make_id("chatcmpl");
        long        created = now_ts();

        struct CbCtx {
            httplib::DataSink* sink{nullptr};
            std::string id;
            long created{0};
            ModelSnap* snap{nullptr};
        };
        auto ctx = std::make_shared<CbCtx>();
        ctx->id      = id;
        ctx->created = created;
        ctx->snap    = snap.get();

        res.status = 200;
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");

        res.set_chunked_content_provider("text/event-stream",
            [snap, prompt_tokens, temperature, max_tokens, ctx, req_guard](
                size_t offset, httplib::DataSink& sink) -> bool
            {
                if (offset > 0) return false;
                ctx->sink = &sink;

                tls_decode_buffer.clear();
                StreamCallback cb = [](int token, float*, void* userdata) -> bool {
                    auto* c = static_cast<CbCtx*>(userdata);
                    if (g_interrupted) return false;

                    tls_decode_buffer.push_back(token);
                    bool is_eos = (token == 0);
                    if (tls_decode_buffer.size() >= 16 || is_eos) {
                        std::string text = decode_with_fallback(
                            c->snap->tokenizer, c->snap->gigatoken,
                            tls_decode_buffer);
                        if (!text.empty()) {
                            json delta = {
                                {"role",    "assistant"},
                                {"content", text}
                            };
                            json choice = {
                                {"index", 0},
                                {"delta", delta}
                            };
                            json chunk = {
                                {"id",      c->id},
                                {"object",  "chat.completion.chunk"},
                                {"created", std::to_string(c->created)},
                                {"model",   "default"},
                                {"choices", {choice}}
                            };
                            std::string sse = "data: " + chunk.dump() + "\n\n";
                            if (!c->sink->write(sse.data(), sse.size()))
                                return false;
                        }
                        tls_decode_buffer.clear();
                    }
                    return true;
                };

                generate_stream(prompt_tokens, max_tokens, temperature,
                    snap->cfg, snap->embedding, snap->layers,
                    snap->final_norm, snap->layer_norms, snap->rope,
                    cb, ctx.get());

                // Flush remaining buffered tokens
                if (!tls_decode_buffer.empty()) {
                    std::string text = decode_with_fallback(
                        ctx->snap->tokenizer, ctx->snap->gigatoken,
                        tls_decode_buffer);
                    if (!text.empty()) {
                        json delta = {{"role","assistant"}, {"content",text}};
                        json chunk = {
                            {"id",ctx->id}, {"object","chat.completion.chunk"},
                            {"created",std::to_string(ctx->created)}, {"model","default"},
                            {"choices",{{{"index",0},{"delta",delta}}}}
                        };
                        std::string sse = "data: " + chunk.dump() + "\n\n";
                        sink.write(sse.data(), sse.size());
                    }
                    tls_decode_buffer.clear();
                }

                sink.write("data: [DONE]\n\n", 16);
                sink.done();
                return true;
            }
        );

    } else {
        // ── Non-streaming ───────────────────────────────────────────────
        std::vector<int> output_tokens;
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            auto result = generate(
                prompt_tokens, max_tokens, temperature,
                snap->cfg, snap->embedding, snap->layers,
                snap->final_norm, snap->layer_norms, snap->rope);
            output_tokens = result.first;
        }

        std::vector<int> all_tokens = prompt_tokens;
        all_tokens.insert(all_tokens.end(),
                          output_tokens.begin(), output_tokens.end());

        std::string decoded;
        std::string prompt_decoded;
        {
            decoded        = decode_with_fallback(snap->tokenizer, snap->gigatoken,
                                                  all_tokens);
            prompt_decoded = decode_with_fallback(snap->tokenizer, snap->gigatoken,
                                                  prompt_tokens);
        }

        std::string generated_text;
        if (decoded.size() > prompt_decoded.size()) {
            generated_text = decoded.substr(prompt_decoded.size());
            while (!generated_text.empty() && generated_text[0] == ' ')
                generated_text.erase(0, 1);
        } else if (!decoded.empty()) {
            generated_text = decoded;
        }

        int pt_count  = (int)prompt_tokens.size();
        int ct_count  = (int)output_tokens.size();
        std::string id = make_id("chatcmpl");
        long created  = now_ts();

        std::string finish_reason = output_tokens.empty() ? "length" :
            (output_tokens.back() == 0 ? "stop" : "length");

        json message = {
            {"role",    "assistant"},
            {"content", generated_text}
        };
        json choice = {
            {"index",         0},
            {"message",       message},
            {"finish_reason", finish_reason},
            {"logprobs",      nullptr}
        };
        json usage = {
            {"prompt_tokens",     pt_count},
            {"completion_tokens", ct_count},
            {"total_tokens",      pt_count + ct_count}
        };

        send_json(res, json{
            {"id",      id},
            {"object",  "chat.completion"},
            {"created", std::to_string(created)},
            {"model",   "default"},
            {"choices", {choice}},
            {"usage",   usage}
        }.dump());
    }
}

// ── POST /v1/completions ────────────────────────────────────────────────

void handle_completions(const httplib::Request& req,
                        httplib::Response& res)
{
    add_cors_headers(res);
    touch_last_request();
    if (!check_api_key(req, res)) return;

    if (!rate_limit_allow(req)) {
        send_error(res, "Too many requests", 429, "rate_limited");
        return;
    }

    if (!g_model.loaded) {
        if (!g_model.model_dir.empty()) {
            Logger::info("Auto-reloading model from {}...", g_model.model_dir);
            if (!init_server(g_model.model_dir)) {
                send_error(res, "Failed to auto-reload model", 500, "model_load_failed");
                return;
            }
        } else {
            send_error(res, "Model not loaded and no directory configured", 503, "model_not_loaded");
            return;
        }
    }

    // Reserve the request slot atomically against the memory limit (see
    // try_reserve_request). RequestGuard releases it on completion.
    if (!try_reserve_request()) {
        send_error(res, "Service Unavailable: Request would exceed memory limit", 503, "service_unavailable");
        return;
    }
    auto req_guard = std::make_shared<RequestGuard>();

    json req_body;
    try {
        req_body = json::parse(req.body);
    } catch (...) {
        send_error(res, "Invalid JSON body", 400, "invalid_request");
        return;
    }

    std::string prompt;
    bool        stream      = false;
    float       temperature = 0.7f;
    int         max_tokens  = 256;

    // Parameter extraction can throw (e.g. a string where a number is
    // expected). Wrap it so malformed fields yield the generic error rather
    // than leaking exception text to the client.
    try {
        prompt      = req_body.value("prompt", std::string());
        stream      = req_body.value("stream", false);
        temperature = req_body.value("temperature", 0.7f);
        max_tokens  = req_body.value("max_tokens", 256);
    } catch (const std::exception& e) {
        Logger::error("Bad request parameter: {}", e.what());
        send_error(res, "Internal server error", 500, "internal_error");
        return;
    }
    if (prompt.empty()) {
        send_error(res, "No prompt provided", 400, "invalid_request");
        return;
    }

    if (!sanitize_max_tokens(max_tokens, res)) return;

    std::vector<int> prompt_tokens;
    int max_position_embeddings = 0;
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        prompt_tokens = tokenize_with_helper(prompt);
        max_position_embeddings = g_model.cfg.max_position_embeddings;
    }
    if (prompt_tokens.empty()) {
        send_error(res, "Tokenization failed", 500, "tokenization_error");
        return;
    }

    // Defense-in-depth for the KVCache overflow: reject prompts longer than
    // the model context, and never generate more tokens than fit in it.
    if ((size_t)max_position_embeddings < prompt_tokens.size()) {
        send_error(res, "Prompt too long: exceeds model context", 400, "invalid_request");
        return;
    }
    int context_budget = max_position_embeddings - (int)prompt_tokens.size();
    if (max_tokens > context_budget) max_tokens = std::max(1, context_budget);

    if (temperature < 0.01f) temperature = 0.0f;

    auto snap = get_model_snapshot();

    if (stream) {
        // ── Streaming ───────────────────────────────────────────────────
        std::string id      = make_id("cmpl");
        long        created = now_ts();

        struct CbCtx {
            httplib::DataSink* sink{nullptr};
            std::string id;
            long created{0};
            ModelSnap* snap{nullptr};
        };
        auto ctx = std::make_shared<CbCtx>();
        ctx->id      = id;
        ctx->created = created;
        ctx->snap    = snap.get();

        res.status = 200;
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");

        res.set_chunked_content_provider("text/event-stream",
            [snap, prompt_tokens, temperature, max_tokens, ctx, req_guard](
                size_t offset, httplib::DataSink& sink) -> bool
            {
                if (offset > 0) return false;
                ctx->sink = &sink;

                tls_decode_buffer.clear();
                StreamCallback cb = [](int token, float*, void* userdata) -> bool {
                    auto* c = static_cast<CbCtx*>(userdata);
                    if (g_interrupted) return false;

                    tls_decode_buffer.push_back(token);
                    bool is_eos = (token == 0);
                    if (tls_decode_buffer.size() >= 16 || is_eos) {
                        std::string text = decode_with_fallback(
                            c->snap->tokenizer, c->snap->gigatoken,
                            tls_decode_buffer);
                        if (!text.empty()) {
                            json choice = {
                                {"index",         0},
                                {"text",          text},
                                {"logprobs",      nullptr},
                                {"finish_reason", nullptr}
                            };
                            json chunk = {
                                {"id",      c->id},
                                {"object",  "text_completion"},
                                {"created", std::to_string(c->created)},
                                {"model",   "default"},
                                {"choices", {choice}}
                            };
                            std::string sse = "data: " + chunk.dump() + "\n\n";
                            if (!c->sink->write(sse.data(), sse.size()))
                                return false;
                        }
                        tls_decode_buffer.clear();
                    }
                    return true;
                };

                generate_stream(prompt_tokens, max_tokens, temperature,
                    snap->cfg, snap->embedding, snap->layers,
                    snap->final_norm, snap->layer_norms, snap->rope,
                    cb, ctx.get());

                // Flush remaining buffered tokens
                if (!tls_decode_buffer.empty()) {
                    std::string text = decode_with_fallback(
                        ctx->snap->tokenizer, ctx->snap->gigatoken,
                        tls_decode_buffer);
                    if (!text.empty()) {
                        json choice = {
                            {"index",0}, {"text",text},
                            {"logprobs",nullptr}, {"finish_reason",nullptr}
                        };
                        json chunk = {
                            {"id",ctx->id}, {"object","text_completion"},
                            {"created",std::to_string(ctx->created)}, {"model","default"},
                            {"choices",{choice}}
                        };
                        std::string sse = "data: " + chunk.dump() + "\n\n";
                        sink.write(sse.data(), sse.size());
                    }
                    tls_decode_buffer.clear();
                }

                sink.write("data: [DONE]\n\n", 16);
                sink.done();
                return true;
            }
        );

    } else {
        // ── Non-streaming ───────────────────────────────────────────────
        std::vector<int> output_tokens;
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            auto result = generate(
                prompt_tokens, max_tokens, temperature,
                snap->cfg, snap->embedding, snap->layers,
                snap->final_norm, snap->layer_norms, snap->rope);
            output_tokens = result.first;
        }

        std::vector<int> all_tokens = prompt_tokens;
        all_tokens.insert(all_tokens.end(),
                          output_tokens.begin(), output_tokens.end());

        std::string decoded, prompt_decoded;
        {
            decoded        = decode_with_fallback(snap->tokenizer, snap->gigatoken,
                                                  all_tokens);
            prompt_decoded = decode_with_fallback(snap->tokenizer, snap->gigatoken,
                                                  prompt_tokens);
        }

        std::string generated_text;
        if (decoded.size() > prompt_decoded.size()) {
            generated_text = decoded.substr(prompt_decoded.size());
            while (!generated_text.empty() && generated_text[0] == ' ')
                generated_text.erase(0, 1);
        } else if (!decoded.empty()) {
            generated_text = decoded;
        }

        int pt_count = (int)prompt_tokens.size();
        int ct_count = (int)output_tokens.size();
        std::string id = make_id("cmpl");
        long created = now_ts();

        std::string finish_reason = output_tokens.empty() ? "length" :
            (output_tokens.back() == 0 ? "stop" : "length");

        json choice = {
            {"index",         0},
            {"text",          generated_text},
            {"logprobs",      nullptr},
            {"finish_reason", finish_reason}
        };
        json usage = {
            {"prompt_tokens",     pt_count},
            {"completion_tokens", ct_count},
            {"total_tokens",      pt_count + ct_count}
        };

        send_json(res, json{
            {"id",      id},
            {"object",  "text_completion"},
            {"created", std::to_string(created)},
            {"model",   "default"},
            {"choices", {choice}},
            {"usage",   usage}
        }.dump());
    }
}

// ── GET /health ─────────────────────────────────────────────────────────

void handle_health(const httplib::Request& req, httplib::Response& res) {
    add_cors_headers(res);
    touch_last_request();
    if (!check_api_key(req, res)) return;
    send_json(res, json{
        {"status", g_model.loaded ? "ok" : "not_loaded"},
        {"model",  "default"}
    }.dump());
}

// ── OPTIONS (CORS preflight) ────────────────────────────────────────────

void handle_options(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    add_cors_headers(res);
    res.status = 204;
}

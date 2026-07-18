/*
 * handlers.cpp — HTTP route handler implementations for Terllama server
 *
 * Uses nlohmann/json for JSON construction, posix_spawn for Python subprocess.
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
#include <chrono>
#include <thread>
#include <memory>
#include <spawn.h>
#include <sys/wait.h>
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
// API KEY AUTH CHECK
// ═══════════════════════════════════════════════════════════════════════════

// Returns true if request is authorized (or no key set).
// Sends 401 and returns false on mismatch.
static bool check_api_key(const httplib::Request& req,
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
// TOKENIZER (Python subprocess for encode only; native C++ decode)
// ═══════════════════════════════════════════════════════════════════════════
// The encode path (tokenize) is called once per request and uses Python
// via posix_spawn. The decode path is called per-token in streaming and
// uses native C++ (Tokenizer from GGUF metadata) for performance.

extern char **environ;

static int run_python_script(const std::string& script_path) {
    pid_t pid;
    std::string python = "python3";
    const char* argv[] = {"python3", script_path.c_str(), nullptr};

    int ret = posix_spawnp(&pid, python.c_str(), nullptr, nullptr,
                           const_cast<char* const*>(argv), environ);
    if (ret != 0) {
        Logger::error("posix_spawnp failed for {} (errno={})", script_path, ret);
        return -1;
    }
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

std::vector<int> tokenize_with_helper(const std::string& prompt,
                                      const std::string& helper_dir)
{
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file  = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        if (!pf) { Logger::error("Cannot write prompt file"); return {}; }
        pf << prompt;
    }
    std::string script = helper_dir + "/tokenize_helper.py";
    int ret = run_python_script(script);
    if (ret != 0) { Logger::error("Tokenization failed"); return {}; }

    std::vector<int> tokens;
    std::ifstream tf(token_file);
    int tid;
    while (tf >> tid) tokens.push_back(tid);
    return tokens;
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
    auto ts = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    uint64_t id = (static_cast<uint64_t>(ts) << 20) | (counter++ & 0xFFFFF);
    char buf[64];
    snprintf(buf, sizeof(buf), "%s-%llx", prefix,
             static_cast<unsigned long long>(id));
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
};

#include <sys/stat.h>

struct RequestGuard {
    RequestGuard() {
        g_active_requests++;
    }
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
            g_model.model_dir + "/model_decomposed_i2s.bin",
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
        for (const auto& b : l.i2s_blocks) {
            total += b.packed.size() * sizeof(uint8_t);
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

static ModelSnap snapshot_model() {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    return {
        g_model.cfg,
        g_model.embedding,
        g_model.layers,
        g_model.final_norm,
        g_model.layer_norms,
        g_model.rope
    };
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

    if (g_memory_limit > 0) {
        size_t model_mem = get_model_size_bytes();
        size_t kv_mem = get_kv_cache_size_bytes();
        int active = g_active_requests.load();
        size_t projected_mem = model_mem + (active + 1) * kv_mem;
        if (projected_mem > g_memory_limit) {
            Logger::warn("Request rejected: projected memory ({} MB) exceeds limit ({} MB)", 
                projected_mem / (1024 * 1024), g_memory_limit / (1024 * 1024));
            send_error(res, "Service Unavailable: Request would exceed memory limit", 503, "service_unavailable");
            return;
        }
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

    std::string model      = req_body.value("model", std::string("default"));
    bool        stream     = req_body.value("stream", false);
    float       temperature = req_body.value("temperature", 0.7f);
    int         max_tokens  = req_body.value("max_tokens", 256);

    // Parse messages array
    std::vector<Message> messages;
    if (req_body.contains("messages") && req_body["messages"].is_array()) {
        for (const auto& m : req_body["messages"]) {
            Message msg;
            msg.role    = m.value("role",    std::string());
            msg.content = m.value("content", std::string());
            if (!msg.role.empty()) messages.push_back(std::move(msg));
        }
    }
    if (messages.empty()) {
        send_error(res, "No messages provided", 400, "invalid_request");
        return;
    }

    std::string prompt = build_chat_prompt(messages);

    // Tokenize
    std::vector<int> prompt_tokens;
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        prompt_tokens = tokenize_with_helper(prompt, g_model.helper_dir);
    }
    if (prompt_tokens.empty()) {
        send_error(res, "Tokenization failed", 500, "tokenization_error");
        return;
    }

    if (temperature < 0.01f) temperature = 0.0f;

    auto snap = std::make_shared<ModelSnap>(snapshot_model());

    if (stream) {
        // ── Streaming (SSE via chunked transfer) ────────────────────────
        std::string id      = make_id("chatcmpl");
        long        created = now_ts();

        struct CbCtx {
            httplib::DataSink* sink{nullptr};
            std::string id;
            long created{0};
        };
        auto ctx = std::make_shared<CbCtx>();
        ctx->id      = id;
        ctx->created = created;

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
                        std::string text = g_model.tokenizer.decode(tls_decode_buffer);
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
                    std::string text = g_model.tokenizer.decode(tls_decode_buffer);
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
                g_model.cfg, g_model.embedding, g_model.layers,
                g_model.final_norm, g_model.layer_norms, g_model.rope);
            output_tokens = result.first;
        }

        std::vector<int> all_tokens = prompt_tokens;
        all_tokens.insert(all_tokens.end(),
                          output_tokens.begin(), output_tokens.end());

        std::string decoded;
        std::string prompt_decoded;
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            decoded        = g_model.tokenizer.decode(all_tokens);
            prompt_decoded = g_model.tokenizer.decode(prompt_tokens);
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

    if (g_memory_limit > 0) {
        size_t model_mem = get_model_size_bytes();
        size_t kv_mem = get_kv_cache_size_bytes();
        int active = g_active_requests.load();
        size_t projected_mem = model_mem + (active + 1) * kv_mem;
        if (projected_mem > g_memory_limit) {
            Logger::warn("Request rejected: projected memory ({} MB) exceeds limit ({} MB)", 
                projected_mem / (1024 * 1024), g_memory_limit / (1024 * 1024));
            send_error(res, "Service Unavailable: Request would exceed memory limit", 503, "service_unavailable");
            return;
        }
    }

    auto req_guard = std::make_shared<RequestGuard>();

    json req_body;
    try {
        req_body = json::parse(req.body);
    } catch (...) {
        send_error(res, "Invalid JSON body", 400, "invalid_request");
        return;
    }

    std::string prompt = req_body.value("prompt", std::string());
    if (prompt.empty()) {
        send_error(res, "No prompt provided", 400, "invalid_request");
        return;
    }

    bool   stream      = req_body.value("stream", false);
    float  temperature = req_body.value("temperature", 0.7f);
    int    max_tokens  = req_body.value("max_tokens", 256);

    std::vector<int> prompt_tokens;
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        prompt_tokens = tokenize_with_helper(prompt, g_model.helper_dir);
    }
    if (prompt_tokens.empty()) {
        send_error(res, "Tokenization failed", 500, "tokenization_error");
        return;
    }

    if (temperature < 0.01f) temperature = 0.0f;

    auto snap = std::make_shared<ModelSnap>(snapshot_model());

    if (stream) {
        // ── Streaming ───────────────────────────────────────────────────
        std::string id      = make_id("cmpl");
        long        created = now_ts();

        struct CbCtx {
            httplib::DataSink* sink{nullptr};
            std::string id;
            long created{0};
        };
        auto ctx = std::make_shared<CbCtx>();
        ctx->id      = id;
        ctx->created = created;

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
                        std::string text = g_model.tokenizer.decode(tls_decode_buffer);
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
                    std::string text = g_model.tokenizer.decode(tls_decode_buffer);
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
                g_model.cfg, g_model.embedding, g_model.layers,
                g_model.final_norm, g_model.layer_norms, g_model.rope);
            output_tokens = result.first;
        }

        std::vector<int> all_tokens = prompt_tokens;
        all_tokens.insert(all_tokens.end(),
                          output_tokens.begin(), output_tokens.end());

        std::string decoded, prompt_decoded;
        {
            std::lock_guard<std::mutex> lock(g_model_mutex);
            decoded        = g_model.tokenizer.decode(all_tokens);
            prompt_decoded = g_model.tokenizer.decode(prompt_tokens);
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
    (void)req;
    add_cors_headers(res);
    touch_last_request();
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

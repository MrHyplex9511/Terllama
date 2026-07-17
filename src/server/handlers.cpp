/*
 * handlers.cpp — HTTP route handler implementations for Terllama server
 *
 * Uses nlohmann/json for JSON construction, posix_spawn for Python subprocess.
 * API response shapes are byte-identical to the original manual JSON output.
 */
#include "server/handlers.h"
#include "inference.h"

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
// TOKENIZER (Python subprocess via posix_spawn)
// ═══════════════════════════════════════════════════════════════════════════

extern char **environ;

static int run_python_script(const std::string& script_path) {
    pid_t pid;
    const char* python = "python3";
    char* const argv[] = {
        const_cast<char*>(python),
        const_cast<char*>(script_path.c_str()),
        nullptr
    };
    int ret = posix_spawnp(&pid, python, nullptr, nullptr, argv, environ);
    if (ret != 0) {
        std::cerr << "posix_spawnp failed for " << script_path
                  << " (errno=" << ret << ")" << std::endl;
        return -1;
    }
    int status;
    waitpid(pid, &status, 0);
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

std::vector<int> tokenize_with_python(const std::string& prompt,
                                      const std::string& helper_dir)
{
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file  = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        if (!pf) { std::cerr << "Cannot write prompt file\n"; return {}; }
        pf << prompt;
    }
    std::string script = helper_dir + "/tokenize_helper.py";
    int ret = run_python_script(script);
    if (ret != 0) { std::cerr << "Tokenization failed\n"; return {}; }

    std::vector<int> tokens;
    std::ifstream tf(token_file);
    int tid;
    while (tf >> tid) tokens.push_back(tid);
    return tokens;
}

std::string decode_with_python(const std::vector<int>& tokens,
                               const std::string& helper_dir)
{
    std::string token_file = "/tmp/ternary_decode_in.txt";
    std::string out_file   = "/tmp/ternary_decode_out.txt";
    {
        std::ofstream tf(token_file);
        if (!tf) return "?";
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) tf << " ";
            tf << tokens[i];
        }
    }
    std::string script = helper_dir + "/decode_helper.py";
    int ret = run_python_script(script);
    if (ret != 0) { std::cerr << "Decoding failed\n"; return "?"; }

    std::ifstream of(out_file);
    std::stringstream ss;
    ss << of.rdbuf();
    return ss.str();
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
    std::string helper_dir;
};

static ModelSnap snapshot_model() {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    return {
        g_model.cfg,
        g_model.embedding,
        g_model.layers,
        g_model.final_norm,
        g_model.layer_norms,
        g_model.rope,
        g_model.helper_dir
    };
}

// ═══════════════════════════════════════════════════════════════════════════
// HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

// ── GET /v1/models ──────────────────────────────────────────────────────

void handle_models(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    add_cors_headers(res);
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
    if (!check_api_key(req, res)) return;

    if (!g_model.loaded) {
        send_error(res, "Model not loaded", 503, "model_not_loaded");
        return;
    }

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
        prompt_tokens = tokenize_with_python(prompt, g_model.helper_dir);
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
            std::string helper_dir;
            std::string id;
            long created{0};
        };
        auto ctx = std::make_shared<CbCtx>();
        ctx->helper_dir = snap->helper_dir;
        ctx->id         = id;
        ctx->created    = created;

        res.status = 200;
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");

        res.set_chunked_content_provider("text/event-stream",
            [snap, prompt_tokens, temperature, max_tokens, ctx](
                size_t offset, httplib::DataSink& sink) -> bool
            {
                if (offset > 0) return false;
                ctx->sink = &sink;

                StreamCallback cb = [](int token, float*, void* userdata) -> bool {
                    auto* c = static_cast<CbCtx*>(userdata);
                    std::vector<int> single = {token};
                    std::string text = decode_with_python(single, c->helper_dir);

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
                    return true;
                };

                generate_stream(prompt_tokens, max_tokens, temperature,
                    snap->cfg, snap->embedding, snap->layers,
                    snap->final_norm, snap->layer_norms, snap->rope,
                    cb, ctx.get());

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
            decoded        = decode_with_python(all_tokens, g_model.helper_dir);
            prompt_decoded = decode_with_python(prompt_tokens, g_model.helper_dir);
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
    if (!check_api_key(req, res)) return;

    if (!g_model.loaded) {
        send_error(res, "Model not loaded", 503, "model_not_loaded");
        return;
    }

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
        prompt_tokens = tokenize_with_python(prompt, g_model.helper_dir);
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
            std::string helper_dir;
            std::string id;
            long created{0};
        };
        auto ctx = std::make_shared<CbCtx>();
        ctx->helper_dir = snap->helper_dir;
        ctx->id         = id;
        ctx->created    = created;

        res.status = 200;
        res.set_header("Content-Type", "text/event-stream");
        res.set_header("Cache-Control", "no-cache");
        res.set_header("Connection", "keep-alive");
        res.set_header("Access-Control-Allow-Origin", "*");

        res.set_chunked_content_provider("text/event-stream",
            [snap, prompt_tokens, temperature, max_tokens, ctx](
                size_t offset, httplib::DataSink& sink) -> bool
            {
                if (offset > 0) return false;
                ctx->sink = &sink;

                StreamCallback cb = [](int token, float*, void* userdata) -> bool {
                    auto* c = static_cast<CbCtx*>(userdata);
                    std::vector<int> single = {token};
                    std::string text = decode_with_python(single, c->helper_dir);

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
                    return true;
                };

                generate_stream(prompt_tokens, max_tokens, temperature,
                    snap->cfg, snap->embedding, snap->layers,
                    snap->final_norm, snap->layer_norms, snap->rope,
                    cb, ctx.get());

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
            decoded        = decode_with_python(all_tokens, g_model.helper_dir);
            prompt_decoded = decode_with_python(prompt_tokens, g_model.helper_dir);
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

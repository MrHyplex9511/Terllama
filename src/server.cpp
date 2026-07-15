/*
 * server.cpp — HTTP server for Terllama (OpenAI-compatible API)
 *
 * Endpoints:
 *   GET  /v1/models              List available models
 *   POST /v1/chat/completions    Chat completions (streaming + non-streaming)
 *   POST /v1/completions         Text completions (streaming + non-streaming)
 *   GET  /health                 Health check
 *   GET  /                       Serve web UI or API listing
 *
 * Build:
 *   g++ -std=c++17 -O3 -fopenmp -I. -Ithird_party \
 *       src/server.cpp src/dispatcher.cpp src/kernel_scalar.cpp src/kernel_avx2.cpp \
 *       -o terllama-server -lm -fopenmp -lpthread
 *
 * Usage:
 *   ./terllama-server [port]
 *   TERLLAMA_MODEL_DIR=/path/to/model ./terllama-server
 *   TERLLAMA_PORT=8080 ./terllama-server
 */

#include "model.h"
#include "loader.h"
#include "kernel_decl.h"
#include "inference.h"
#include <httplib.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <limits.h>
#include <unistd.h>
#include <libgen.h>
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>

// ═══════════════════════════════════════════════════════════════════════════
// GLOBAL MODEL STATE (loaded once at startup, read-only after that)
// ═══════════════════════════════════════════════════════════════════════════

static struct {
    ModelConfig cfg;
    std::vector<float> embedding;
    std::vector<LayerData> layers;
    std::vector<float> final_norm;
    std::vector<NormWeights> layer_norms;
    RoPECache rope;
    std::string model_dir;
    std::string helper_dir;
    CPUArch arch{CPUArch::UNKNOWN};
    bool loaded{false};
} g_model;

static std::mutex g_model_mutex;

// ═══════════════════════════════════════════════════════════════════════════
// JSON HELPERS (no nlohmann/json dependency)
// ═══════════════════════════════════════════════════════════════════════════

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x",
                             static_cast<unsigned char>(c));
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static std::string json_encode_string(const std::string& s) {
    return "\"" + json_escape(s) + "\"";
}

static std::string json_obj(
    std::initializer_list<std::pair<const char*, std::string>> fields)
{
    std::string out = "{";
    bool first = true;
    for (auto& [k, v] : fields) {
        if (!first) out += ",";
        first = false;
        out += json_encode_string(k) + ":" + v;
    }
    out += "}";
    return out;
}

static std::string json_arr(
    std::initializer_list<std::string> elems)
{
    std::string out = "[";
    bool first = true;
    for (auto& e : elems) {
        if (!first) out += ",";
        first = false;
        out += e;
    }
    out += "]";
    return out;
}

// Extract a JSON string value by key: "key":"<value>"
// Handles basic escape sequences. Returns empty on failure.
static std::string json_extract_string(const std::string& json,
                                       const std::string& key)
{
    std::string search = "\"" + key + "\":\"";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    std::string val;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            switch (json[pos + 1]) {
                case '"':  val += '"';  break;
                case '\\': val += '\\'; break;
                case 'n':  val += '\n'; break;
                case 'r':  val += '\r'; break;
                case 't':  val += '\t'; break;
                default:   val += json[pos + 1];
            }
            pos += 2;
        } else {
            val += json[pos++];
        }
    }
    return val;
}

// Extract a numeric value (int or float) as a string by key
static std::string json_extract_number_str(const std::string& json,
                                           const std::string& key)
{
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
        pos++;
    size_t end = pos;
    if (end < json.size() && json[end] == '-') end++;
    while (end < json.size() && (isdigit(json[end]) || json[end] == '.' ||
           json[end] == 'e' || json[end] == 'E' || json[end] == '+' ||
           json[end] == '-'))
        end++;
    return json.substr(pos, end - pos);
}

static int json_extract_int(const std::string& json, const std::string& key,
                            int def = 0)
{
    auto s = json_extract_number_str(json, key);
    return s.empty() ? def : std::stoi(s);
}

static float json_extract_float(const std::string& json, const std::string& key,
                                float def = 0.0f)
{
    auto s = json_extract_number_str(json, key);
    return s.empty() ? def : std::stof(s);
}

static bool json_extract_bool(const std::string& json, const std::string& key,
                              bool def = false)
{
    std::string search = "\"" + key + "\":";
    auto pos = json.find(search);
    if (pos == std::string::npos) return def;
    pos += search.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
        pos++;
    if (json.substr(pos, 4) == "true")  return true;
    if (json.substr(pos, 5) == "false") return false;
    return def;
}

// Extract the inner content of a JSON array by key: "key":[<content>]
// Handles nested brackets to find the matching closing bracket.
static std::string json_extract_array_raw(const std::string& json,
                                          const std::string& key)
{
    std::string search = "\"" + key + "\":[";
    auto pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    int depth = 0;
    size_t start = pos;
    while (pos < json.size()) {
        if (json[pos] == '[') depth++;
        else if (json[pos] == ']') {
            if (depth == 0) break;
            depth--;
        }
        pos++;
    }
    if (pos >= json.size()) return "";
    return json.substr(start, pos - start);
}

struct Message {
    std::string role;
    std::string content;
};

// Parse the "messages" array from a chat completion request body.
// Walks through [{"role":"...","content":"..."}, ...] finding each object.
static std::vector<Message> json_extract_messages(const std::string& json) {
    std::vector<Message> messages;
    auto arr = json_extract_array_raw(json, "messages");
    if (arr.empty()) return messages;

    size_t i = 0;
    while (i < arr.size()) {
        // Advance to next '{'
        while (i < arr.size() && arr[i] != '{') i++;
        if (i >= arr.size()) break;

        // Find matching '}'
        size_t obj_end = i + 1;
        int depth = 1;
        while (obj_end < arr.size() && depth > 0) {
            if (arr[obj_end] == '{') depth++;
            else if (arr[obj_end] == '}') depth--;
            obj_end++;
        }

        std::string obj_str = arr.substr(i, obj_end - i);
        std::string role    = json_extract_string(obj_str, "role");
        std::string content = json_extract_string(obj_str, "content");
        if (!role.empty()) {
            messages.push_back({role, content});
        }
        i = obj_end;
    }
    return messages;
}

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER (Python helper bridge — same pattern as main.cpp)
// ═══════════════════════════════════════════════════════════════════════════

static std::vector<int> tokenize_with_python(const std::string& prompt,
                                             const std::string& helper_dir)
{
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file  = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        if (!pf) { std::cerr << "Cannot write prompt file\n"; return {}; }
        pf << prompt;
    }
    std::string cmd = "python3 " + helper_dir + "/tokenize_helper.py";
    int ret = system(cmd.c_str());
    if (ret != 0) { std::cerr << "Tokenization failed\n"; return {}; }

    std::vector<int> tokens;
    std::ifstream tf(token_file);
    int tid;
    while (tf >> tid) tokens.push_back(tid);
    return tokens;
}

static std::string decode_with_python(const std::vector<int>& tokens,
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
    std::string cmd = "python3 " + helper_dir + "/decode_helper.py";
    int ret = system(cmd.c_str());
    if (ret != 0) { std::cerr << "Decoding failed\n"; return "?"; }

    std::ifstream of(out_file);
    std::stringstream ss;
    ss << of.rdbuf();
    return ss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// MODEL LOADING
// ═══════════════════════════════════════════════════════════════════════════

static bool init_server(const std::string& model_dir) {
    std::lock_guard<std::mutex> lock(g_model_mutex);
    if (g_model.loaded) return true;

    g_model.model_dir  = model_dir;
    // Resolve scripts/ relative to this binary (co-located in repo)
    g_model.helper_dir = []{
        char buf[PATH_MAX];
        ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf)-1);
        if (len != -1) { buf[len]='\0'; return std::string(dirname(buf))+"/scripts"; }
        return std::string("scripts");
    }();
    g_model.arch       = detect_cpu_arch();

    // Allow environment override for testing
    const char* arch_override = std::getenv("TERLLAMA_ARCH");
    if (arch_override) {
        std::string ao(arch_override);
        if (ao == "scalar")       g_model.arch = CPUArch::X86_64_SCALAR;
        else if (ao == "sse42")   g_model.arch = CPUArch::X86_64_SSE42;
        else if (ao == "avx")     g_model.arch = CPUArch::X86_64_AVX;
        else if (ao == "avx2")    g_model.arch = CPUArch::X86_64_AVX2;
        else if (ao == "avx512")  g_model.arch = CPUArch::X86_64_AVX512;
        else if (ao == "neon")    g_model.arch = CPUArch::ARM64_NEON;
    }

    std::cout << "Terllama — CPU: " << cpu_arch_name(g_model.arch)
              << "  |  Model: " << model_dir << std::endl;

    try {
        std::cout << "Loading config..." << std::endl;
        g_model.cfg = load_config(model_dir + "/model_extra.bin");

        std::cout << "Loading embedding ("
                  << g_model.cfg.vocab_size << "x"
                  << g_model.cfg.hidden_size << ")..." << std::endl;
        g_model.embedding = load_embedding(model_dir + "/model_extra.bin",
                                           g_model.cfg);

        std::cout << "Loading " << g_model.cfg.num_hidden_layers
                  << " layer norms..." << std::endl;
        g_model.layer_norms = load_layer_norms(model_dir + "/model_extra.bin",
                                                g_model.cfg);

        std::cout << "Loading final norm..." << std::endl;
        g_model.final_norm = load_final_norm(model_dir + "/model_extra.bin",
                                              g_model.cfg);

        // Load decomposed linear layers. Prefer I2_S format.
        std::string i2s_path = model_dir + "/model_decomposed_i2s.bin";
        std::string old_path = model_dir + "/model_decomposed.bin";
        {
            std::ifstream test_i2s(i2s_path);
            if (test_i2s.good()) {
                std::cout << "Using I2_S format model..." << std::endl;
                g_model.layers = load_decomposed_layers_i2s(i2s_path);
            } else {
                std::cout << "Loading decomposed linear layers (ALS format)..."
                          << std::endl;
                g_model.layers = load_decomposed_layers(old_path);
            }
        }
        std::cout << "  Loaded " << g_model.layers.size() << " layers."
                  << std::endl;

        std::cout << "Building RoPE cache..." << std::endl;
        g_model.rope = build_rope_cache(
            g_model.cfg.max_position_embeddings,
            g_model.cfg.head_dim,
            g_model.cfg.rope_theta);

        g_model.loaded = true;
        std::cout << "Server initialized." << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPLETION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

static std::string build_chat_prompt(const std::vector<Message>& messages) {
    std::string prompt;
    bool first_user = true;
    for (const auto& msg : messages) {
        if (msg.role == "system") {
            // System message: prepend with [INST] markers
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

// OpenAI-style completion ID (hex-timestamp-based)
static std::string make_id(const char* prefix = "chatcmpl") {
    static std::atomic<uint64_t> counter{0};
    auto ts = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    uint64_t id = (static_cast<uint64_t>(ts) << 20) | (counter++ & 0xFFFFF);
    char buf[64];
    snprintf(buf, sizeof(buf), "%s-%llx", prefix,
             static_cast<unsigned long long>(id));
    return buf;
}

static long now_ts() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

// ═══════════════════════════════════════════════════════════════════════════
// CORS
// ═══════════════════════════════════════════════════════════════════════════

static void add_cors_headers(httplib::Response& res) {
    res.set_header("Access-Control-Allow-Origin",  "*");
    res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
    res.set_header("Access-Control-Max-Age",       "86400");
}

// ═══════════════════════════════════════════════════════════════════════════
// ERROR / JSON RESPONSE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

static std::string error_body(const std::string& message,
                              const std::string& type = "server_error")
{
    return json_obj({
        {"error", json_obj({
            {"message", json_encode_string(message)},
            {"type",    json_encode_string(type)}
        })}
    });
}

static void send_json(httplib::Response& res, const std::string& body,
                      int status = 200)
{
    res.status = status;
    res.set_content(body, "application/json");
}

static void send_error(httplib::Response& res, const std::string& message,
                       int status = 500,
                       const std::string& type = "server_error")
{
    send_json(res, error_body(message, type), status);
}

// ═══════════════════════════════════════════════════════════════════════════
// HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

// ── GET /v1/models ──────────────────────────────────────────────────────

static void handle_models(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    add_cors_headers(res);

    std::string entry = json_obj({
        {"id",         json_encode_string("default")},
        {"object",     json_encode_string("model")},
        {"created",    std::to_string(now_ts())},
        {"owned_by",   json_encode_string("terllama")}
    });

    send_json(res, json_obj({
        {"object", json_encode_string("list")},
        {"data",   json_arr({entry})}
    }));
}

// ── POST /v1/chat/completions ───────────────────────────────────────────

static void handle_chat_completions(const httplib::Request& req,
                                    httplib::Response& res)
{
    add_cors_headers(res);

    if (!g_model.loaded) {
        send_error(res, "Model not loaded", 503, "model_not_loaded");
        return;
    }

    // Parse request body
    std::string model = json_extract_string(req.body, "model");
    if (model.empty()) model = "default";

    bool   stream      = json_extract_bool(req.body, "stream", false);
    float  temperature = json_extract_float(req.body, "temperature", 0.7f);
    int    max_tokens  = json_extract_int(req.body, "max_tokens", 256);

    auto messages = json_extract_messages(req.body);
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

    // Snapshot model state for this request
    struct Snap {
        ModelConfig cfg;
        std::vector<float> embedding;
        std::vector<LayerData> layers;
        std::vector<float> final_norm;
        std::vector<NormWeights> layer_norms;
        RoPECache rope;
        std::string helper_dir;
    };
    auto snap = std::make_shared<Snap>();
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        snap->cfg         = g_model.cfg;
        snap->embedding   = g_model.embedding;
        snap->layers      = g_model.layers;
        snap->final_norm  = g_model.final_norm;
        snap->layer_norms = g_model.layer_norms;
        snap->rope        = g_model.rope;
        snap->helper_dir  = g_model.helper_dir;
    }

    if (stream) {
        // ── Streaming (SSE via chunked transfer) ────────────────────────
        std::string id      = make_id("chatcmpl");
        long        created = now_ts();

        // Shared context for the streaming callback
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
                // Only generate on the first invocation
                if (offset > 0) return false;

                ctx->sink = &sink;

                // StreamCallback: decode one token and send SSE
                StreamCallback cb = [](int token, float*, void* userdata) -> bool {
                    auto* c = static_cast<CbCtx*>(userdata);
                    std::vector<int> single = {token};
                    std::string text = decode_with_python(single, c->helper_dir);

                    std::string delta = json_obj({
                        {"role",    json_encode_string("assistant")},
                        {"content", json_encode_string(text)}
                    });
                    std::string choice = json_obj({
                        {"index", "0"},
                        {"delta", delta}
                    });
                    std::string chunk = json_obj({
                        {"id",      json_encode_string(c->id)},
                        {"object",  json_encode_string("chat.completion.chunk")},
                        {"created", std::to_string(c->created)},
                        {"model",   json_encode_string("default")},
                        {"choices", json_arr({choice})}
                    });

                    std::string sse = "data: " + chunk + "\n\n";
                    if (!c->sink->write(sse.data(), sse.size())) {
                        return false; // client disconnected
                    }
                    return true;
                };

                auto ret = generate_stream(
                    prompt_tokens, max_tokens, temperature,
                    snap->cfg, snap->embedding, snap->layers,
                    snap->final_norm, snap->layer_norms, snap->rope,
                    cb, ctx.get());

                (void)ret;
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

        // Decode full output
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

        // Extract generated portion
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

        std::string message = json_obj({
            {"role",    json_encode_string("assistant")},
            {"content", json_encode_string(generated_text)}
        });
        std::string choice = json_obj({
            {"index",         "0"},
            {"message",       message},
            {"finish_reason", json_encode_string(finish_reason)},
            {"logprobs",      "null"}
        });
        std::string usage = json_obj({
            {"prompt_tokens",     std::to_string(pt_count)},
            {"completion_tokens", std::to_string(ct_count)},
            {"total_tokens",      std::to_string(pt_count + ct_count)}
        });

        send_json(res, json_obj({
            {"id",      json_encode_string(id)},
            {"object",  json_encode_string("chat.completion")},
            {"created", std::to_string(created)},
            {"model",   json_encode_string("default")},
            {"choices", json_arr({choice})},
            {"usage",   usage}
        }));
    }
}

// ── POST /v1/completions ────────────────────────────────────────────────

static void handle_completions(const httplib::Request& req,
                               httplib::Response& res)
{
    add_cors_headers(res);

    if (!g_model.loaded) {
        send_error(res, "Model not loaded", 503, "model_not_loaded");
        return;
    }

    std::string prompt = json_extract_string(req.body, "prompt");
    if (prompt.empty()) {
        send_error(res, "No prompt provided", 400, "invalid_request");
        return;
    }

    bool   stream      = json_extract_bool(req.body, "stream", false);
    float  temperature = json_extract_float(req.body, "temperature", 0.7f);
    int    max_tokens  = json_extract_int(req.body, "max_tokens", 256);

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

    struct Snap {
        ModelConfig cfg;
        std::vector<float> embedding;
        std::vector<LayerData> layers;
        std::vector<float> final_norm;
        std::vector<NormWeights> layer_norms;
        RoPECache rope;
        std::string helper_dir;
    };
    auto snap = std::make_shared<Snap>();
    {
        std::lock_guard<std::mutex> lock(g_model_mutex);
        snap->cfg         = g_model.cfg;
        snap->embedding   = g_model.embedding;
        snap->layers      = g_model.layers;
        snap->final_norm  = g_model.final_norm;
        snap->layer_norms = g_model.layer_norms;
        snap->rope        = g_model.rope;
        snap->helper_dir  = g_model.helper_dir;
    }

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

                    std::string choice = json_obj({
                        {"index",         "0"},
                        {"text",          json_encode_string(text)},
                        {"logprobs",      "null"},
                        {"finish_reason", "null"}
                    });
                    std::string chunk = json_obj({
                        {"id",      json_encode_string(c->id)},
                        {"object",  json_encode_string("text_completion")},
                        {"created", std::to_string(c->created)},
                        {"model",   json_encode_string("default")},
                        {"choices", json_arr({choice})}
                    });

                    std::string sse = "data: " + chunk + "\n\n";
                    if (!c->sink->write(sse.data(), sse.size())) {
                        return false;
                    }
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

        std::string choice = json_obj({
            {"index",         "0"},
            {"text",          json_encode_string(generated_text)},
            {"logprobs",      "null"},
            {"finish_reason", json_encode_string(finish_reason)}
        });
        std::string usage = json_obj({
            {"prompt_tokens",     std::to_string(pt_count)},
            {"completion_tokens", std::to_string(ct_count)},
            {"total_tokens",      std::to_string(pt_count + ct_count)}
        });

        send_json(res, json_obj({
            {"id",      json_encode_string(id)},
            {"object",  json_encode_string("text_completion")},
            {"created", std::to_string(created)},
            {"model",   json_encode_string("default")},
            {"choices", json_arr({choice})},
            {"usage",   usage}
        }));
    }
}

// ── GET /health ─────────────────────────────────────────────────────────

static void handle_health(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    add_cors_headers(res);
    send_json(res, json_obj({
        {"status", json_encode_string(g_model.loaded ? "ok" : "not_loaded")},
        {"model",  json_encode_string("default")}
    }));
}

// ── OPTIONS (CORS preflight) ────────────────────────────────────────────

static void handle_options(const httplib::Request& req, httplib::Response& res) {
    (void)req;
    add_cors_headers(res);
    res.status = 204;
}

// ═══════════════════════════════════════════════════════════════════════════
// SERVER MAIN (callable from terllama serve)
// ═══════════════════════════════════════════════════════════════════════════

int server_main(int argc, char** argv) {
    srand((unsigned int)time(nullptr));

    // Model directory
    std::string model_dir = std::getenv("TERLLAMA_MODEL_DIR")
        ? std::string(std::getenv("TERLLAMA_MODEL_DIR"))
        : ".";

    // Port
    int port = 8375;
    if (argc > 1) {
        port = std::stoi(argv[1]);
    } else if (const char* env_port = std::getenv("TERLLAMA_PORT")) {
        port = std::stoi(env_port);
    }

    std::string web_dir = model_dir + "/web";

    // ─── Init model ─────────────────────────────────────────────────────
    std::cout << "Initializing Terllama server..." << std::endl;
    if (!init_server(model_dir)) {
        std::cerr << "FATAL: failed to load model from " << model_dir << std::endl;
        return 1;
    }

    // ─── HTTP server ────────────────────────────────────────────────────
    httplib::Server svr;

    // Routes
    svr.Options(".*", handle_options);

    svr.Get ("/v1/models",           handle_models);
    svr.Post("/v1/chat/completions", handle_chat_completions);
    svr.Post("/v1/completions",      handle_completions);
    svr.Get ("/health",              handle_health);

    // Static file serving for web UI
    {
        std::ifstream test_web(web_dir + "/index.html");
        if (test_web.good()) {
            svr.set_base_dir(web_dir, "/");
            std::cout << "Web UI at http://0.0.0.0:" << port << "/" << std::endl;
        } else {
            svr.Get("/", [](const httplib::Request& req, httplib::Response& res) {
                (void)req;
                add_cors_headers(res);
                send_json(res, json_obj({
                    {"service",   json_encode_string("Terllama Inference Server")},
                    {"version",   json_encode_string("1.0.0")},
                    {"endpoints", json_arr({
                        json_encode_string("GET  /v1/models"),
                        json_encode_string("POST /v1/chat/completions"),
                        json_encode_string("POST /v1/completions"),
                        json_encode_string("GET  /health")
                    })}
                }));
            });
            std::cout << "No web/index.html — serving API listing at /" << std::endl;
        }
    }

    // Error handler
    svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        (void)req;
        add_cors_headers(res);
        if (res.status == 404) {
            send_error(res, "Not found: " + req.path, 404, "not_found");
        }
    });

    // Exception handler
    svr.set_exception_handler([](const httplib::Request& req, httplib::Response& res,
                                  std::exception_ptr ep) {
        (void)req;
        add_cors_headers(res);
        try {
            if (ep) std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            send_error(res, std::string("Internal error: ") + e.what(), 500);
        } catch (...) {
            send_error(res, "Unknown internal error", 500);
        }
    });

    svr.set_payload_max_length(10 * 1024 * 1024); // 10 MB max
    svr.set_read_timeout(300, 0);
    svr.set_write_timeout(300, 0);

    // ─── Banner ─────────────────────────────────────────────────────────
    std::cout << "\n"
              << "╔══════════════════════════════════════════════╗\n"
              << "║        Terllama Inference Server            ║\n"
              << "╠══════════════════════════════════════════════╣\n"
              << "║  CPU:    " << cpu_arch_name(g_model.arch);
    {
        int pad = 35 - (int)strlen(cpu_arch_name(g_model.arch));
        for (int i = 0; i < pad; i++) std::cout << ' ';
        std::cout << "║\n";
    }
    std::cout << "║  Layers: " << g_model.cfg.num_hidden_layers
              << "  Hidden: " << g_model.cfg.hidden_size
              << "  Heads: " << g_model.cfg.num_attention_heads
              << "        ║\n"
              << "║  Port:   " << port;
    {
        int pad = (port < 10000 ? 1 : 0) + (port < 1000 ? 1 : 0);
        for (int i = 0; i < 36 - pad; i++) std::cout << ' ';
        std::cout << "║\n";
    }
    std::cout << "║  API:    http://0.0.0.0:" << port << "/v1/models";
    {
        int pad = 18 - (port > 9999 ? 5 : port > 999 ? 4 : 3);
        for (int i = 0; i < pad; i++) std::cout << ' ';
        std::cout << "║\n";
    }
    std::cout << "╚══════════════════════════════════════════════╝\n"
              << std::endl;

    svr.listen("0.0.0.0", port);
    return 0;
}

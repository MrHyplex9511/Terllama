/*
 * handlers.h — HTTP route handler declarations for Terllama server
 *
 * Shared state (g_model, g_model_mutex, g_api_key) is defined in server.cpp
 * and accessed extern here. All handler implementations live in handlers.cpp.
 */
#pragma once
#include <string>
#include <vector>
#include <mutex>

#include <httplib.h>

#include "model.h"
#include "loader.h"
#include "inference.h"

// ═══════════════════════════════════════════════════════════════════════════
// SHARED GLOBAL STATE (defined in server.cpp)
// ═══════════════════════════════════════════════════════════════════════════

struct ServerModelState {
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
};

extern ServerModelState g_model;
extern std::mutex       g_model_mutex;
extern std::string      g_api_key;

// ═══════════════════════════════════════════════════════════════════════════
// MESSAGE STRUCT (chat prompt building)
// ═══════════════════════════════════════════════════════════════════════════

struct Message {
    std::string role;
    std::string content;
};

// ═══════════════════════════════════════════════════════════════════════════
// RESPONSE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

void add_cors_headers(httplib::Response& res);
std::string error_body(const std::string& message,
                       const std::string& type = "server_error");
void send_json(httplib::Response& res, const std::string& body,
               int status = 200);
void send_error(httplib::Response& res, const std::string& message,
                int status = 500,
                const std::string& type = "server_error");

// ═══════════════════════════════════════════════════════════════════════════
// TOKENIZER HELPERS (Python subprocess bridge via posix_spawn)
// ═══════════════════════════════════════════════════════════════════════════

std::vector<int> tokenize_with_python(const std::string& prompt,
                                      const std::string& helper_dir);
std::string decode_with_python(const std::vector<int>& tokens,
                               const std::string& helper_dir);

// ═══════════════════════════════════════════════════════════════════════════
// COMPLETION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

std::string build_chat_prompt(const std::vector<Message>& messages);
std::string make_id(const char* prefix = "chatcmpl");
long now_ts();

// ═══════════════════════════════════════════════════════════════════════════
// ROUTE HANDLERS
// ═══════════════════════════════════════════════════════════════════════════

void handle_models(const httplib::Request& req, httplib::Response& res);
void handle_chat_completions(const httplib::Request& req,
                             httplib::Response& res);
void handle_completions(const httplib::Request& req,
                        httplib::Response& res);
void handle_health(const httplib::Request& req, httplib::Response& res);
void handle_options(const httplib::Request& req, httplib::Response& res);

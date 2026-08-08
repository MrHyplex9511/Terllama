/*
 * protocol.h — Shared wire protocol for Terllama distributed inference.
 *
 * Terllama cluster topology:
 *
 *   Coordinator (terllama cluster) ── HTTP ──► Worker 0 (embedding + layers[0:k0))
 *                                            ── HTTP ──► Worker 1 (layers[k0:k1))
 *                                            ── HTTP ──► ... ──► Worker N (layers+lm_head)
 *
 * Roles (pipeline parallelism):
 *   - rank 0 (first):   embedding + layers[0 : start_of_next)
 *                       forward input = token id; output = hidden state
 *   - middle ranks:     layers[start : end)
 *                       forward input = hidden state; output = hidden state
 *   - last rank:        layers[start : num_hidden_layers) + final_norm + lm_head
 *                       forward input = hidden state; output = logits (vocab_size floats)
 *
 * KV cache is owned locally by each worker (per-layer), never transferred.
 * RoPE cache is deterministic from config → rebuilt identically on every worker.
 * Hidden states are float (FP32) — the same format used by inference.cpp.
 *
 * Wire format (all tensor payloads, binary, no JSON for tensors):
 *
 *   Forward request body (binary):
 *     u32 magic      = 0x544C4652 ("TLFR")
 *     u32 seq_pos    = global token position (KV index on this worker)
 *     u32 input_kind = 0 = token id (rank 0), 1 = hidden state floats
 *     u32 count      = number of floats (1 token id, or hidden_size for state)
 *     f32 data[count]
 *
 *   Forward response body (binary):
 *     u32 magic  = 0x544C4652
 *     u32 count  = hidden_size (state) or vocab_size (logits)
 *     f32 data[count]
 *
 * Control endpoints use JSON via nlohmann/json (third_party/json.hpp).
 */
#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <json.hpp>

namespace tldist {

constexpr uint32_t kForwardMagic = 0x544C4652;  // "TLFR"

// ─── Shard spec (half-open [start_layer, end_layer)) ───────────────────────
struct ShardSpec {
    int device_rank = 0;        // pipeline rank (0 = first)
    int world_size = 1;         // total workers in pipeline
    int start_layer = 0;        // inclusive
    int end_layer = 0;          // exclusive
    int n_layers = 0;           // total transformer layers in the model
    bool is_first = false;      // owns embedding (start_layer == 0)
    bool is_last = false;       // owns final_norm + lm_head (end_layer == n_layers)
};

// ─── Forward message framing ───────────────────────────────────────────────
enum : uint32_t {
    kInputToken = 0,   // rank 0: request carries a token id
    kInputHidden = 1,  // others: request carries hidden state floats
};

inline std::vector<uint8_t> pack_forward_request(int seq_pos, uint32_t input_kind,
                                                 const std::vector<float>& data) {
    std::vector<uint8_t> buf(16 + data.size() * sizeof(float));
    uint32_t magic = kForwardMagic;
    uint32_t count = (uint32_t)data.size();
    std::memcpy(buf.data(), &magic, 4);
    std::memcpy(buf.data() + 4, &seq_pos, 4);
    std::memcpy(buf.data() + 8, &input_kind, 4);
    std::memcpy(buf.data() + 12, &count, 4);
    if (!data.empty())
        std::memcpy(buf.data() + 16, data.data(), data.size() * sizeof(float));
    return buf;
}

// Returns false on magic mismatch / too-short body.
inline bool unpack_forward_request(const uint8_t* body, size_t body_len,
                                   int& seq_pos, uint32_t& input_kind,
                                   std::vector<float>& data) {
    if (body_len < 16) return false;
    uint32_t magic;
    std::memcpy(&magic, body, 4);
    if (magic != kForwardMagic) return false;
    uint32_t count;
    std::memcpy(&seq_pos, body + 4, 4);
    std::memcpy(&input_kind, body + 8, 4);
    std::memcpy(&count, body + 12, 4);
    if (body_len < 16 + (size_t)count * sizeof(float)) return false;
    data.resize(count);
    if (count) std::memcpy(data.data(), body + 16, (size_t)count * sizeof(float));
    return true;
}

inline std::vector<uint8_t> pack_forward_response(const std::vector<float>& data) {
    std::vector<uint8_t> buf(8 + data.size() * sizeof(float));
    uint32_t magic = kForwardMagic;
    uint32_t count = (uint32_t)data.size();
    std::memcpy(buf.data(), &magic, 4);
    std::memcpy(buf.data() + 4, &count, 4);
    if (!data.empty())
        std::memcpy(buf.data() + 8, data.data(), data.size() * sizeof(float));
    return buf;
}

inline bool unpack_forward_response(const uint8_t* body, size_t body_len,
                                    std::vector<float>& data) {
    if (body_len < 8) return false;
    uint32_t magic, count;
    std::memcpy(&magic, body, 4);
    if (magic != kForwardMagic) return false;
    std::memcpy(&count, body + 4, 4);
    if (body_len < 8 + (size_t)count * sizeof(float)) return false;
    data.resize(count);
    if (count) std::memcpy(data.data(), body + 8, (size_t)count * sizeof(float));
    return true;
}

// ─── JSON helpers (nlohmann) ───────────────────────────────────────────────
inline void to_json(nlohmann::json& j, const ShardSpec& s) {
    j = nlohmann::json{
        {"device_rank", s.device_rank}, {"world_size", s.world_size},
        {"start_layer", s.start_layer}, {"end_layer", s.end_layer},
        {"n_layers", s.n_layers},       {"is_first", s.is_first},
        {"is_last", s.is_last},
    };
}

inline void from_json(const nlohmann::json& j, ShardSpec& s) {
    s.device_rank = j.value("device_rank", 0);
    s.world_size = j.value("world_size", 1);
    s.start_layer = j.value("start_layer", 0);
    s.end_layer = j.value("end_layer", 0);
    s.n_layers = j.value("n_layers", 0);
    s.is_first = j.value("is_first", false);
    s.is_last = j.value("is_last", false);
}

// ─── HTTP endpoint names (worker side) ─────────────────────────────────────
inline const char* kHealthPath = "/health";
inline const char* kLoadPath = "/load";
inline const char* kForwardPath = "/forward";
inline const char* kResetPath = "/reset";

// ─── Worker control-plane messages (JSON) ──────────────────────────────────
//
// GET /health → 200 { "ok": true, "model_loaded": bool, "model_dir": string,
//                    "shard": {ShardSpec}, "ram_available_bytes": int64,
//                    "model_size_bytes": int64, "error": string }
//
// POST /load   → request JSON { "model_path": string, "shard": {ShardSpec} }
//                200 { "ok": true, "error": "" }  |  500 { "ok": false, "error": ... }
//
// POST /reset  → 200 { "ok": true }  (clears this worker's KV cache)

struct HealthInfo {
    bool ok = false;
    bool model_loaded = false;
    std::string model_dir;
    ShardSpec shard;
    int64_t ram_available_bytes = 0;
    int64_t model_size_bytes = 0;
    std::string error;
};

struct LoadRequest {
    std::string model_path;
    ShardSpec shard;
};

inline void to_json(nlohmann::json& j, const HealthInfo& h) {
    j = nlohmann::json{
        {"ok", h.ok},
        {"model_loaded", h.model_loaded},
        {"model_dir", h.model_dir},
        {"shard", h.shard},
        {"ram_available_bytes", h.ram_available_bytes},
        {"model_size_bytes", h.model_size_bytes},
        {"error", h.error},
    };
}

inline void from_json(const nlohmann::json& j, LoadRequest& r) {
    r.model_path = j.value("model_path", "");
    if (j.contains("shard"))
        r.shard = j.at("shard").get<ShardSpec>();
}

}  // namespace tldist

/*
 * coordinator.h — Terllama distributed inference coordinator.
 *
 * Partition a Terllama model across N workers (pipeline layer-sharding,
 * see distributed/protocol.h) and serve an OpenAI-compatible API.
 *
 * Worker roles:
 *   rank 0:         /forward(token_id)  -> hidden state
 *   middle workers: /forward(hidden)    -> hidden state
 *   last worker:    /forward(hidden)    -> logits (vocab_size floats)
 *
 * KV caches are worker-local; the coordinator POSTs /reset to every worker
 * before each request and drives the pipeline per token with a global
 * sequence position.
 */
#pragma once

#include "protocol.h"
#include "partitioner.h"

#include "model.h"
#include "core/tokenizer.h"
#include "core/gigatoken_wrapper.h"

#include <httplib.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

// One worker's client + shard state.
struct ClusterWorker {
    std::string url;                         // normalized "http://host:port"
    std::shared_ptr<httplib::Client> client; // per-worker HTTP client
    tldist::ShardSpec shard;                 // assigned pipeline shard
    int64_t ram_available_bytes = 0;         // reported via /health
};

class Coordinator {
public:
    Coordinator(std::vector<std::string> worker_urls,  // "host:port" entries
                std::string model_path,
                std::string api_port);

    /// Health-check workers, load shards, load the tokenizer and serve the
    /// OpenAI API. Blocks forever on the HTTP server (returns only on error).
    bool start();

    /// Pipeline-parallel generation core (used by both API endpoints).
    /// Resets all workers, prefills the prompt, then autoregressively
    /// generates up to max_tokens tokens. Returns generated token ids.
    /// on_token is invoked with each generated token id (may be empty).
    std::vector<int> generate(const std::vector<int>& prompt_tokens,
                              int max_tokens, float temperature,
                              const std::vector<int>& prev_tokens,
                              std::function<bool(int)> on_token);

private:
    std::vector<ClusterWorker> workers_;
    std::string model_path_;
    std::string api_port_;
    std::string model_name_;

    ModelConfig cfg_;
    Tokenizer tokenizer_;                        // native (GGUF / tokenizer.json)
    std::shared_ptr<GigaTokenWrapper> gigatoken_; // HF tokenizer.json encode/decode

    std::mutex gen_mutex_;   // serializes generation (v1: one stream at a time)

    /// Walk the pipeline for one token step. input_kind is kInputToken for
    /// rank 0 (input holds a single token id) or kInputHidden for the rest.
    /// Every worker receives the same seq_pos. Returns the last worker's
    /// output (hidden for middle workers, logits for the last worker).
    std::vector<float> pipeline(int seq_pos, const std::vector<float>& input,
                                uint32_t input_kind);

    // ── worker comms (throw std::runtime_error with worker index) ─────────
    tldist::HealthInfo get_health(size_t idx);
    void post_load(size_t idx, const tldist::LoadRequest& req);
    void post_reset(size_t idx);

    // ── helpers ───────────────────────────────────────────────────────────
    std::vector<int> tokenize(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    static std::string make_id(const char* prefix);
    static long now_ts();
};

/*
 * inference.h — Model inference for Terllama
 *
 * Ops: RMS norm, RoPE, KV cache, attention, MLP,
 * transformer block, model forward.
 *
 * Ternary layers dispatch via kernel_dispatch.h.
 *
 * DECLARATIONS ONLY — implementations in src/core/inference.cpp
 */
#pragma once
#include "model.h"
#include "kernel_decl.h"
#include "core/logger.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>

struct NormWeights;  // defined in loader.h

// ─── RoPE Cache ───────────────────────────────────────────────────────────
struct RoPECache {
    std::vector<float> sin, cos;
    int max_seq_len{0}, head_dim{0};
};

// ─── KV Cache ─────────────────────────────────────────────────────────────
struct KVCache {
    int max_seq{0};
    int n_layers{0}, n_kv_heads{0}, head_dim{0}, hidden_size{0};
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    std::vector<int> seq_lens;

    KVCache() = default;
    KVCache(int max_seq, int n_layers, int n_kv, int hd, int hs);
    void append(int layer, const float* k, const float* v, int seq_pos);
    void get_k(int layer, float* out);
    void get_v(int layer, float* out);
    int length(int layer) const;
};

// ─── Streaming callback ───────────────────────────────────────────────────
using StreamCallback = bool (*)(int token, float* logits, void* userdata);

// ─── Dispatch ─────────────────────────────────────────────────────────────
void ternary_linear_dispatch(const LayerData& layer,
                             const float* input, float* output);

// ─── RMS Normalization ────────────────────────────────────────────────────
void rms_norm(float* x, const float* weight, int n, float eps);

// ─── RoPE ─────────────────────────────────────────────────────────────────
RoPECache build_rope_cache(int max_seq_len, int head_dim, float theta);
void apply_rope(float* q, float* k, int seq_pos,
                int n_heads, int n_kv_heads, int head_dim,
                const RoPECache& rope);

// ─── Attention ────────────────────────────────────────────────────────────
void attention(float* x, int seq_pos, const ModelConfig& cfg,
               const LayerData& q_proj, const LayerData& k_proj,
               const LayerData& v_proj, const LayerData& o_proj,
               const RoPECache& rope, KVCache& kv_cache, int layer_idx);

// ─── SiLU + MLP ───────────────────────────────────────────────────────────
float silu(float x);
void mlp_forward(float* x, const LayerData& gate_proj,
                 const LayerData& up_proj, const LayerData& down_proj,
                 int intermediate_size);

// ─── MoTE MLP ─────────────────────────────────────────────────────────────
struct MoTELayerData;  // forward decl from model.h
void mote_mlp_forward(float* x, const MoTELayerData& mote,
                       int hidden_size, int intermediate_size);

// ─── Transformer Block ────────────────────────────────────────────────────
struct MoTELayerData;  // forward decl
void transformer_block(float* x, int seq_pos, int layer_idx,
                       const ModelConfig& cfg,
                       const std::vector<LayerData>& layers,
                       const NormWeights& norms,
                       const RoPECache& rope, KVCache& kv_cache,
                       const std::vector<MoTELayerData>* mote_layers = nullptr);

// ─── Full Model Forward ───────────────────────────────────────────────────
float* model_forward(int token, int seq_pos,
                     const ModelConfig& cfg,
                     const std::vector<float>& embedding,
                     const std::vector<LayerData>& layers,
                     const std::vector<float>& final_norm,
                     const std::vector<NormWeights>& layer_norms,
                     const RoPECache& rope, KVCache& kv_cache,
                     const std::vector<MoTELayerData>* mote_layers = nullptr);

// ─── Sampling ─────────────────────────────────────────────────────────────
int sample_argmax(const float* logits, int n);
int sample_multinomial(const float* logits, int n, float temp,
                       const std::vector<int>& prev_tokens = {},
                       float repeat_penalty = 1.0f);

// ─── Generation ───────────────────────────────────────────────────────────
bool generate_stream(const std::vector<int>& prompt_tokens,
                     int max_tokens, float temperature,
                     const ModelConfig& cfg,
                     const std::vector<float>& embedding,
                     const std::vector<LayerData>& layers,
                     const std::vector<float>& final_norm,
                     const std::vector<NormWeights>& layer_norms,
                     const RoPECache& rope,
                     StreamCallback callback, void* userdata,
                     const std::vector<MoTELayerData>* mote_layers = nullptr);

std::pair<std::vector<int>, double> generate(
    const std::vector<int>& prompt_tokens,
    int max_tokens, float temperature,
    const ModelConfig& cfg,
    const std::vector<float>& embedding,
    const std::vector<LayerData>& layers,
    const std::vector<float>& final_norm,
    const std::vector<NormWeights>& layer_norms,
    const RoPECache& rope,
    const std::vector<MoTELayerData>* mote_layers = nullptr);

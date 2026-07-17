/*
 * inference.cpp — Model inference implementations
 *
 * All function bodies moved from inference.h.
 * No 'inline' on definitions. File-scope thread_local
 * for model_forward persistent buffers.
 */
#include "inference.h"
#include "loader.h"   // find_layer_index
#include "core/logger.h"

// ─── File-scope thread_local buffers for model_forward ────────────────────
static thread_local std::vector<float> tls_x;
static thread_local std::vector<float> tls_logits;

// ═══════════════════════════════════════════════════════════════════════════
// KV CACHE
// ═══════════════════════════════════════════════════════════════════════════
KVCache::KVCache(int max_seq, int n_layers, int n_kv, int hd, int hs)
    : max_seq(max_seq), n_layers(n_layers), n_kv_heads(n_kv),
      head_dim(hd), hidden_size(hs),
      k_cache(n_layers), v_cache(n_layers), seq_lens(n_layers, 0) {
    for (int i = 0; i < n_layers; i++) {
        k_cache[i].resize(max_seq * n_kv_heads * hd);
        v_cache[i].resize(max_seq * n_kv_heads * hd);
    }
    Logger::debug("KVCache ctor: max_seq=%d layers=%d kv_heads=%d head_dim=%d",
                  max_seq, n_layers, n_kv, hd);
}

void KVCache::append(int layer, const float* k, const float* v, int seq_pos) {
    int n = n_kv_heads * head_dim;
    std::copy(k, k + n, &k_cache[layer][seq_pos * n]);
    std::copy(v, v + n, &v_cache[layer][seq_pos * n]);
    seq_lens[layer] = seq_pos + 1;
}

void KVCache::get_k(int layer, float* out) {
    int len = seq_lens[layer];
    int n = n_kv_heads * head_dim;
    std::copy(k_cache[layer].data(), k_cache[layer].data() + len * n, out);
}

void KVCache::get_v(int layer, float* out) {
    int len = seq_lens[layer];
    int n = n_kv_heads * head_dim;
    std::copy(v_cache[layer].data(), v_cache[layer].data() + len * n, out);
}

int KVCache::length(int layer) const {
    return seq_lens[layer];
}

// ═══════════════════════════════════════════════════════════════════════════
// DISPATCH: I2_S → raw FP32 → standard ternary
// ═══════════════════════════════════════════════════════════════════════════
void ternary_linear_dispatch(const LayerData& layer,
                              const float* input, float* output) {
    if (layer.has_raw_weights) {
        Logger::debug("ternary_linear(raw) out=%d in=%d",
                      layer.out_features, layer.in_features);
        #pragma omp parallel for
        for (int row = 0; row < layer.out_features; row++) {
            float sum = 0.0f;
            const float* w = &layer.raw_weights[row * layer.in_features];
            for (int j = 0; j < layer.in_features; j++)
                sum += w[j] * input[j];
            output[row] = sum;
        }
    } else if (layer.has_i2s) {
        Logger::debug("ternary_linear(i2s) out=%d in=%d",
                      layer.out_features, layer.in_features);
        ternary_linear_i2s(layer, input, output);
    } else {
        Logger::debug("ternary_linear(standard) out=%d in=%d",
                      layer.out_features, layer.in_features);
        ternary_linear(layer, input, output);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RMS NORMALIZATION
// ═══════════════════════════════════════════════════════════════════════════
void rms_norm(float* x, const float* weight, int n, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) x[i] *= weight[i] * rms;
}

// ═══════════════════════════════════════════════════════════════════════════
// ROPE (Rotary Position Embedding)
// ═══════════════════════════════════════════════════════════════════════════
RoPECache build_rope_cache(int max_seq_len, int head_dim, float theta) {
    RoPECache c;
    c.max_seq_len = max_seq_len;
    c.head_dim = head_dim;
    c.sin.resize(max_seq_len * (head_dim/2));
    c.cos.resize(max_seq_len * (head_dim/2));
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int j = 0; j < head_dim/2; j++) {
            float freq = 1.0f / std::pow(theta, 2.0f * j / head_dim);
            float angle = pos * freq;
            c.sin[pos * (head_dim/2) + j] = std::sin(angle);
            c.cos[pos * (head_dim/2) + j] = std::cos(angle);
        }
    }
    Logger::info("RoPE cache built: max_seq=%d head_dim=%d theta=%.1f",
                 max_seq_len, head_dim, theta);
    return c;
}

void apply_rope(float* q, float* k, int seq_pos,
                int n_heads, int n_kv_heads, int head_dim,
                const RoPECache& rope) {
    int hd2 = head_dim / 2;
    const float* sin = &rope.sin[seq_pos * hd2];
    const float* cos = &rope.cos[seq_pos * hd2];

    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int j = 0; j < hd2; j++) {
            float x0 = qh[2*j], x1 = qh[2*j+1];
            qh[2*j]   = x0 * cos[j] - x1 * sin[j];
            qh[2*j+1] = x0 * sin[j] + x1 * cos[j];
        }
    }
    for (int h = 0; h < n_kv_heads; h++) {
        float* kh = k + h * head_dim;
        for (int j = 0; j < hd2; j++) {
            float x0 = kh[2*j], x1 = kh[2*j+1];
            kh[2*j]   = x0 * cos[j] - x1 * sin[j];
            kh[2*j+1] = x0 * sin[j] + x1 * cos[j];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ATTENTION
// ═══════════════════════════════════════════════════════════════════════════
void attention(float* x, int seq_pos, const ModelConfig& cfg,
               const LayerData& q_proj, const LayerData& k_proj,
               const LayerData& v_proj, const LayerData& o_proj,
               const RoPECache& rope, KVCache& kv_cache, int layer_idx) {
    int H = cfg.num_attention_heads;
    int KV = cfg.num_key_value_heads;
    int HD = cfg.head_dim;
    int HS = cfg.hidden_size;
    int G = H / KV;

    Logger::debug("attention layer=%d seq_pos=%d H=%d KV=%d HD=%d",
                  layer_idx, seq_pos, H, KV, HD);

    std::vector<float> q(HS), k(KV * HD), v(KV * HD);
    ternary_linear_dispatch(q_proj, x, q.data());
    ternary_linear_dispatch(k_proj, x, k.data());
    ternary_linear_dispatch(v_proj, x, v.data());

    apply_rope(q.data(), k.data(), seq_pos, H, KV, HD, rope);

    kv_cache.append(layer_idx, k.data(), v.data(), seq_pos);
    int kv_len = kv_cache.length(layer_idx);

    std::vector<float> k_full(kv_len * KV * HD);
    std::vector<float> v_full(kv_len * KV * HD);
    kv_cache.get_k(layer_idx, k_full.data());
    kv_cache.get_v(layer_idx, v_full.data());

    std::vector<float> attn_scores(H * kv_len);
    for (int h = 0; h < H; h++) {
        int kh = h / G;
        float* qh = q.data() + h * HD;
        float* scores = attn_scores.data() + h * kv_len;
        for (int t = 0; t < kv_len; t++) {
            float* kt = k_full.data() + t * KV * HD + kh * HD;
            float s = 0.0f;
            for (int j = 0; j < HD; j++) s += qh[j] * kt[j];
            scores[t] = s / std::sqrt((float)HD);
        }
    }

    for (int h = 0; h < H; h++) {
        float* scores = attn_scores.data() + h * kv_len;
        float maxv = *std::max_element(scores, scores + kv_len);
        float sum = 0.0f;
        for (int t = 0; t < kv_len; t++) {
            scores[t] = std::exp(scores[t] - maxv);
            sum += scores[t];
        }
        for (int t = 0; t < kv_len; t++) scores[t] /= sum;
    }

    std::vector<float> attn_out(H * HD, 0.0f);
    for (int h = 0; h < H; h++) {
        int kh = h / G;
        float* outh = attn_out.data() + h * HD;
        float* scores = attn_scores.data() + h * kv_len;
        for (int t = 0; t < kv_len; t++) {
            float* vt = v_full.data() + t * KV * HD + kh * HD;
            for (int j = 0; j < HD; j++) outh[j] += scores[t] * vt[j];
        }
    }

    ternary_linear_dispatch(o_proj, attn_out.data(), x);
}

// ═══════════════════════════════════════════════════════════════════════════
// SiLU + MLP
// ═══════════════════════════════════════════════════════════════════════════
float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

void mlp_forward(float* x, const LayerData& gate_proj,
                 const LayerData& up_proj, const LayerData& down_proj,
                 int intermediate_size) {
    Logger::debug("mlp_forward intermediate=%d", intermediate_size);
    std::vector<float> gate(intermediate_size), up(intermediate_size);
    ternary_linear_dispatch(gate_proj, x, gate.data());
    ternary_linear_dispatch(up_proj, x, up.data());
    for (int i = 0; i < intermediate_size; i++)
        gate[i] = silu(gate[i]) * up[i];
    ternary_linear_dispatch(down_proj, gate.data(), x);
}

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFORMER BLOCK
// ═══════════════════════════════════════════════════════════════════════════
void transformer_block(float* x, int seq_pos, int layer_idx,
                       const ModelConfig& cfg,
                       const std::vector<LayerData>& layers,
                       const NormWeights& norms,
                       const RoPECache& rope, KVCache& kv_cache) {
    int HS = cfg.hidden_size;
    int IS = cfg.intermediate_size;

    Logger::debug("transformer_block layer=%d seq_pos=%d", layer_idx, seq_pos);

    auto layer_name = [&](const std::string& suffix) {
        return "model.layers." + std::to_string(layer_idx) + "." + suffix;
    };

    int idx_q = find_layer_index(layers, layer_name("self_attn.q_proj"));
    int idx_k = find_layer_index(layers, layer_name("self_attn.k_proj"));
    int idx_v = find_layer_index(layers, layer_name("self_attn.v_proj"));
    int idx_o = find_layer_index(layers, layer_name("self_attn.o_proj"));
    int idx_g = find_layer_index(layers, layer_name("mlp.gate_proj"));
    int idx_u = find_layer_index(layers, layer_name("mlp.up_proj"));
    int idx_d = find_layer_index(layers, layer_name("mlp.down_proj"));

    // Attention with residual
    std::vector<float> residual(x, x + HS);
    rms_norm(x, norms.input_layernorm.data(), HS, cfg.rms_norm_eps);
    attention(x, seq_pos, cfg,
              layers[idx_q], layers[idx_k], layers[idx_v], layers[idx_o],
              rope, kv_cache, layer_idx);
    for (int i = 0; i < HS; i++) x[i] += residual[i];

    // MLP with residual
    std::copy(x, x + HS, residual.begin());
    rms_norm(x, norms.post_attention_layernorm.data(), HS, cfg.rms_norm_eps);
    mlp_forward(x, layers[idx_g], layers[idx_u], layers[idx_d], IS);
    for (int i = 0; i < HS; i++) x[i] += residual[i];
}

// ═══════════════════════════════════════════════════════════════════════════
// FULL MODEL FORWARD
// ═══════════════════════════════════════════════════════════════════════════
float* model_forward(int token, int seq_pos,
                     const ModelConfig& cfg,
                     const std::vector<float>& embedding,
                     const std::vector<LayerData>& layers,
                     const std::vector<float>& final_norm,
                     const std::vector<NormWeights>& layer_norms,
                     const RoPECache& rope, KVCache& kv_cache) {
    Logger::debug("model_forward token=%d seq_pos=%d", token, seq_pos);

    if (tls_x.size() != (size_t)cfg.hidden_size)
        tls_x.resize(cfg.hidden_size);

    // Embedding lookup
    std::copy(&embedding[token * cfg.hidden_size],
              &embedding[(token + 1) * cfg.hidden_size],
              tls_x.data());

    // Transformer blocks
    for (int i = 0; i < cfg.num_hidden_layers; i++) {
        transformer_block(tls_x.data(), seq_pos, i, cfg, layers,
                          layer_norms[i], rope, kv_cache);
    }

    // Final RMSNorm
    rms_norm(tls_x.data(), final_norm.data(), cfg.hidden_size, cfg.rms_norm_eps);

    // LM head (ternary)
    int idx_lmhead = find_layer_index(layers, "lm_head");
    if (tls_logits.size() != (size_t)cfg.vocab_size)
        tls_logits.resize(cfg.vocab_size);
    ternary_linear_dispatch(layers[idx_lmhead], tls_x.data(), tls_logits.data());

    return tls_logits.data();
}

// ═══════════════════════════════════════════════════════════════════════════
// SAMPLING
// ═══════════════════════════════════════════════════════════════════════════
int sample_argmax(const float* logits, int n) {
    return (int)(std::max_element(logits, logits + n) - logits);
}

int sample_multinomial(const float* logits, int n, float temp,
                       const std::vector<int>& prev_tokens,
                       float repeat_penalty) {
    Logger::debug("sample_multinomial temp=%.2f vocab=%d", temp, n);
    std::vector<float> probs(n);
    float maxv = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        float val = logits[i];
        if (repeat_penalty > 1.0f) {
            for (int pt : prev_tokens) {
                if (i == pt) {
                    if (val < 0) val *= repeat_penalty;
                    else val /= repeat_penalty;
                    break;
                }
            }
        }
        probs[i] = std::exp((val - maxv) / temp);
        sum += probs[i];
    }
    float r = (float)rand() / RAND_MAX * sum;
    float acc = 0.0f;
    for (int i = 0; i < n; i++) {
        acc += probs[i];
        if (r < acc) return i;
    }
    return n - 1;
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION LOOP (with per-token streaming callback)
// ═══════════════════════════════════════════════════════════════════════════
bool generate_stream(const std::vector<int>& prompt_tokens,
                     int max_tokens, float temperature,
                     const ModelConfig& cfg,
                     const std::vector<float>& embedding,
                     const std::vector<LayerData>& layers,
                     const std::vector<float>& final_norm,
                     const std::vector<NormWeights>& layer_norms,
                     const RoPECache& rope,
                     StreamCallback callback, void* userdata) {

    Logger::info("generate_stream: prompt=%zu tokens max=%d temp=%.2f",
                 prompt_tokens.size(), max_tokens, temperature);

    KVCache kv_cache(cfg.max_position_embeddings, cfg.num_hidden_layers,
                     cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size);

    // Prefill
    for (int pos = 0; pos < (int)prompt_tokens.size(); pos++) {
        model_forward(prompt_tokens[pos], pos, cfg, embedding,
                      layers, final_norm, layer_norms, rope, kv_cache);
    }

    // Autoregressive
    std::vector<int> output_tokens;
    int next_token = prompt_tokens.back();
    for (int i = 0; i < max_tokens; i++) {
        int pos = (int)prompt_tokens.size() + i;
        float* logits = model_forward(next_token, pos, cfg, embedding,
                                       layers, final_norm, layer_norms, rope, kv_cache);

        std::vector<int> recent;
        for (int j = std::max(0, (int)output_tokens.size() - 8);
             j < (int)output_tokens.size(); j++)
            recent.push_back(output_tokens[j]);

        if (temperature < 0.01f)
            next_token = sample_argmax(logits, cfg.vocab_size);
        else
            next_token = sample_multinomial(logits, cfg.vocab_size, temperature,
                                            recent, 1.15f);

        output_tokens.push_back(next_token);

        if (callback && !callback(next_token, logits, userdata)) {
            Logger::info("generate_stream: aborted by callback at token %d", i);
            return false;
        }

        if (next_token == 0) break;
    }
    Logger::info("generate_stream: completed %zu output tokens", output_tokens.size());
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION LOOP (batch return, original)
// ═══════════════════════════════════════════════════════════════════════════
std::pair<std::vector<int>, double> generate(
    const std::vector<int>& prompt_tokens,
    int max_tokens, float temperature,
    const ModelConfig& cfg,
    const std::vector<float>& embedding,
    const std::vector<LayerData>& layers,
    const std::vector<float>& final_norm,
    const std::vector<NormWeights>& layer_norms,
    const RoPECache& rope) {

    Logger::info("generate: prompt=%zu tokens max=%d temp=%.2f",
                 prompt_tokens.size(), max_tokens, temperature);

    KVCache kv_cache(cfg.max_position_embeddings, cfg.num_hidden_layers,
                     cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size);

    auto t_start = std::chrono::high_resolution_clock::now();

    // Prefill
    for (int pos = 0; pos < (int)prompt_tokens.size(); pos++) {
        model_forward(prompt_tokens[pos], pos, cfg, embedding,
                      layers, final_norm, layer_norms, rope, kv_cache);
    }

    // Autoregressive
    std::vector<int> output_tokens;
    int next_token = prompt_tokens.back();
    for (int i = 0; i < max_tokens; i++) {
        int pos = (int)prompt_tokens.size() + i;
        float* logits = model_forward(next_token, pos, cfg, embedding,
                                       layers, final_norm, layer_norms, rope, kv_cache);

        std::vector<int> recent;
        for (int j = std::max(0, (int)output_tokens.size() - 8);
             j < (int)output_tokens.size(); j++)
            recent.push_back(output_tokens[j]);

        if (temperature < 0.01f)
            next_token = sample_argmax(logits, cfg.vocab_size);
        else
            next_token = sample_multinomial(logits, cfg.vocab_size, temperature,
                                            recent, 1.15f);

        output_tokens.push_back(next_token);
        if (next_token == 0) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    Logger::info("generate: %zu tokens in %.0f ms (%.1f tok/s)",
                 output_tokens.size(), total_ms,
                 1000.0 * output_tokens.size() / (total_ms > 0 ? total_ms : 1.0));

    return {output_tokens, total_ms};
}

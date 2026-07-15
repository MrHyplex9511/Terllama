/*
 * full_inference.cpp - Full SmolLM2-135M inference with ternary weights
 *
 * Loads model_decomposed.bin (ternary linear layers) and model_extra.bin
 * (embedding + RMSNorm weights), runs autoregressive generation using
 * integer add/sub for linear layers.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -mavx512f -mavx512dq -fopenmp -o full_inference full_inference.cpp -lm
 *
 * Run:
 *   ./full_inference "Hey SmolLM2! How are you today?" [max_tokens=40]
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <sstream>
#include <immintrin.h>

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════
struct ModelConfig {
    int32_t vocab_size, hidden_size, intermediate_size;
    int32_t num_hidden_layers, num_attention_heads, num_key_value_heads;
    float rms_norm_eps, rope_theta;
    int32_t max_position_embeddings;
    int32_t head_dim; // derived
};

// ═══════════════════════════════════════════════════════════════════════════
// TERNARY DECODING
// ═══════════════════════════════════════════════════════════════════════════
inline int8_t decode_ternary(const uint8_t* data, size_t pos) {
    size_t byte_idx = (pos * 2) / 8;
    int bit_offset = (pos * 2) % 8;
    unsigned int bits = (data[byte_idx] >> (6 - bit_offset)) & 0x3;
    if (bits == 0) return 0;
    if (bits == 1) return 1;
    return -1; // bits == 2
}

// ═══════════════════════════════════════════════════════════════════════════
// LAYER DATA (from model_decomposed.bin)
// ═══════════════════════════════════════════════════════════════════════════
// Fused bitplane term: single uint32 per word = (nz << 16) | neg
// One cache line touch instead of two.
struct alignas(64) BitplaneTerm {
    int32_t alpha_exp;
    std::vector<uint32_t> combined;  // upper 16=nz, lower 16=neg
    size_t n_elements;
};

struct alignas(64) LayerData {
    std::string name;
    int32_t out_features, in_features;
    int32_t num_terms;
    std::vector<BitplaneTerm> terms;
};

// ═══════════════════════════════════════════════════════════════════════════
// BINARY LOADERS
// ═══════════════════════════════════════════════════════════════════════════
ModelConfig load_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    ModelConfig cfg;
    int32_t vs, hs, is, nl, nah, nkv, mpe;
    float rne, rt;
    f.read(reinterpret_cast<char*>(&vs), 4);
    f.read(reinterpret_cast<char*>(&hs), 4);
    f.read(reinterpret_cast<char*>(&is), 4);
    f.read(reinterpret_cast<char*>(&nl), 4);
    f.read(reinterpret_cast<char*>(&nah), 4);
    f.read(reinterpret_cast<char*>(&nkv), 4);
    f.read(reinterpret_cast<char*>(&rne), 4);
    f.read(reinterpret_cast<char*>(&rt), 4);
    f.read(reinterpret_cast<char*>(&mpe), 4);
    cfg = {vs, hs, is, nl, nah, nkv, rne, rt, mpe, hs / nah};
    return cfg;
}

std::vector<float> load_embedding(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    f.seekg(4*9, std::ios::beg); // skip 9 int32/float config fields
    std::vector<float> emb(cfg.vocab_size * cfg.hidden_size);
    f.read(reinterpret_cast<char*>(emb.data()), emb.size() * sizeof(float));
    return emb;
}

std::vector<float> load_final_norm(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    // Skip config (36 bytes) + embedding
    f.seekg(36 + cfg.vocab_size * cfg.hidden_size * 4, std::ios::beg);
    std::vector<float> fn(cfg.hidden_size);
    f.read(reinterpret_cast<char*>(fn.data()), fn.size() * sizeof(float));
    return fn;
}

struct NormWeights {
    std::vector<float> input_layernorm;
    std::vector<float> post_attention_layernorm;
};

std::vector<NormWeights> load_layer_norms(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    // Skip config (36) + embedding (vocab*hidden*4) + final norm (hidden*4)
    int64_t offset = 36 + (int64_t)cfg.vocab_size * cfg.hidden_size * 4 + cfg.hidden_size * 4;
    f.seekg(offset, std::ios::beg);
    std::vector<NormWeights> norms(cfg.num_hidden_layers);
    for (int i = 0; i < cfg.num_hidden_layers; i++) {
        norms[i].input_layernorm.resize(cfg.hidden_size);
        norms[i].post_attention_layernorm.resize(cfg.hidden_size);
        f.read(reinterpret_cast<char*>(norms[i].input_layernorm.data()), cfg.hidden_size * 4);
        f.read(reinterpret_cast<char*>(norms[i].post_attention_layernorm.data()), cfg.hidden_size * 4);
    }
    return norms;
}

int find_layer_index(const std::vector<LayerData>& layers, const std::string& name) {
    for (int i = 0; i < (int)layers.size(); i++)
        if (layers[i].name == name) return i;
    std::cerr << "Layer not found: " << name << std::endl;
    exit(1);
    return -1;
}

std::vector<LayerData> load_decomposed_layers(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0xDEADBEEF) { std::cerr << "Bad magic\n"; exit(1); }
    uint32_t num_layers;
    f.read(reinterpret_cast<char*>(&num_layers), 4);
    std::vector<LayerData> layers(num_layers);
    for (uint32_t i = 0; i < num_layers; i++) {
        auto& ld = layers[i];
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        ld.name.resize(name_len);
        f.read(&ld.name[0], name_len);
        f.read(reinterpret_cast<char*>(&ld.out_features), 4);
        f.read(reinterpret_cast<char*>(&ld.in_features), 4);
        f.read(reinterpret_cast<char*>(&ld.num_terms), 4);
        ld.terms.resize(ld.num_terms);
        for (int t = 0; t < ld.num_terms; t++) {
            auto& term = ld.terms[t];
            size_t n_elements = (size_t)ld.out_features * ld.in_features;
            size_t n_bytes = (n_elements * 2 + 7) / 8;
            f.read(reinterpret_cast<char*>(&term.alpha_exp), 4);
            // Read packed data into temp buffer
            std::vector<uint8_t> packed(n_bytes);
            f.read(reinterpret_cast<char*>(packed.data()), n_bytes);
            // Convert to coalesced bitplane format
            // combined[word] = (nz << 16) | neg, one uint32 per 16-element chunk
            term.n_elements = n_elements;
            int words_per_row = (ld.in_features + 15) / 16;
            size_t n_words = (size_t)ld.out_features * words_per_row;
            term.combined.assign(n_words, 0);
            for (int i = 0; i < ld.out_features; i++) {
                for (int j = 0; j < ld.in_features; j++) {
                    size_t pos = (size_t)i * ld.in_features + j;
                    int8_t tv = decode_ternary(packed.data(), pos);
                    int word = j / 16;
                    int bit = j % 16;
                    size_t abs_word = (size_t)i * words_per_row + word;
                    if (tv == 1) term.combined[abs_word] |= (1 << (bit + 16));  // nz only
                    else if (tv == -1) term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit);  // nz + neg
                }
            }
        }
    }
    return layers;
}

// ═══════════════════════════════════════════════════════════════════════════
// TERNARY LINEAR LAYER (add/sub kernel)
// ═══════════════════════════════════════════════════════════════════════════
void ternary_linear_naive(const LayerData& layer, const float* input, float* output) {
    const int32_t out_f = layer.out_features;
    const int32_t in_f = layer.in_features;
    int words_per_row = (in_f + 15) / 16;
    std::fill(output, output + out_f, 0.0f);

    for (int t = 0; t < layer.num_terms; t++) {
        int32_t ae = layer.terms[t].alpha_exp;
        if (ae == -128) continue;
        const auto& comb = layer.terms[t].combined;
        for (int i = 0; i < out_f; i++) {
            float sum = 0.0f;
            size_t base = (size_t)i * words_per_row;
            for (int j = 0; j < in_f; j++) {
                int word = j / 16;
                int bit = j % 16;
                uint32_t c = comb[base + word];
                bool is_nz = (c >> (bit + 16)) & 1;
                bool is_neg = (c >> bit) & 1;
                if (is_nz) sum += is_neg ? -input[j] : input[j];
            }
            output[i] += std::ldexp(sum, ae);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AVX-512 TERNARY LINEAR KERNEL
// ═══════════════════════════════════════════════════════════════════════════
#ifdef __AVX512F__
// ─── AVX-512 kernel: coalesced memory (row-major, fused combined[] array) ───
//
// 1. Fused combined[] = (nz << 16) | neg → ONE uint32 load per word (was 2)
// 2. Row-major outer: process ALL terms per row → input loaded ONCE (was N×)
// 3. __restrict__ + aligned loads for prefetcher hints
void ternary_linear_avx512(const LayerData& __restrict__ layer,
                           const float* __restrict__ input,
                           float* __restrict__ output) {
    const int32_t out_f = layer.out_features;
    const int32_t in_f = layer.in_features;
    const int32_t N = layer.num_terms;

    // Pre-compute per-term: alpha_exp, words_per_row, stride
    // Avoid re-fetching from layer.terms[t] in the inner loop
    struct TermInfo {
        int32_t ae;
        size_t stride;  // words per row in this term's combined[]
    };
    TermInfo terms[32];  // enough for up to 32 terms (we use 12-15)
    const uint32_t* term_data[32];
    int n_active = 0;
    for (int t = 0; t < N; t++) {
        int32_t ae = layer.terms[t].alpha_exp;
        if (ae == -128) continue;
        terms[n_active].ae = ae;
        terms[n_active].stride = (size_t)((in_f + 15) / 16);
        term_data[n_active] = layer.terms[t].combined.data();
        n_active++;
    }
    if (n_active == 0) { std::fill(output, output + out_f, 0.0f); return; }

    int words_per_row = (in_f + 15) / 16;
    int rem = in_f % 16;
    int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
    uint32_t tail_mask = rem > 0 ? (uint32_t)((1 << rem) - 1) : 0;

    std::fill(output, output + out_f, 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < out_f; i++) {
        // Per-term SIMD accumulators (stay in zmm registers)
        __m512 vacc[32];
        for (int t = 0; t < n_active; t++) vacc[t] = _mm512_setzero_ps();

        // Process ALL terms per word → input loaded ONCE
        for (int w = 0; w < full_words; w++) {
            __m512 v = _mm512_loadu_ps(&input[w * 16]);

            for (int t = 0; t < n_active; t++) {
                size_t idx = i * terms[t].stride + w;
                uint32_t comb = term_data[t][idx];
                uint16_t nzw = comb >> 16;
                uint16_t negw = comb & 0xFFFF;
                __mmask16 mask_add = nzw & ~negw;
                __mmask16 mask_sub = nzw & negw;
                vacc[t] = _mm512_mask_add_ps(vacc[t], mask_add, vacc[t], v);
                vacc[t] = _mm512_mask_sub_ps(vacc[t], mask_sub, vacc[t], v);
            }
        }

        // Handle tail (last partial word)
        if (rem > 0) {
            __m512 v = _mm512_loadu_ps(&input[full_words * 16]);
            for (int t = 0; t < n_active; t++) {
                size_t idx = i * terms[t].stride + full_words;
                uint32_t comb = term_data[t][idx] & (tail_mask | (tail_mask << 16));
                uint16_t nzw = comb >> 16;
                uint16_t negw = comb & 0xFFFF;
                __mmask16 mask_add = nzw & ~negw;
                __mmask16 mask_sub = nzw & negw;
                vacc[t] = _mm512_mask_add_ps(vacc[t], mask_add, vacc[t], v);
                vacc[t] = _mm512_mask_sub_ps(vacc[t], mask_sub, vacc[t], v);
            }
        }

        // Reduce once per term, apply alpha, sum
        float result = 0.0f;
        for (int t = 0; t < n_active; t++) {
            result += std::ldexp(_mm512_reduce_add_ps(vacc[t]), terms[t].ae);
        }
        output[i] = result;
    }
}
#endif // __AVX512F__

// ═══════════════════════════════════════════════════════════════════════════
// TERNARY LINEAR DISPATCH
// ═══════════════════════════════════════════════════════════════════════════
inline void ternary_linear(const LayerData& layer, const float* input, float* output) {
#ifdef __AVX512F__
    ternary_linear_avx512(layer, input, output);
#else
    ternary_linear_naive(layer, input, output);
#endif
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
struct RoPECache {
    std::vector<float> sin, cos;
    int max_seq_len, head_dim;
};

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
    return c;
}

void apply_rope(float* q, float* k, int seq_pos, int n_heads, int n_kv_heads,
                int head_dim, const RoPECache& rope) {
    // Apply RoPE to Q (n_heads * head_dim) and K (n_kv_heads * head_dim)
    // For each head, apply rotation to pairs (2i, 2i+1)
    int hd2 = head_dim / 2;
    const float* sin = &rope.sin[seq_pos * hd2];
    const float* cos = &rope.cos[seq_pos * hd2];

    // Q heads
    for (int h = 0; h < n_heads; h++) {
        float* qh = q + h * head_dim;
        for (int j = 0; j < hd2; j++) {
            float x0 = qh[2*j], x1 = qh[2*j+1];
            qh[2*j]   = x0 * cos[j] - x1 * sin[j];
            qh[2*j+1] = x0 * sin[j] + x1 * cos[j];
        }
    }
    // K heads
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
// KV CACHE
// ═══════════════════════════════════════════════════════════════════════════
struct KVCache {
    int max_seq;
    int n_layers, n_kv_heads, head_dim, hidden_size;
    std::vector<std::vector<float>> k_cache; // [layer][pos * n_kv_heads * head_dim]
    std::vector<std::vector<float>> v_cache; // [layer][pos * n_kv_heads * head_dim]
    std::vector<int> seq_lens;

    KVCache(int max_seq, int n_layers, int n_kv, int hd, int hs)
        : max_seq(max_seq), n_layers(n_layers), n_kv_heads(n_kv),
          head_dim(hd), hidden_size(hs),
          k_cache(n_layers), v_cache(n_layers), seq_lens(n_layers, 0) {
        for (int i = 0; i < n_layers; i++) {
            k_cache[i].resize(max_seq * n_kv_heads * hd);
            v_cache[i].resize(max_seq * n_kv_heads * hd);
        }
    }

    void append(int layer, const float* k, const float* v, int seq_pos) {
        int n = n_kv_heads * head_dim;
        std::copy(k, k + n, &k_cache[layer][seq_pos * n]);
        std::copy(v, v + n, &v_cache[layer][seq_pos * n]);
        seq_lens[layer] = seq_pos + 1;
    }

    void get_k(int layer, float* out) {
        int len = seq_lens[layer];
        int n = n_kv_heads * head_dim;
        std::copy(k_cache[layer].data(), k_cache[layer].data() + len * n, out);
    }

    void get_v(int layer, float* out) {
        int len = seq_lens[layer];
        int n = n_kv_heads * head_dim;
        std::copy(v_cache[layer].data(), v_cache[layer].data() + len * n, out);
    }

    int length(int layer) const { return seq_lens[layer]; }
};

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
    int G = H / KV; // groups

    // Project Q, K, V
    std::vector<float> q(HS), k(KV * HD), v(KV * HD);
    ternary_linear(q_proj, x, q.data());
    ternary_linear(k_proj, x, k.data());
    ternary_linear(v_proj, x, v.data());

    // Apply RoPE
    apply_rope(q.data(), k.data(), seq_pos, H, KV, HD, rope);

    // Append to KV cache
    kv_cache.append(layer_idx, k.data(), v.data(), seq_pos);
    int kv_len = kv_cache.length(layer_idx);

    // Retrieve full K, V
    std::vector<float> k_full(kv_len * KV * HD);
    std::vector<float> v_full(kv_len * KV * HD);
    kv_cache.get_k(layer_idx, k_full.data());
    kv_cache.get_v(layer_idx, v_full.data());

    // Scaled dot-product attention
    // For each Q head, attend over all KV heads (with GQA grouping)
    std::vector<float> attn_scores(H * kv_len);
    for (int h = 0; h < H; h++) {
        int kh = h / G; // corresponding KV head index
        float* qh = q.data() + h * HD;
        float* scores = attn_scores.data() + h * kv_len;
        for (int t = 0; t < kv_len; t++) {
            float* kt = k_full.data() + t * KV * HD + kh * HD;
            float s = 0.0f;
            for (int j = 0; j < HD; j++) s += qh[j] * kt[j];
            scores[t] = s / std::sqrt((float)HD);
        }
    }

    // Softmax
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

    // Weighted sum of V
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

    // Output projection
    ternary_linear(o_proj, attn_out.data(), x);
}

// ═══════════════════════════════════════════════════════════════════════════
// SiLU ACTIVATION
// ═══════════════════════════════════════════════════════════════════════════
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// ═══════════════════════════════════════════════════════════════════════════
// MLP
// ═══════════════════════════════════════════════════════════════════════════
void mlp_forward(float* x, const LayerData& gate_proj, const LayerData& up_proj,
                 const LayerData& down_proj, int intermediate_size) {
    std::vector<float> gate(intermediate_size), up(intermediate_size);
    ternary_linear(gate_proj, x, gate.data());
    ternary_linear(up_proj, x, up.data());
    // SiLU(gate) * up (element-wise)
    for (int i = 0; i < intermediate_size; i++)
        gate[i] = silu(gate[i]) * up[i];
    // Down projection
    ternary_linear(down_proj, gate.data(), x);
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

    // Build layer name prefix
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
    static thread_local std::vector<float> x(cfg.hidden_size);
    static thread_local bool inited = false;
    if (!inited) { x.resize(cfg.hidden_size); inited = true; }

    // Embedding lookup
    std::copy(&embedding[token * cfg.hidden_size],
              &embedding[(token + 1) * cfg.hidden_size],
              x.data());

    // Transformer blocks
    for (int i = 0; i < cfg.num_hidden_layers; i++) {
        transformer_block(x.data(), seq_pos, i, cfg, layers,
                         layer_norms[i], rope, kv_cache);
    }

    // Final RMSNorm
    rms_norm(x.data(), final_norm.data(), cfg.hidden_size, cfg.rms_norm_eps);

    // LM head (ternary)
    int idx_lmhead = find_layer_index(layers, "lm_head");
    static thread_local std::vector<float> logits(cfg.vocab_size);
    if (logits.size() != (size_t)cfg.vocab_size) logits.resize(cfg.vocab_size);
    ternary_linear(layers[idx_lmhead], x.data(), logits.data());

    return logits.data();
}

// ═══════════════════════════════════════════════════════════════════════════
// GENERATION
// ═══════════════════════════════════════════════════════════════════════════
int sample_argmax(const float* logits, int n) {
    return (int)(std::max_element(logits, logits + n) - logits);
}

int sample_multinomial(const float* logits, int n, float temp) {
    // Apply temperature and sample
    std::vector<float> probs(n);
    float maxv = *std::max_element(logits, logits + n);
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = std::exp((logits[i] - maxv) / temp);
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
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
// ─── Helper: write prompt to file, call Python tokenizer, read tokens back ─
std::vector<int> tokenize_with_python(const std::string& prompt) {
    // Write prompt to temp file (avoids shell escaping hell)
    std::string prompt_file = "/tmp/ternary_prompt.txt";
    std::string token_file = "/tmp/ternary_tokens.txt";
    {
        std::ofstream pf(prompt_file);
        pf << prompt;
    }
    std::string helper = "/media/extra/Symlinks/BitNet/utils/export/tokenize_helper.py";
    std::string cmd = "python3 " + helper;
    int ret = system(cmd.c_str());
    if (ret != 0) { std::cerr << "Tokenization failed\n"; exit(1); }

    std::vector<int> tokens;
    std::ifstream tf(token_file);
    int tid;
    while (tf >> tid) tokens.push_back(tid);
    return tokens;
}

std::string decode_with_python(const std::vector<int>& tokens) {
    std::string token_file = "/tmp/ternary_decode_in.txt";
    std::string out_file = "/tmp/ternary_decode_out.txt";
    {
        std::ofstream tf(token_file);
        for (size_t i = 0; i < tokens.size(); i++) {
            if (i > 0) tf << " ";
            tf << tokens[i];
        }
    }
    std::string helper = "/media/extra/Symlinks/BitNet/utils/export/decode_helper.py";
    std::string cmd = "python3 " + helper;
    int ret = system(cmd.c_str());
    if (ret != 0) { std::cerr << "Decoding failed\n"; return "?"; }

    std::ifstream of(out_file);
    std::stringstream ss;
    ss << of.rdbuf();
    return ss.str();
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " \"prompt\" [max_tokens=40] [temp=0.7]" << std::endl;
        return 1;
    }

    std::string prompt = argv[1];
    int max_tokens = (argc > 2) ? std::stoi(argv[2]) : 40;
    float temperature = (argc > 3) ? std::stof(argv[3]) : 0.7f;
    std::string ex_dir = "/media/extra/Symlinks/BitNet/utils/export";

    srand(42);

    // ─── Load everything ────────────────────────────────────────────────
    std::cout << "Loading config..." << std::endl;
    auto cfg = load_config(ex_dir + "/model_extra.bin");

    std::cout << "Loading embedding (" << cfg.vocab_size << "\xC3\x97" << cfg.hidden_size << ")..." << std::endl;
    auto embedding = load_embedding(ex_dir + "/model_extra.bin", cfg);

    std::cout << "Loading " << cfg.num_hidden_layers << " layer norms..." << std::endl;
    auto layer_norms = load_layer_norms(ex_dir + "/model_extra.bin", cfg);

    std::cout << "Loading final norm..." << std::endl;
    auto final_norm = load_final_norm(ex_dir + "/model_extra.bin", cfg);

    std::cout << "Loading decomposed linear layers..." << std::endl;
    auto layers = load_decomposed_layers(ex_dir + "/model_decomposed.bin");
    std::cout << "  Loaded " << layers.size() << " layers." << std::endl;

    std::cout << "Building RoPE cache..." << std::endl;
    auto rope = build_rope_cache(cfg.max_position_embeddings, cfg.head_dim, cfg.rope_theta);

    // ─── Tokenize ───────────────────────────────────────────────────────
    std::cout << "\nTokenizing prompt..." << std::endl;
    auto prompt_tokens = tokenize_with_python(prompt);
    std::cout << "Prompt tokens (" << prompt_tokens.size() << "): ";
    for (int t : prompt_tokens) std::cout << t << " ";
    std::cout << std::endl;

    // ─── Generation ────────────────────────────────────────────────────
    std::cout << "\n=== Generating (ternary weights) ===" << std::endl;
    std::cout << "Prompt: " << prompt << std::endl;

    KVCache kv_cache(cfg.max_position_embeddings, cfg.num_hidden_layers,
                     cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size);

    auto t_start = std::chrono::high_resolution_clock::now();
    size_t mem_before = (size_t)layers.size() * sizeof(LayerData) + embedding.size() * sizeof(float);

    // Prefill
    for (int pos = 0; pos < (int)prompt_tokens.size(); pos++) {
        model_forward(prompt_tokens[pos], pos, cfg, embedding,
                      layers, final_norm, layer_norms, rope, kv_cache);
    }

    // Autoregressive generation
    std::vector<int> output_tokens;
    int next_token = prompt_tokens.back();
    for (int i = 0; i < max_tokens; i++) {
        int pos = (int)prompt_tokens.size() + i;
        float* logits = model_forward(next_token, pos, cfg, embedding,
                                      layers, final_norm, layer_norms, rope, kv_cache);

        if (temperature < 0.01f)
            next_token = sample_argmax(logits, cfg.vocab_size);
        else
            next_token = sample_multinomial(logits, cfg.vocab_size, temperature);

        output_tokens.push_back(next_token);
        if (next_token == 0) break;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double ms_per_token = total_ms / (prompt_tokens.size() + output_tokens.size());

    // ─── Decode ─────────────────────────────────────────────────────────
    std::vector<int> all_tokens = prompt_tokens;
    all_tokens.insert(all_tokens.end(), output_tokens.begin(), output_tokens.end());
    std::string decoded = decode_with_python(all_tokens);
    std::string prompt_decoded = decode_with_python(prompt_tokens);

    // ─── Report ─────────────────────────────────────────────────────────
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Full response:" << std::endl;
    std::cout << decoded << std::endl;

    // Extract generated part (after prompt)
    std::string generated_only;
    if (decoded.size() > prompt_decoded.size()) {
        generated_only = decoded.substr(prompt_decoded.size());
        while (!generated_only.empty() && generated_only[0] == ' ') generated_only.erase(0, 1);
    } else {
        // If the model repeated the prompt, try to find it
        size_t pos = decoded.find(prompt_decoded);
        if (pos != std::string::npos)
            generated_only = decoded.substr(pos + prompt_decoded.size());
    }
    std::cout << "\nGenerated text: \"" << generated_only << "\"" << std::endl;

    std::cout << "\n── Performance ──" << std::endl;
    std::cout << "  Total time:       " << total_ms << " ms" << std::endl;
    std::cout << "  Prompt tokens:    " << prompt_tokens.size() << std::endl;
    std::cout << "  Generated tokens: " << output_tokens.size() << std::endl;
    std::cout << "  Total tokens:     " << (prompt_tokens.size() + output_tokens.size()) << std::endl;
    std::cout << "  ms/token:         " << ms_per_token << std::endl;
    std::cout << "  tokens/sec:       " << (1000.0 / ms_per_token) << std::endl;

    // Memory
    size_t file_layers = 376584171;
    size_t file_extra = 113400000;
    size_t mem_kv_max = (size_t)cfg.max_position_embeddings * cfg.num_hidden_layers
                        * cfg.num_key_value_heads * cfg.head_dim * 2 * 4;
    std::cout << "\n── Memory ──" << std::endl;
    std::cout << "  Ternary weights:  " << (file_layers / 1e6) << " MB" << std::endl;
    std::cout << "  Embedding+norms:  " << (file_extra / 1e6) << " MB" << std::endl;
    std::cout << "  KV cache (max):   " << (mem_kv_max / 1e6) << " MB" << std::endl;
    std::cout << "  Total:            " << ((file_layers + file_extra + mem_kv_max) / 1e6) << " MB" << std::endl;
    std::cout << "  FP32 baseline:    540 MB" << std::endl;

    return 0;
}

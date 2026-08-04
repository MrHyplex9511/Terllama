/*
 * model.h: Shared model data structures for Terllama
 *
 * Binary format, layer metadata, and bitplane encoding
 * for all kernels and inference pipeline.
 */
#pragma once
#include <cstdint>
#include <string>
#include <vector>

// ═══════════════════════════════════════════════════════════════════════════
// MODEL CONFIG
// ═══════════════════════════════════════════════════════════════════════════
struct ModelConfig {
    int32_t vocab_size, hidden_size, intermediate_size;
    int32_t num_hidden_layers, num_attention_heads, num_key_value_heads;
    float rms_norm_eps, rope_theta;
    int32_t max_position_embeddings;
    int32_t head_dim; // derived
    int32_t eos_token_id = 0; // default 0 for backward compat
};

// ═══════════════════════════════════════════════════════════════════════════
// BITPLANE ENCODING
// ═══════════════════════════════════════════════════════════════════════════
// Fused bitplane term: single uint32 per 16-element word
//   combined = (nz << 16) | neg
//   nz bit set  → element is non-zero (ternary ±1)
//   neg bit set → element is -1 (if nz also set)
//   neg bit clear → element is +1 (if nz also set)
//   nz bit clear → element is 0
//
// One cache line touch instead of two (was separate nz/neg arrays).
struct alignas(64) BitplaneTerm {
    int32_t alpha_exp{0};
    size_t n_elements{0};
    std::vector<uint32_t> combined;  // upper 16=nz, lower 16=neg
};

// TQ1.0 packed weight block: 120 elements → 24 bytes Base-3 encoded (5 trits/byte)
// ~1.58 bits/weight, 27% smaller than block-scaled ternary (4 trits/byte, 2 bits/weight)
struct TQ1Block {
    std::vector<uint8_t> packed;  // Base-3 encoded (5 trits/byte)
    float scale{0.0f};            // per-block scale
};

// ─── TQ1.0 pack/unpack utilities (Base-3 encoding, 5 trits/byte) ─────────────
inline int tq1_encode_5_trits(const int8_t trits[5]) {
    // Map -1→0, 0→1, +1→2, then encode in Base-3
    int val = 0;
    for (int i = 4; i >= 0; i--) {
        val = val * 3 + (trits[i] + 1);  // -1→0, 0→1, +1→2
    }
    return val;  // 0..242, fits in uint8_t
}

inline void tq1_decode_byte(uint8_t byte, int8_t trits[5]) {
    // Decode Base-3 byte into 5 ternary values
    int val = byte;
    for (int i = 0; i < 5; i++) {
        int digit = val % 3;
        trits[i] = (int8_t)(digit - 1);  // 0→-1, 1→0, 2→+1
        val /= 3;
    }
}

// Decode a TQ1 block (120 elements → 24 bytes) into ternary int8 array
inline void tq1_decode_block(const uint8_t* packed, int8_t* ternary, int n) {
    int n_bytes = (n + 4) / 5;  // ceil(n/5)
    for (int i = 0; i < n_bytes; i++) {
        int8_t trits[5];
        tq1_decode_byte(packed[i], trits);
        for (int j = 0; j < 5 && i * 5 + j < n; j++) {
            ternary[i * 5 + j] = trits[j];
        }
    }
}

// Block-scaled ternary weight block: 128 elements → 32 bytes codes + 4 bytes scale
// Codes: 4 ternary values per byte, {0=-1, 1=0, 2=+1} (MSB first).
struct BlockTerm {
    std::vector<uint8_t> packed;  // packed codes (4 values/byte)
    float scale{0.0f};            // per-block scale
};

struct alignas(64) LayerData {
    std::string name;
    int32_t out_features{0}, in_features{0};
    int32_t num_terms{0};
    std::vector<BitplaneTerm> terms;

    // TQ1.0 storage (most compact: ~1.58 bits/weight)
    bool has_tq1{false};
    std::vector<TQ1Block> tq1_blocks;
    int tq1_qk{120};  // 120 elements per block → 24 bytes packed

    // Block-scaled ternary storage (ALS): each term is a full set of
    // BlockTerm (out_features*n_blocks). Single-term layers are a 1-element set.
    bool has_blocks{false};
    int block_qk{128};                                  // block size (elements)
    std::vector<std::vector<BlockTerm>> block_terms;    // [term] → [out_f * n_blocks]

    // ─── Lazy contiguous cache for block kernels ────────────────────────
    // Built once on first matmul (ternary_linear_blocks) instead of rebuilding
    // the packed layout on every call. Saves ~40MB of memcpy + 200+
    // allocations per generated token. Mutable: built lazily, reused across
    // tokens. Safe because a single generate loop runs serially and each
    // server request operates on its own snapshot copy.
    // Per-term: [term][row] pointer arrays; term 0 = single-term path.
    mutable bool term_cache_built{false};
    mutable std::vector<std::vector<uint8_t>> term_contig_data;      // [term] flat packed
    mutable std::vector<std::vector<float>> term_contig_scales;      // [term] flat scales
    mutable std::vector<std::vector<const uint8_t*>> term_block_data;   // [term][row]
    mutable std::vector<std::vector<const float*>> term_block_scales;   // [term][row]

    // Raw FP32 weights (for layers unsuitable for ternary, e.g. lm_head)
    bool has_raw_weights{false};
    std::vector<float> raw_weights;    // flat [out_features * in_features]
};

// ═══════════════════════════════════════════════════════════════════════════
// MoTE — Mixture of Ternary Experts
// ═══════════════════════════════════════════════════════════════════════════
struct MoTEConfig {
    int num_experts = 4;
    int top_k = 1;
    bool use_shared_expert = true;
};

struct MoTELayerData {
    // Shared expert (original FFN — always active)
    LayerData gate_proj, up_proj, down_proj;

    // Expert config
    int num_experts{0};
    int top_k{1};

    // K routed ternary experts (each has own gate/up/down)
    std::vector<LayerData> expert_gate;
    std::vector<LayerData> expert_up;
    std::vector<LayerData> expert_down;

    // Router weights: FP32 [hidden_size × num_experts], row-major
    std::vector<float> router_weight;
    float router_scale{1.0f};

    // Flag
    bool is_mote{false};
};

// ═══════════════════════════════════════════════════════════════════════════
// CPU ARCHITECTURE ENUM
// ═══════════════════════════════════════════════════════════════════════════
enum class CPUArch : uint8_t {
    UNKNOWN = 0,
    X86_64_SCALAR,   // No SIMD
    X86_64_SSE42,    // SSE4.2  (128-bit)
    X86_64_AVX,      // AVX     (256-bit, no FMA)
    X86_64_AVX2,     // AVX2+FMA(256-bit)
    X86_64_AVX512,   // AVX-512 (512-bit)
    ARM64_NEON,      // ARM64 NEON (128-bit)
    ARM64_SCALAR,    // ARM64 scalar fallback
};

inline const char* cpu_arch_name(CPUArch a) {
    switch (a) {
        case CPUArch::UNKNOWN:      return "unknown";
        case CPUArch::X86_64_SCALAR: return "x86-64 scalar";
        case CPUArch::X86_64_SSE42:  return "x86-64 SSE4.2";
        case CPUArch::X86_64_AVX:    return "x86-64 AVX";
        case CPUArch::X86_64_AVX2:   return "x86-64 AVX2+FMA";
        case CPUArch::X86_64_AVX512: return "x86-64 AVX-512";
        case CPUArch::ARM64_NEON:    return "ARM64 NEON";
        case CPUArch::ARM64_SCALAR:  return "ARM64 scalar";
    }
    return "?";
}

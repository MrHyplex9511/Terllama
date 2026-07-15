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

// I2_S packed weight block: 128 elements → 32 bytes codes + 4 bytes scale
struct I2SBlock {
    std::vector<uint8_t> packed;  // packed I2_S codes (4 values/byte)
    float scale{0.0f};            // per-block scale
};

struct alignas(64) LayerData {
    std::string name;
    int32_t out_features{0}, in_features{0};
    int32_t num_terms{0};
    std::vector<BitplaneTerm> terms;

    // I2_S direct storage (optional, for new kernel path)
    bool has_i2s{false};
    std::vector<I2SBlock> i2s_blocks;  // one block per (out_feature, 128-element chunk)
    int i2s_qk{128};                    // block size

    // Raw FP32 weights (for layers unsuitable for ternary, e.g. lm_head)
    bool has_raw_weights{false};
    std::vector<float> raw_weights;    // flat [out_features * in_features]
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

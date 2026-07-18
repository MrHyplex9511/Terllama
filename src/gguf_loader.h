/*
 * gguf_loader.h — GGUF format parser + Q2_0 decoder for Terllama
 *
 * Parses GGUF files (v3) containing Q2_0 (g128) ternary weights,
 * converts to I2_S format in memory for use with existing kernels.
 *
 * Supports:
 *   - GGUF v3 header + metadata
 *   - Q2_0 (g128) quantized tensor decoding → I2_S blocks
 *   - F32/F16 unquantized tensor extraction (embedding, norms)
 *   - Qwen3/LLaMA tensor naming conventions
 */
#pragma once
#include "model.h"
#include "core/tokenizer.h"
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════
// GGUF / GGML TYPE CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

constexpr uint32_t GGUF_MAGIC       = 0x46554747u;  // "GGUF"
constexpr uint32_t GGUF_VERSION      = 3;
constexpr uint32_t Q2_0_BLOCK_SIZE   = 128;           // Bonsai g128 variant
constexpr uint32_t Q2_0_BLOCK_BYTES  = 34;            // 2 (FP16 scale) + 32 (codes)

enum GGMLType : uint32_t {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q2_0 = 10,
};

// GGUF metadata value types
enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

// ═══════════════════════════════════════════════════════════════════════════
// TENSOR INFO FROM GGUF
// ═══════════════════════════════════════════════════════════════════════════

struct GGUFTensorInfo {
    std::string name;
    uint32_t type;              // GGMLType
    std::vector<uint64_t> dims; // [out_features, in_features, ...]
    uint64_t offset;            // absolute file offset to tensor data
};

// ═══════════════════════════════════════════════════════════════════════════
// PARSED GGUF FILE DATA
// ═══════════════════════════════════════════════════════════════════════════

struct GGUFFile {
    std::unordered_map<std::string, std::string> metadata_str;
    std::unordered_map<std::string, int64_t>     metadata_int;
    std::unordered_map<std::string, float>       metadata_float;
    std::vector<GGUFTensorInfo> tensors;
    std::vector<uint8_t> file_data;  // full file in memory
    bool valid{false};

    // Tokenizer vocab (extracted from metadata arrays)
    std::vector<std::string> tokenizer_tokens;
    std::vector<float> tokenizer_scores;
    std::vector<int32_t> tokenizer_types;
    std::string tokenizer_model;  // "llama" or "gpt2"
    int32_t bos_token_id{-1}, eos_token_id{-1};
};

// ═══════════════════════════════════════════════════════════════════════════
// FP16 -> FP32 CONVERSION (software, no F16C required)
// ═══════════════════════════════════════════════════════════════════════════

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h & 0x7C00) >> 10;
    uint32_t mant = h & 0x03FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t v = sign;
            float f;
            std::memcpy(&f, &v, sizeof(f));
            return f; // +/- zero
        }
        // Subnormal: normalize
        int shift = 10;
        while ((mant & 0x0400) == 0) { mant <<= 1; shift--; }
        exp = 1 + (127 - 15) - (uint32_t)shift;
        mant = (mant & 0x03FF) << 13;
        uint32_t v = sign | (exp << 23) | mant;
        float f;
        std::memcpy(&f, &v, sizeof(f));
        return f;
    }
    if (exp == 31) {
        // Inf or NaN
        uint32_t v = sign | 0x7F800000 | (mant << 13);
        float f;
        std::memcpy(&f, &v, sizeof(f));
        return f;
    }
    exp = exp + (127 - 15);
    mant = mant << 13;
    uint32_t v = sign | (exp << 23) | mant;
    float f;
    std::memcpy(&f, &v, sizeof(f));
    return f;
}

// ═══════════════════════════════════════════════════════════════════════════
// Q2_0 BLOCK DECODER
// ═══════════════════════════════════════════════════════════════════════════
// Q2_0 block (g128): [2 bytes FP16 scale] [32 bytes packed codes]
// Codes: 0→-1, 1→0, 2→+1  (same as I2_S)
// I2_S block: [32 bytes codes] [4 bytes FP32 scale]
// Conversion: read FP16 → FP32, reorder to I2_S layout

inline void decode_q2_0_block(const uint8_t* block_data,
                               uint8_t* codes_out, float* scale_out) {
    // Read FP16 scale
    uint16_t scale_fp16;
    std::memcpy(&scale_fp16, block_data, 2);
    *scale_out = fp16_to_fp32(scale_fp16);

    // Copy 32 bytes codes (same format as I2_S)
    std::memcpy(codes_out, block_data + 2, 32);
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN API: Load GGUF model into Terllama data structures
// ═══════════════════════════════════════════════════════════════════════════
//
// Parses a .gguf file, extracts model config from metadata, and populates:
//   - cfg: ModelConfig
//   - embedding: word embedding weights
//   - layer_norms: per-layer RMS norm weights
//   - final_norm: final RMS norm weights
//   - layers: quantized linear layers (converted to I2_S blocks)
//
// Returns true on success.

struct NormWeights;  // forward-declared, defined in loader.h

bool load_gguf_model(const std::string& path,
                     ModelConfig& cfg,
                     std::vector<float>& embedding,
                     std::vector<NormWeights>& layer_norms,
                     std::vector<float>& final_norm,
                     std::vector<LayerData>& layers,
                     Tokenizer* tokenizer = nullptr);

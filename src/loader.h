/*
 * loader.h: Binary file I/O for Terllama model files
 *
 * Binary formats:
 *   model_decomposed.bin        ALS block-scaled format (magic 0xDEADBEEF)
 *   model_extra.bin             embedding + RMSNorm weights
 *
 * Also supports GGUF format (via gguf_loader.h):
 *   model.gguf                  GGUF Q2_0 ternary weights (e.g. Bonsai)
 */
#pragma once
#include "model.h"
#include "gguf_loader.h"
#include "core/tokenizer.h"
#include "core/logger.h"
#include <sys/stat.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <cctype>
#include <algorithm>
#include <cmath>

// ═══════════════════════════════════════════════════════════════════════════
// TERNARY DECODER (packed → bitplane, old format)
// ═══════════════════════════════════════════════════════════════════════════
inline int8_t decode_ternary(const uint8_t* data, size_t pos) {
    // Packed format: per-element, 2 bits = {non_zero (MSB), is_neg (LSB)}
    //   00 = 0, 10 (binary 2) = +1, 11 (binary 3) = -1
    size_t byte_idx = (pos * 2) / 8;
    int bit_offset = (pos * 2) % 8;
    unsigned int bits = (data[byte_idx] >> (6 - bit_offset)) & 0x3;
    if (bits == 0) return 0;
    if (bits == 2) return 1;
    if (bits == 3) return -1;
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// BLOCK-SCALED TERNARY DECODER (4 ternary values per byte)
// ═══════════════════════════════════════════════════════════════════════════
// Format: 4 ternary values per byte, codes {0=-1, 1=0, 2=+1}
// Per byte (MSB to LSB): [elem0(2bit), elem1(2bit), elem2(2bit), elem3(2bit)]
inline void decode_block_ternary(const uint8_t* packed, int8_t* ternary, int qk) {
    for (int i = 0; i < qk / 4; i++) {
        uint8_t byte = packed[i];
        ternary[i*4 + 0] = ((byte >> 6) & 0x03) - 1;  // 0→-1, 1→0, 2→+1
        ternary[i*4 + 1] = ((byte >> 4) & 0x03) - 1;
        ternary[i*4 + 2] = ((byte >> 2) & 0x03) - 1;
        ternary[i*4 + 3] = (byte & 0x03) - 1;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG LOADER
// ═══════════════════════════════════════════════════════════════════════════
inline ModelConfig load_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open model config: " + path); }
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

    // ─── Config validation ──────────────────────────────────────────────
    // Reject truncated/crafted config headers. In particular mpe=0 would
    // produce a zero-size KV cache → buffer overflow on the first token.
    // Every dimension must be positive, head_dim (= hidden_size /
    // num_attention_heads) must be >= 1, and mpe must be within a sane
    // range (it is capped further to 4096 below for KV memory).
    // Failure mode: throw, never exit(1) — init_server() and the server's
    // auto-reload path (handlers.cpp) catch std::exception and return a
    // clean error, keeping a running server alive on a bad/truncated model
    // dir. cmd_show also already wraps load_config in try/catch. A sentinel
    // config was rejected: callers that don't check it would proceed with
    // silently-zero dimensions.
    if (vs <= 0 || hs <= 0 || is <= 0 || nl <= 0 ||
        nah <= 0 || nkv <= 0 || hs / nah <= 0 ||
        mpe <= 0 || mpe > 1000000) {
        Logger::error("Invalid model config in {}: vocab={} hidden={} intermediate={} layers={} heads={} kv_heads={} mpe={}",
                      path, vs, hs, is, nl, nah, nkv, mpe);
        throw std::runtime_error("Invalid model config in " + path + " (truncated or corrupt model header)");
    }

    // KV cache memory guard: max_position_embeddings is the rope/KV length,
    // so huge config values (e.g. Qwen3 40960) would pre-allocate ~9 GB of
    // cache on this machine. Cap the *effective* context; generation beyond
    // this is not supported for such models (short-prompt testing is fine).
    if (mpe > 4096) {
        Logger::warn("Capping max_position_embeddings {} -> 4096 (KV cache memory guard)", mpe);
        mpe = 4096;
    }
    cfg = {vs, hs, is, nl, nah, nkv, rne, rt, mpe, hs / nah};
    return cfg;
}

// ═══════════════════════════════════════════════════════════════════════════
// EMBEDDING + NORM LOADERS
// ═══════════════════════════════════════════════════════════════════════════
inline std::vector<float> load_embedding(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open model file: " + path); }
    f.seekg(4*9, std::ios::beg); // skip 9 int32/float config fields
    std::vector<float> emb(cfg.vocab_size * cfg.hidden_size);
    f.read(reinterpret_cast<char*>(emb.data()), emb.size() * sizeof(float));
    return emb;
}

inline std::vector<float> load_final_norm(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open model file: " + path); }
    f.seekg(36 + cfg.vocab_size * cfg.hidden_size * 4, std::ios::beg);
    std::vector<float> fn(cfg.hidden_size);
    f.read(reinterpret_cast<char*>(fn.data()), fn.size() * sizeof(float));
    return fn;
}

struct NormWeights {
    std::vector<float> input_layernorm;
    std::vector<float> post_attention_layernorm;
    std::vector<float> attn_sub_norm;  // BitNet: scale attention output before residual
    std::vector<float> ffn_sub_norm;   // BitNet: scale FFN output before residual
    std::vector<float> q_norm;         // Qwen3: per-head Q RMSNorm weight [head_dim]
    std::vector<float> k_norm;         // Qwen3: per-head K RMSNorm weight [head_dim]
};

inline std::vector<NormWeights> load_layer_norms(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open model file: " + path); }
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

// ═══════════════════════════════════════════════════════════════════════════
// LAYER INDEX LOOKUP
// ═══════════════════════════════════════════════════════════════════════════
inline int find_layer_index(const std::vector<LayerData>& layers, const std::string& name) {
    for (int i = 0; i < (int)layers.size(); i++)
        if (layers[i].name == name) return i;
    // Missing layer in a truncated/crafted model: propagate instead of
    // exit(1) so a bad file cannot kill a running server (DoS). Throw —
    // callers in inference.cpp/mote_builder.cpp index the result
    // unconditionally, so returning -1 would be unchecked UB; throwing is
    // caught per-request by httplib's exception handler (→ 500, server
    // survives) or at load time by init_server (→ clean load failure).
    Logger::error("Layer not found: {}", name);
    throw std::runtime_error("Layer not found in model file: " + name);
}

// ═══════════════════════════════════════════════════════════════════════════
// DECOMPOSED LAYER LOADER (model_decomposed.bin)
// ═══════════════════════════════════════════════════════════════════════════
inline std::vector<LayerData> load_decomposed_layers(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open model file: " + path); }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0xDEADBEEF) {
        Logger::error("Bad magic");
        throw std::runtime_error("Bad magic in decomposed model file");
    }
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
        if (ld.num_terms == 0) {
            // Raw FP32 layer (e.g. lm_head): data_len(uint32) + float32 data
            ld.has_raw_weights = true;
            uint32_t data_len;
            f.read(reinterpret_cast<char*>(&data_len), 4);
            ld.raw_weights.resize(data_len / sizeof(float));
            f.read(reinterpret_cast<char*>(ld.raw_weights.data()), data_len);
            continue;
        }
        ld.terms.resize(ld.num_terms);
        for (int t = 0; t < ld.num_terms; t++) {
            auto& term = ld.terms[t];
            size_t n_elements = (size_t)ld.out_features * ld.in_features;
            size_t n_bytes = (n_elements * 2 + 7) / 8;
            f.read(reinterpret_cast<char*>(&term.alpha_exp), 4);
            std::vector<uint8_t> packed(n_bytes);
            f.read(reinterpret_cast<char*>(packed.data()), n_bytes);
            // Coalesced bitplane format: combined[word] = (nz << 16) | neg
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
                    if (tv == 1) term.combined[abs_word] |= (1 << (bit + 16));       // nz only
                    else if (tv == -1) term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit); // nz + neg
                }
            }
        }
    }
    return layers;
}

// ═══════════════════════════════════════════════════════════════════════════
// ALS LAYER LOADER (model_decomposed.bin)
// ═══════════════════════════════════════════════════════════════════════════
// ALS is the only quantized format. One magic is accepted:
//   magic 0xDEADBEEF — layer layout is auto-sniffed per layer:
//   * old ALS export: u32 num_terms after in_f (0 = raw FP32, else
//     num_terms × [alpha(int32) + packed 2-bit ternary codes]); loaded
//     as classic bitplane terms (matches load_decomposed_layers).
//   * new ALS export: u8 layer_type + u32 data_len after in_f.
//
// New-format layer_type: 1=RAW_FP32, 2=multi-term block container, 0=single-term.
//
// Parse a packed block-term buffer into a BlockTerm vector.
// Layout (per row): [block0_codes][block0_scale][block1_codes][block1_scale]...
static std::vector<BlockTerm> parse_block_terms(const uint8_t* data, size_t data_len,
                                                int out_f, int in_f, int qk) {
    int n_blocks = (in_f + qk - 1) / qk;
    int codes_per_block = qk / 4;
    std::vector<BlockTerm> blocks((size_t)out_f * n_blocks);
    int row_stride = n_blocks * (codes_per_block + (int)sizeof(float));
    for (int row = 0; row < out_f; row++) {
        for (int b = 0; b < n_blocks; b++) {
            int block_idx = row * n_blocks + b;
            int offset = row * row_stride + b * (codes_per_block + (int)sizeof(float));
            if (offset + codes_per_block + (int)sizeof(float) > (int)data_len) {
                Logger::error("Block data truncated at row {} block {}", row, b);
                throw std::runtime_error("ALS block data truncated in model file");
            }
            blocks[block_idx].packed.assign(
                data + offset, data + offset + codes_per_block);
            float scale;
            std::memcpy(&scale, data + offset + codes_per_block, sizeof(float));
            blocks[block_idx].scale = scale;
        }
    }
    return blocks;
}

// Decode a set of block terms into a single combined[] bitplane term.
// Legacy fallback; the real multi-term path is ternary_linear_blocks.
// Only term 0 is used for multi-term layers.
static BitplaneTerm decode_block_terms_to_combined(const std::vector<BlockTerm>& blocks,
                                                   int out_f, int in_f, int qk) {
    int n_blocks = (in_f + qk - 1) / qk;
    int words_per_row = (in_f + 15) / 16;
    size_t n_words = (size_t)out_f * words_per_row;

    BitplaneTerm term;
    term.alpha_exp = 0;  // scale handled per-block
    term.n_elements = (size_t)out_f * in_f;
    term.combined.assign(n_words, 0);

    std::vector<int8_t> decoded(qk);
    for (int row = 0; row < out_f; row++) {
        for (int b = 0; b < n_blocks; b++) {
            int block_idx = row * n_blocks + b;
            int block_start = b * qk;
            int block_end = std::min(block_start + qk, in_f);
            int block_size = block_end - block_start;

            decode_block_ternary(blocks[block_idx].packed.data(), decoded.data(), qk);

            for (int j = 0; j < block_size; j++) {
                int8_t tv = decoded[j];
                int word = (block_start + j) / 16;
                int bit = (block_start + j) % 16;
                size_t abs_word = (size_t)row * words_per_row + word;
                if (tv == 1)       term.combined[abs_word] |= (1 << (bit + 16));
                else if (tv == -1) term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit);
            }
        }
    }
    return term;
}

// Parse an ALS layer in the new layer_type layout.
//   data layout: u8 layer_type + u32 data_len already consumed by caller.
//   layer_type: 1=RAW_FP32, 2=multi-term container, 0=single-term.
//   build_combined: decode term 0 to combined[] bitplane (legacy fallback;
//   only ever passed false today).
static void parse_layer_type_data(LayerData& ld, uint8_t layer_type,
                                  const std::vector<uint8_t>& data,
                                  bool build_combined) {
    int qk = 128;
    ld.has_blocks = true;
    ld.block_qk = qk;

    if (layer_type == 1) {
        // RAW_FP32: store flat float weights
        ld.has_raw_weights = true;
        ld.has_blocks = false;
        ld.raw_weights.resize(data.size() / sizeof(float));
        std::memcpy(ld.raw_weights.data(), data.data(), data.size());
        return;
    }

    if (layer_type == 2) {
        // ─── Multi-term block container ─────────────────────────────────
        // data = [num_terms:u32][term0_len:u32][term0_data]...[termN_len:u32][termN_data]
        uint32_t num_terms;
        if (data.size() < 4) {
            Logger::error("ALS multi-term: bad header");
            throw std::runtime_error("ALS multi-term: bad header in model file");
        }
        std::memcpy(&num_terms, data.data(), 4);
        if (num_terms == 0 || num_terms > 32) {
            Logger::error("ALS multi-term: invalid num_terms={}", num_terms);
            throw std::runtime_error("ALS multi-term: invalid num_terms in model file");
        }
        ld.block_terms.resize(num_terms);
        size_t pos = 4;
        for (uint32_t t = 0; t < num_terms; t++) {
            if (pos + 4 > data.size()) {
                Logger::error("ALS multi-term: truncated term len");
                throw std::runtime_error("ALS multi-term: truncated term length in model file");
            }
            uint32_t tlen;
            std::memcpy(&tlen, data.data() + pos, 4);
            pos += 4;
            if (pos + tlen > data.size()) {
                Logger::error("ALS multi-term: truncated term data");
                throw std::runtime_error("ALS multi-term: truncated term data in model file");
            }
            ld.block_terms[t] = parse_block_terms(data.data() + pos, tlen,
                                                  ld.out_features, ld.in_features, qk);
            pos += tlen;
        }
        // Backward-compat combined[] from term 0 (degraded fallback only)
        if (build_combined) {
            ld.num_terms = 1;
            ld.terms.push_back(decode_block_terms_to_combined(ld.block_terms[0],
                                                              ld.out_features, ld.in_features, qk));
        }
        return;
    }

    // ─── Single-term (layer_type == 0) ────────────────────────────────
    ld.block_terms.resize(1);
    ld.block_terms[0] = parse_block_terms(data.data(), data.size(),
                                          ld.out_features, ld.in_features, qk);

    if (build_combined) {
        // Decode to combined[] for backward-compatible kernels
        ld.num_terms = 1;
        ld.terms.push_back(decode_block_terms_to_combined(ld.block_terms[0],
                                                          ld.out_features, ld.in_features, qk));
    }
}

// Sniff the ALS layer layout after [out_f][in_f]: new exports store a u8
// layer_type (≤2) followed by u32 data_len (bytes shift in → large u32);
// old exports store u32 num_terms (≤32, 0 = raw FP32).
// Returns 0 = layer_type layout, 1 = old num_terms layout.
static int sniff_als_layout(const std::ifstream& f, uint32_t x) {
    return x <= 32 ? 1 : 0;
}

inline std::vector<LayerData> load_decomposed_layers_als(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open model file: " + path); }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0xDEADBEEF) {
        Logger::error("Bad magic: 0x{:x}", magic);
        throw std::runtime_error("Bad magic in ALS model file");
    }
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

        // ALS (0xDEADBEEF): sniff layout from the u32 following in_f.
        uint32_t x;
        f.read(reinterpret_cast<char*>(&x), 4);
        if (sniff_als_layout(f, x) == 1) {
            // ─── Old ALS layout: u32 num_terms (0 = raw) ────────────────
            if (x == 0) {
                // Raw FP32 layer: data_len(u32) + float32 data
                uint32_t data_len;
                f.read(reinterpret_cast<char*>(&data_len), 4);
                ld.has_raw_weights = true;
                ld.raw_weights.resize(data_len / sizeof(float));
                f.read(reinterpret_cast<char*>(ld.raw_weights.data()), data_len);
                continue;
            }
            ld.terms.resize(x);
            for (uint32_t t = 0; t < x; t++) {
                auto& term = ld.terms[t];
                size_t n_elements = (size_t)ld.out_features * ld.in_features;
                size_t n_bytes = (n_elements * 2 + 7) / 8;
                f.read(reinterpret_cast<char*>(&term.alpha_exp), 4);
                std::vector<uint8_t> packed(n_bytes);
                f.read(reinterpret_cast<char*>(packed.data()), n_bytes);
                // Coalesced bitplane format: combined[word] = (nz << 16) | neg
                term.n_elements = n_elements;
                int words_per_row = (ld.in_features + 15) / 16;
                size_t n_words = (size_t)ld.out_features * words_per_row;
                term.combined.assign(n_words, 0);
                for (int r = 0; r < ld.out_features; r++) {
                    for (int j = 0; j < ld.in_features; j++) {
                        size_t pos = (size_t)r * ld.in_features + j;
                        int8_t tv = decode_ternary(packed.data(), pos);
                        int word = j / 16;
                        int bit = j % 16;
                        size_t abs_word = (size_t)r * words_per_row + word;
                        if (tv == 1) term.combined[abs_word] |= (1 << (bit + 16));
                        else if (tv == -1) term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit);
                    }
                }
            }
            continue;
        }

        // ─── New ALS layout: u8 layer_type + u32 data_len ───────────────
        // The export interleaves [layer_type:u8][data_len:u32], so the u32 `x`
        // read above already consumed layer_type + data_len's low 3 bytes.
        // Reassemble the length: low 24 bits from x>>8, high byte from stream.
        uint8_t layer_type = (uint8_t)(x & 0xFF);
        if (layer_type > 2) {
            Logger::error("ALS layer {}: bad layer_type {}", ld.name, (int)layer_type);
            throw std::runtime_error("ALS layer: bad layer_type in model file");
        }
        uint32_t data_len = (x >> 8) & 0x00FFFFFF;
        uint8_t len_hi;
        f.read(reinterpret_cast<char*>(&len_hi), 1);
        data_len |= (uint32_t(len_hi) << 24);
        std::vector<uint8_t> data(data_len);
        f.read(reinterpret_cast<char*>(data.data()), data_len);
        parse_layer_type_data(ld, layer_type, data, /*build_combined=*/false);
    }
    return layers;
}

// ═══════════════════════════════════════════════════════════════════════════
// TQ1.0 LAYER LOADER (model_tq1.bin)
// ═══════════════════════════════════════════════════════════════════════════
// Format: magic(TQ1_=0x5F315154), num_layers(uint32)
//   per layer: name_len(uint32), name, out_f(uint32), in_f(uint32),
//              layer_type(uint8), data_len(uint32), packed_data
//   layer_type: 0=TQ1, 1=RAW_FP32
//   TQ1 packed_data = Base-3 codes(24 bytes per 120-element block) + scales(float per block)
//
// TQ1.0 packs 5 ternary trits per byte (vs block-scaled ternary's 4), giving
// ~27% smaller files and ~1.58 bits/weight.  At load time we convert TQ1→
// block-scaled ternary so all existing block kernels work transparently.
inline std::vector<LayerData> load_decomposed_layers_tq1(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open model file: " + path); }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x5F315154) {  // "TQ1_"
        Logger::error("Bad magic (expected TQ1_): 0x{:x}", magic);
        throw std::runtime_error("Bad magic in TQ1 model file");
    }
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
        uint8_t layer_type;
        f.read(reinterpret_cast<char*>(&layer_type), 1);
        uint32_t data_len;
        f.read(reinterpret_cast<char*>(&data_len), 4);

        std::vector<uint8_t> data(data_len);
        f.read(reinterpret_cast<char*>(data.data()), data_len);

        if (layer_type == 1) {
            ld.has_raw_weights = true;
            ld.raw_weights.resize(data_len / sizeof(float));
            std::memcpy(ld.raw_weights.data(), data.data(), data_len);
            continue;
        }

        // TQ1.0 → convert to block-scaled ternary for kernel compatibility
        int tq1_qk = 120;
        int blk_qk = 128;
        int n_tq1_blocks = (ld.in_features + tq1_qk - 1) / tq1_qk;
        int n_blk_blocks = (ld.in_features + blk_qk - 1) / blk_qk;

        // Decode TQ1 → int8 ternary, then repack as 2-bit codes
        std::vector<int8_t> all_ternary((size_t)ld.out_features * ld.in_features);

        // Data layout (per row): [block0_codes(24B)][block0_scale(4B)]...
        int tq1_codes_per_block = tq1_qk / 5;  // 24
        int row_stride = n_tq1_blocks * (tq1_codes_per_block + (int)sizeof(float));

        for (int row = 0; row < ld.out_features; row++) {
            for (int b = 0; b < n_tq1_blocks; b++) {
                int block_start = b * tq1_qk;
                int block_end = std::min(block_start + tq1_qk, ld.in_features);
                int block_size = block_end - block_start;
                int offset = row * row_stride + b * (tq1_codes_per_block + (int)sizeof(float));

                // Decode TQ1 codes to int8 ternary
                tq1_decode_block(data.data() + offset,
                                 &all_ternary[(size_t)row * ld.in_features + block_start],
                                 block_size);
            }
        }

        // Store as block-scaled ternary (128 elements per block), single term
        ld.has_blocks = true;
        ld.block_qk = blk_qk;
        ld.block_terms.resize(1);
        ld.block_terms[0].resize((size_t)ld.out_features * n_blk_blocks);

        for (int row = 0; row < ld.out_features; row++) {
            for (int b = 0; b < n_blk_blocks; b++) {
                int block_idx = row * n_blk_blocks + b;
                int block_start = b * blk_qk;
                int block_end = std::min(block_start + blk_qk, ld.in_features);
                int block_size = block_end - block_start;

                // Encode int8 ternary → 2-bit codes
                ld.block_terms[0][block_idx].packed.resize(blk_qk / 4, 0);
                for (int j = 0; j < block_size; j++) {
                    int8_t tv = all_ternary[(size_t)row * ld.in_features + block_start + j];
                    uint8_t code = (uint8_t)(tv + 1);  // -1→0, 0→1, +1→2
                    int byte_idx = j / 4;
                    int shift = 6 - 2 * (j % 4);
                    ld.block_terms[0][block_idx].packed[byte_idx] |= (code << shift);
                }
                // Compute block scale: max absolute value
                float max_abs = 0.0f;
                for (int j = 0; j < block_size; j++) {
                    float v = std::abs((float)all_ternary[(size_t)row * ld.in_features + block_start + j]);
                    if (v > max_abs) max_abs = v;
                }
                ld.block_terms[0][block_idx].scale = max_abs > 0.0f ? max_abs : 1.0f;
            }
        }

        // Also decode to bitplane combined[] for backward compatibility
        int words_per_row = (ld.in_features + 15) / 16;
        size_t n_words = (size_t)ld.out_features * words_per_row;
        BitplaneTerm term;
        term.alpha_exp = 0;
        term.n_elements = (size_t)ld.out_features * ld.in_features;
        term.combined.assign(n_words, 0);

        for (int row = 0; row < ld.out_features; row++) {
            for (int j = 0; j < ld.in_features; j++) {
                int8_t tv = all_ternary[(size_t)row * ld.in_features + j];
                int word = j / 16;
                int bit = j % 16;
                size_t abs_word = (size_t)row * words_per_row + word;
                if (tv == 1)      term.combined[abs_word] |= (1 << (bit + 16));
                else if (tv == -1) term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit);
            }
        }
        ld.num_terms = 1;
        ld.terms.push_back(std::move(term));
    }
    return layers;
}

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED LOADER: auto-detect GGUF vs .bin format
// ═══════════════════════════════════════════════════════════════════════════
// If model_dir/given path ends with .gguf, load directly via GGUF parser.
// Otherwise, load from model_extra.bin + model_decomposed.bin.

inline bool has_gguf_ext(const std::string& path) {
    return path.size() >= 5 && path.substr(path.size() - 5) == ".gguf";
}

inline std::string model_path_for(const std::string& model_dir) {
    // Check if model_dir is a .gguf file directly
    struct stat st;
    if (stat(model_dir.c_str(), &st) == 0 && S_ISREG(st.st_mode) && has_gguf_ext(model_dir)) {
        return model_dir;  // direct .gguf path
    }
    // Check if model_dir contains a .gguf file
    std::string gguf_path = model_dir + "/" + "model.gguf";
    if (stat(gguf_path.c_str(), &st) == 0) return gguf_path;
    gguf_path = model_dir + "/model.q2_0.gguf";
    if (stat(gguf_path.c_str(), &st) == 0) return gguf_path;
    // Scan for any .gguf file in the directory
    DIR* dir = opendir(model_dir.c_str());
    if (dir) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string name = entry->d_name;
            if (name.size() > 5 && name.substr(name.size() - 5) == ".gguf") {
                gguf_path = model_dir + "/" + name;
                closedir(dir);
                return gguf_path;
            }
        }
        closedir(dir);
    }
    return "";  // no GGUF file found
}

struct LoadedModel {
    ModelConfig cfg;
    std::vector<float> embedding;
    std::vector<float> final_norm;
    std::vector<NormWeights> layer_norms;
    std::vector<LayerData> layers;
    Tokenizer tokenizer;  // populated from GGUF metadata
};

inline LoadedModel load_model_from(const std::string& model_path_or_dir) {
    LoadedModel m;

    // Check for .gguf (direct path or inside directory)
    std::string gguf_path;
    struct stat st;
    if (stat(model_path_or_dir.c_str(), &st) == 0 && S_ISREG(st.st_mode) && has_gguf_ext(model_path_or_dir)) {
        gguf_path = model_path_or_dir;
    } else {
        gguf_path = model_path_for(model_path_or_dir);
    }

    if (!gguf_path.empty()) {
        // ── GGUF path ────────────────────────────────────────────────────
        Logger::info("Loading GGUF model: {}", gguf_path);
        if (!load_gguf_model(gguf_path, m.cfg, m.embedding,
                              m.layer_norms, m.final_norm, m.layers,
                              &m.tokenizer)) {
            Logger::error("GGUF load failed");
            throw std::runtime_error("GGUF load failed: " + gguf_path);
        }
    } else {
        // ── Legacy .bin path ─────────────────────────────────────────────
        std::string extra_path = model_path_or_dir + "/model_extra.bin";
        std::string als_path   = model_path_or_dir + "/model_decomposed.bin";

        struct stat st_extra;
        if (stat(extra_path.c_str(), &st_extra) != 0) {
            Logger::error("No model files found in {}", model_path_or_dir);
            throw std::runtime_error("No model files found in " + model_path_or_dir);
        }

        m.cfg = load_config(extra_path);
        m.embedding = load_embedding(extra_path, m.cfg);
        m.final_norm = load_final_norm(extra_path, m.cfg);
        m.layer_norms = load_layer_norms(extra_path, m.cfg);

        if (stat(als_path.c_str(), &st_extra) != 0) {
            Logger::error("No ALS weights (model_decomposed.bin) in {}", model_path_or_dir);
            throw std::runtime_error("No ALS weights (model_decomposed.bin) in " + model_path_or_dir);
        }
        m.layers = load_decomposed_layers_als(als_path);  // ALS (old/new layout)

        // ── Qwen3-style per-head Q/K RMSNorm + head_dim fixup ───────────
        // Qwen3 uses head_dim=128 with Q/K RMSNorm (q_proj is H*128 wide,
        // i.e. wider than hidden_size). Derive the true head_dim from the
        // q_proj layer of layer 0, and lift q_norm/k_norm RAW pseudo-layers
        // out of the flat layer list into NormWeights so transformer_block
        // can apply them. Both are no-ops for standard (SmolLM2-style) ALS.
        for (size_t i = 0; i < m.layers.size(); i++) {
            if (m.layers[i].name == "model.layers.0.self_attn.q_proj" &&
                m.cfg.num_attention_heads > 0) {
                int qd = m.layers[i].out_features / m.cfg.num_attention_heads;
                if (qd != m.cfg.head_dim) {
                    Logger::warn("head_dim derived {} -> {} (from q_proj {}/{})",
                                 m.cfg.head_dim, qd, m.layers[i].out_features,
                                 m.cfg.num_attention_heads);
                    m.cfg.head_dim = qd;
                }
                break;
            }
        }
        // Extract q_norm/k_norm pseudo-layers (named model.layers.<N>.self_attn.q_norm)
        // into layer_norms[N]. They are RAW_FP32 weights of size head_dim.
        // NOTE: must match the FULL name — a bare sscanf("...%d.self_attn.q_norm")
        // also "matches" q_proj/k_proj (only %d assigned, return==1), which would
        // wrongly erase real projection layers. Require the suffix as well.
        auto norm_layer_index = [](const std::string& nm, const char* suffix) -> int {
            const std::string s(suffix);
            if (nm.size() <= s.size()) return -1;
            if (nm.compare(nm.size() - s.size(), s.size(), s) != 0) return -1;
            const std::string pre = nm.substr(0, nm.size() - s.size());
            const std::string pfx = "model.layers.";
            if (pre.compare(0, pfx.size(), pfx) != 0) return -1;
            std::string num = pre.substr(pfx.size());
            if (num.empty() || !std::all_of(num.begin(), num.end(), ::isdigit)) return -1;
            return std::stoi(num);
        };
        if ((int)m.layer_norms.size() == m.cfg.num_hidden_layers) {
            for (auto it = m.layers.begin(); it != m.layers.end(); ) {
                const char* suffix = nullptr;
                int li = -1;
                int lq = norm_layer_index(it->name, ".self_attn.q_norm");
                int lk = norm_layer_index(it->name, ".self_attn.k_norm");
                if (lq >= 0) { suffix = ".self_attn.q_norm"; li = lq; }
                else if (lk >= 0) { suffix = ".self_attn.k_norm"; li = lk; }
                if (suffix) {
                    if (li < m.cfg.num_hidden_layers && it->has_raw_weights) {
                        if (lq >= 0) m.layer_norms[li].q_norm = it->raw_weights;
                        else         m.layer_norms[li].k_norm = it->raw_weights;
                    }
                    it = m.layers.erase(it);
                } else {
                    ++it;
                }
            }
        }

        // ── Tokenizer for legacy .bin models ─────────────────────────────
        // Native vocab comes from a HuggingFace tokenizer.json co-located in
        // the model dir (written by the export script since v1.0.5). If it's
        // absent, GigaToken may still pick it up at runtime; otherwise the
        // tokenizer is unavailable and the engine fails cleanly (no Python).
        std::string tok_json = model_path_or_dir + "/tokenizer.json";
        if (stat(tok_json.c_str(), &st) == 0) {
            if (m.tokenizer.load_from_tokenizer_json(tok_json)) {
                Logger::info("  Tokenizer: {} (from tokenizer.json), vocab={}",
                    m.tokenizer.model_type, m.tokenizer.vocab.size());
            } else {
                Logger::info("  Tokenizer: tokenizer.json present but unreadable — tokenizer unavailable");
            }
        } else {
            Logger::info("  Tokenizer: no tokenizer.json in model dir — tokenizer unavailable");
        }
    }

    return m;
}

// ═══════════════════════════════════════════════════════════════════════════
// MoTE FORMAT SAVE/LOAD
// ═══════════════════════════════════════════════════════════════════════════
// Magic: "MOTE" = 0x45544F4D
// Format:
//   Magic: uint32 = 0x45544F4D
//   config_len: uint32
//   config_json: char[config_len]
//   num_layers: uint32
//   Per layer:
//     name_len, name
//     num_experts: uint32
//     top_k: uint32
//     has_shared: uint8
//     shared_gate_len + shared_gate_data
//     shared_up_len + shared_up_data
//     shared_down_len + shared_down_data
//     For each expert:
//       expert_gate_len + expert_gate_data
//       expert_up_len + expert_up_data
//       expert_down_len + expert_down_data
//     router_weight_len: uint32
//     router_weights: float32[router_weight_len]

constexpr uint32_t MOTE_MAGIC = 0x45544F4D;  // "MOTE"

// ─── Serialize a LayerData to bytes ────────────────────────────────────────
inline std::vector<uint8_t> serialize_layer_data(const LayerData& ld) {
    std::vector<uint8_t> buf;
    auto append = [&](const void* data, size_t sz) {
        const uint8_t* ptr = (const uint8_t*)data;
        buf.insert(buf.end(), ptr, ptr + sz);
    };

    uint32_t nl = (uint32_t)ld.name.size();
    append(&nl, 4);
    append(ld.name.data(), nl);
    append(&ld.out_features, 4);
    append(&ld.in_features, 4);
    uint32_t nt = (uint32_t)ld.num_terms;
    append(&nt, 4);

    uint8_t has_raw = ld.has_raw_weights ? 1 : 0;
    uint8_t has_blocks = ld.has_blocks ? 1 : 0;
    append(&has_raw, 1);

    if (ld.has_raw_weights) {
        uint32_t dw = (uint32_t)(ld.raw_weights.size() * sizeof(float));
        append(&dw, 4);
        append(ld.raw_weights.data(), dw);
        return buf;
    }

    append(&has_blocks, 1);

    if (ld.has_blocks && !ld.block_terms.empty()) {
        // Serialize the first term set (legacy MoTE files stored a single set).
        const auto& blocks = ld.block_terms[0];
        uint32_t nb = (uint32_t)blocks.size();
        append(&nb, 4);
        append(&ld.block_qk, 4);
        for (auto& blk : blocks) {
            uint32_t ps = (uint32_t)blk.packed.size();
            append(&ps, 4);
            append(blk.packed.data(), ps);
            append(&blk.scale, 4);
        }
        return buf;
    }

    // Bitplane terms
    uint32_t nterms = (uint32_t)ld.terms.size();
    append(&nterms, 4);
    for (auto& term : ld.terms) {
        append(&term.alpha_exp, 4);
        uint64_t ne = (uint64_t)term.n_elements;
        append(&ne, 8);
        uint32_t cs = (uint32_t)(term.combined.size() * sizeof(uint32_t));
        append(&cs, 4);
        append(term.combined.data(), cs);
    }
    return buf;
}

// ─── Deserialize a LayerData from buffer, return bytes consumed ───────────
inline size_t deserialize_layer_data(const uint8_t* buf, size_t offset, LayerData& ld) {
    size_t pos = offset;
    auto read32 = [&]() -> uint32_t {
        uint32_t v; std::memcpy(&v, buf + pos, 4); pos += 4; return v;
    };

    uint32_t nl = read32();
    ld.name.assign((const char*)buf + pos, nl); pos += nl;
    ld.out_features = (int32_t)read32();
    ld.in_features = (int32_t)read32();
    ld.num_terms = (int32_t)read32();

    uint8_t has_raw = buf[pos++];
    if (has_raw) {
        ld.has_raw_weights = true;
        uint32_t dw = read32();
        ld.raw_weights.resize(dw / sizeof(float));
        std::memcpy(ld.raw_weights.data(), buf + pos, dw); pos += dw;
        return pos - offset;
    }

    uint8_t has_blocks = buf[pos++];
    if (has_blocks) {
        ld.has_blocks = true;
        uint32_t nb = read32();
        ld.block_qk = (int32_t)read32();
        ld.block_terms.resize(1);
        ld.block_terms[0].resize(nb);
        for (uint32_t b = 0; b < nb; b++) {
            uint32_t ps = read32();
            ld.block_terms[0][b].packed.resize(ps);
            std::memcpy(ld.block_terms[0][b].packed.data(), buf + pos, ps); pos += ps;
            std::memcpy(&ld.block_terms[0][b].scale, buf + pos, 4); pos += 4;
        }
        return pos - offset;
    }

    // Bitplane terms
    uint32_t nterms = read32();
    ld.terms.resize(nterms);
    for (uint32_t t = 0; t < nterms; t++) {
        std::memcpy(&ld.terms[t].alpha_exp, buf + pos, 4); pos += 4;
        uint64_t ne; std::memcpy(&ne, buf + pos, 8); pos += 8;
        ld.terms[t].n_elements = (size_t)ne;
        uint32_t cs = read32();
        ld.terms[t].combined.resize(cs / sizeof(uint32_t));
        std::memcpy(ld.terms[t].combined.data(), buf + pos, cs); pos += cs;
    }
    return pos - offset;
}

// ─── Save MoTE model ──────────────────────────────────────────────────────
inline void save_mote_model(const std::string& path,
                             const MoTEConfig& config,
                             const std::vector<MoTELayerData>& mote_layers) {
    std::ofstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot write: {}", path); throw std::runtime_error("Cannot write " + path); }

    uint32_t magic = MOTE_MAGIC;
    f.write((const char*)&magic, 4);

    std::string cfg_str = "{\"num_experts\":" + std::to_string(config.num_experts)
                        + ",\"top_k\":" + std::to_string(config.top_k)
                        + ",\"use_shared_expert\":" + (config.use_shared_expert ? "true" : "false")
                        + "}";
    uint32_t cfg_len = (uint32_t)cfg_str.size();
    f.write((const char*)&cfg_len, 4);
    f.write(cfg_str.data(), cfg_len);

    uint32_t nl = (uint32_t)mote_layers.size();
    f.write((const char*)&nl, 4);

    for (auto& ml : mote_layers) {
        uint32_t name_len = (uint32_t)ml.gate_proj.name.size();
        f.write((const char*)&name_len, 4);
        f.write(ml.gate_proj.name.data(), name_len);
        uint32_t ne = (uint32_t)ml.num_experts;
        uint32_t tk = (uint32_t)ml.top_k;
        f.write((const char*)&ne, 4);
        f.write((const char*)&tk, 4);
        uint8_t hs = 1;
        f.write((const char*)&hs, 1);

        auto write_ld = [&](const LayerData& ld) {
            auto ser = serialize_layer_data(ld);
            uint32_t sz = (uint32_t)ser.size();
            f.write((const char*)&sz, 4);
            f.write((const char*)ser.data(), sz);
        };

        write_ld(ml.gate_proj);
        write_ld(ml.up_proj);
        write_ld(ml.down_proj);

        for (int e = 0; e < (int)ne; e++) {
            write_ld(ml.expert_gate[e]);
            write_ld(ml.expert_up[e]);
            write_ld(ml.expert_down[e]);
        }

        uint32_t rwl = (uint32_t)(ml.router_weight.size() * sizeof(float));
        f.write((const char*)&rwl, 4);
        f.write((const char*)ml.router_weight.data(), rwl);
    }

    Logger::info("MoTE model saved: {} ({} layers)", path, mote_layers.size());
}

// ─── Peek MoTE config from file ────────────────────────────────────────────
inline MoTEConfig peek_mote_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open " + path); }

    uint32_t magic;
    f.read((char*)&magic, 4);
    if (magic != MOTE_MAGIC) {
        Logger::error("Bad MoTE magic: 0x{:x}", magic);
        throw std::runtime_error("Bad MoTE magic in model file");
    }

    uint32_t cfg_len;
    f.read((char*)&cfg_len, 4);
    std::string cfg_str(cfg_len, '\0');
    f.read(&cfg_str[0], cfg_len);

    MoTEConfig cfg;
    auto extract_int = [&](const std::string& key, int def) {
        auto p = cfg_str.find("\"" + key + "\":");
        if (p == std::string::npos) return def;
        p = cfg_str.find(':', p) + 1;
        while (p < cfg_str.size() && (cfg_str[p] == ' ' || cfg_str[p] == '\t')) p++;
        int v = 0, sign = 1;
        if (cfg_str[p] == '-') { sign = -1; p++; }
        while (p < cfg_str.size() && cfg_str[p] >= '0' && cfg_str[p] <= '9')
            v = v * 10 + (cfg_str[p++] - '0');
        return sign * v;
    };
    auto extract_bool = [&](const std::string& key, bool def) {
        auto p = cfg_str.find("\"" + key + "\":");
        if (p == std::string::npos) return def;
        p = cfg_str.find(':', p) + 1;
        while (p < cfg_str.size() && (cfg_str[p] == ' ' || cfg_str[p] == '\t')) p++;
        return cfg_str.substr(p, 4) == "true";
    };

    cfg.num_experts = extract_int("num_experts", 4);
    cfg.top_k = extract_int("top_k", 1);
    cfg.use_shared_expert = extract_bool("use_shared_expert", true);

    return cfg;
}

// ─── Load MoTE layers from file ────────────────────────────────────────────
inline std::vector<MoTELayerData> load_mote_layers(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); throw std::runtime_error("Cannot open " + path); }

    f.seekg(0, std::ios::end);
    size_t file_size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(file_size);
    f.read((char*)buf.data(), file_size);

    size_t pos = 0;
    auto rd32 = [&]() -> uint32_t {
        uint32_t v; std::memcpy(&v, buf.data() + pos, 4); pos += 4; return v;
    };

    uint32_t magic = rd32();
    if (magic != MOTE_MAGIC) {
        Logger::error("Bad MoTE magic: 0x{:x}", magic);
        throw std::runtime_error("Bad MoTE magic in model file");
    }

    uint32_t cfg_len = rd32();
    pos += cfg_len;

    uint32_t num_layers = rd32();
    std::vector<MoTELayerData> layers(num_layers);

    auto read_ld = [&](const std::string& lname) -> LayerData {
        uint32_t lsz = rd32();
        LayerData ld;
        deserialize_layer_data(buf.data(), pos, ld);
        ld.name = lname;
        pos += lsz;
        return ld;
    };

    for (uint32_t i = 0; i < num_layers; i++) {
        auto& ml = layers[i];
        uint32_t nl = rd32();
        std::string lname((const char*)buf.data() + pos, nl); pos += nl;
        ml.num_experts = (int)rd32();
        ml.top_k = (int)rd32();
        uint8_t hs = buf[pos++]; (void)hs;

        ml.gate_proj = read_ld(lname + ".gate_proj");
        ml.up_proj   = read_ld(lname + ".up_proj");
        ml.down_proj = read_ld(lname + ".down_proj");

        ml.expert_gate.resize(ml.num_experts);
        ml.expert_up.resize(ml.num_experts);
        ml.expert_down.resize(ml.num_experts);
        for (int e = 0; e < ml.num_experts; e++) {
            std::string es = lname + ".expert." + std::to_string(e);
            ml.expert_gate[e] = read_ld(es + ".gate_proj");
            ml.expert_up[e]   = read_ld(es + ".up_proj");
            ml.expert_down[e] = read_ld(es + ".down_proj");
        }

        uint32_t rwl = rd32();
        ml.router_weight.resize(rwl / sizeof(float));
        std::memcpy(ml.router_weight.data(), buf.data() + pos, rwl); pos += rwl;
        ml.router_scale = 1.0f;
        ml.is_mote = true;
    }

    Logger::info("MoTE model loaded: {} layers", num_layers);
    return layers;
}

// ═══════════════════════════════════════════════════════════════════════════
// MoTE FILE DETECTION
// ═══════════════════════════════════════════════════════════════════════════
inline bool is_mote_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t magic;
    f.read((char*)&magic, 4);
    return magic == MOTE_MAGIC;
}

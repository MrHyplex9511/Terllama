/*
 * loader.h: Binary file I/O for Terllama model files
 *
 * Two formats:
 *   model_decomposed.bin        ALS bitplane format (original)
 *   model_decomposed_i2s.bin    I2_S packed format (BitNet-style)
 *   model_extra.bin             embedding + RMSNorm weights
 */
#pragma once
#include "model.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
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
// I2_S DECODER (BitNet-style: 4 ternary values per byte)
// ═══════════════════════════════════════════════════════════════════════════
// Format: 4 ternary values per byte, codes {0=-1, 1=0, 2=+1}
// Per byte (MSB to LSB): [elem0(2bit), elem1(2bit), elem2(2bit), elem3(2bit)]
inline void decode_i2s_block(const uint8_t* packed, int8_t* ternary, int qk) {
    for (int i = 0; i < qk / 4; i++) {
        uint8_t byte = packed[i];
        ternary[i*4 + 0] = ((byte >> 6) & 0x03) - 1;  // 0→-1, 1→0, 2→+1
        ternary[i*4 + 1] = ((byte >> 4) & 0x03) - 1;
        ternary[i*4 + 2] = ((byte >> 2) & 0x03) - 1;
        ternary[i*4 + 3] = (byte & 0x03) - 1;
    }
}

// Decode I2_S row to combined[] bitplane format
inline void i2s_row_to_combined(const uint8_t* row_packed,
                                 int in_features, int qk, uint32_t* combined) {
    int n_blocks = (in_features + qk - 1) / qk;
    std::vector<int8_t> decoded(qk);
    for (int b = 0; b < n_blocks; b++) {
        int block_start = b * qk;
        int block_end = std::min(block_start + qk, in_features);
        int block_size = block_end - block_start;

        // Decode I2_S to int8 ternary values
        decode_i2s_block(row_packed + b * (qk / 4), decoded.data(), qk);

        // Write to combined[] bitplane format
        for (int j = 0; j < block_size; j++) {
            int8_t tv = decoded[j];
            int word = (block_start + j) / 16;
            int bit = (block_start + j) % 16;
            if (tv == 1)      combined[word] |= (1 << (bit + 16));       // nz only
            else if (tv == -1) combined[word] |= (1 << (bit + 16)) | (1 << bit); // nz + neg
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONFIG LOADER
// ═══════════════════════════════════════════════════════════════════════════
inline ModelConfig load_config(const std::string& path) {
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

// ═══════════════════════════════════════════════════════════════════════════
// EMBEDDING + NORM LOADERS
// ═══════════════════════════════════════════════════════════════════════════
inline std::vector<float> load_embedding(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    f.seekg(4*9, std::ios::beg); // skip 9 int32/float config fields
    std::vector<float> emb(cfg.vocab_size * cfg.hidden_size);
    f.read(reinterpret_cast<char*>(emb.data()), emb.size() * sizeof(float));
    return emb;
}

inline std::vector<float> load_final_norm(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    f.seekg(36 + cfg.vocab_size * cfg.hidden_size * 4, std::ios::beg);
    std::vector<float> fn(cfg.hidden_size);
    f.read(reinterpret_cast<char*>(fn.data()), fn.size() * sizeof(float));
    return fn;
}

struct NormWeights {
    std::vector<float> input_layernorm;
    std::vector<float> post_attention_layernorm;
};

inline std::vector<NormWeights> load_layer_norms(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
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
    std::cerr << "Layer not found: " << name << std::endl;
    exit(1);
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════
// DECOMPOSED LAYER LOADER (model_decomposed.bin)
// ═══════════════════════════════════════════════════════════════════════════
inline std::vector<LayerData> load_decomposed_layers(const std::string& path) {
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
// I2_S LAYER LOADER (model_decomposed_i2s.bin)
// ═══════════════════════════════════════════════════════════════════════════
// Format: magic(I2S_=0x5F533249), num_layers(uint32)
//   per layer: name_len(uint32), name, out_f(uint32), in_f(uint32),
//              data_len(uint32), packed_data
//   packed_data = codes(4val/byte) + scales(per-block float32)
inline std::vector<LayerData> load_decomposed_layers_i2s(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::cerr << "Cannot open: " << path << std::endl; exit(1); }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x5F533249) {  // "I2S_"
        std::cerr << "Bad magic (expected I2S_): 0x" << std::hex << magic << std::endl;
        exit(1);
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
            // RAW_FP32: store flat float weights
            ld.has_raw_weights = true;
            ld.raw_weights.resize(data_len / sizeof(float));
            std::memcpy(ld.raw_weights.data(), data.data(), data_len);
            continue;  // skip I2_S parsing
        }

        // I2_S format: codes(32 bytes per block) + scales(4 bytes per block)
        int qk = 128;
        int n_blocks = (ld.in_features + qk - 1) / qk;
        int codes_per_block = qk / 4;

        // Store I2_S blocks for direct kernel path
        ld.has_i2s = true;
        ld.i2s_qk = qk;
        ld.i2s_blocks.resize((size_t)ld.out_features * n_blocks);

        // Data layout (per row): [block0_codes][block0_scale][block1_codes][block1_scale]...
        int row_stride = n_blocks * (codes_per_block + (int)sizeof(float));
        for (int row = 0; row < ld.out_features; row++) {
            for (int b = 0; b < n_blocks; b++) {
                int block_idx = row * n_blocks + b;
                int offset = row * row_stride + b * (codes_per_block + (int)sizeof(float));
                if (offset + codes_per_block + (int)sizeof(float) > (int)data_len) {
                    std::cerr << "I2_S data truncated at row " << row << " block " << b << std::endl;
                    exit(1);
                }
                ld.i2s_blocks[block_idx].packed.assign(
                    data.data() + offset,
                    data.data() + offset + codes_per_block);
                float scale;
                std::memcpy(&scale, data.data() + offset + codes_per_block, sizeof(float));
                ld.i2s_blocks[block_idx].scale = scale;
            }
        }

        // Decode to combined[] for backward-compatible kernels
        int words_per_row = (ld.in_features + 15) / 16;
        size_t n_words = (size_t)ld.out_features * words_per_row;

        // Create a single bitplane term (I2_S is single-term)
        BitplaneTerm term;
        term.alpha_exp = 0;  // scale handled per-block
        term.n_elements = (size_t)ld.out_features * ld.in_features;
        term.combined.assign(n_words, 0);

        for (int row = 0; row < ld.out_features; row++) {
            for (int b = 0; b < n_blocks; b++) {
                int block_idx = row * n_blocks + b;
                int block_start = b * qk;
                int block_end = std::min(block_start + qk, ld.in_features);
                int block_size = block_end - block_start;

                std::vector<int8_t> decoded(qk);
                decode_i2s_block(ld.i2s_blocks[block_idx].packed.data(), decoded.data(), qk);

                for (int j = 0; j < block_size; j++) {
                    int8_t tv = decoded[j];
                    int word = (block_start + j) / 16;
                    int bit = (block_start + j) % 16;
                    size_t abs_word = (size_t)row * words_per_row + word;
                    if (tv == 1)      term.combined[abs_word] |= (1 << (bit + 16));
                    else if (tv == -1) term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit);
                }
            }
        }
        ld.num_terms = 1;
        ld.terms.push_back(std::move(term));
    }
    return layers;
}

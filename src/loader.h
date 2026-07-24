/*
 * loader.h: Binary file I/O for Terllama model files
 *
 * Two formats:
 *   model_decomposed.bin        ALS bitplane format (original)
 *   model_decomposed_i2s.bin    I2_S packed format (BitNet-style)
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
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }
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
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }
    f.seekg(4*9, std::ios::beg); // skip 9 int32/float config fields
    std::vector<float> emb(cfg.vocab_size * cfg.hidden_size);
    f.read(reinterpret_cast<char*>(emb.data()), emb.size() * sizeof(float));
    return emb;
}

inline std::vector<float> load_final_norm(const std::string& path, const ModelConfig& cfg) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }
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
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }
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
    Logger::error("Layer not found: {}", name);
    exit(1);
    return -1;
}

// ═══════════════════════════════════════════════════════════════════════════
// DECOMPOSED LAYER LOADER (model_decomposed.bin)
// ═══════════════════════════════════════════════════════════════════════════
inline std::vector<LayerData> load_decomposed_layers(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0xDEADBEEF) { Logger::error("Bad magic"); exit(1); }
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
// I2_S LAYER LOADER (model_decomposed_i2s.bin)
// ═══════════════════════════════════════════════════════════════════════════
// Format: magic(I2S_=0x5F533249), num_layers(uint32)
//   per layer: name_len(uint32), name, out_f(uint32), in_f(uint32),
//              data_len(uint32), packed_data
//   packed_data = codes(4val/byte) + scales(per-block float32)
inline std::vector<LayerData> load_decomposed_layers_i2s(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0x5F533249) {  // "I2S_"
        Logger::error("Bad magic (expected I2S_): 0x{:x}", magic);
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
                    Logger::error("I2_S data truncated at row {} block {}", row, b);
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

// ═══════════════════════════════════════════════════════════════════════════
// UNIFIED LOADER: auto-detect GGUF vs .bin format
// ═══════════════════════════════════════════════════════════════════════════
// If model_dir/given path ends with .gguf, load directly via GGUF parser.
// Otherwise, load from model_extra.bin + model_decomposed_i2s.bin.

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
            exit(1);
        }
    } else {
        // ── Legacy .bin path ─────────────────────────────────────────────
        std::string extra_path = model_path_or_dir + "/model_extra.bin";
        std::string i2s_path   = model_path_or_dir + "/model_decomposed_i2s.bin";
        std::string als_path   = model_path_or_dir + "/model_decomposed.bin";

        struct stat st_extra;
        if (stat(extra_path.c_str(), &st_extra) != 0) {
            Logger::error("No model files found in {}", model_path_or_dir);
            exit(1);
        }

        m.cfg = load_config(extra_path);
        m.embedding = load_embedding(extra_path, m.cfg);
        m.final_norm = load_final_norm(extra_path, m.cfg);
        m.layer_norms = load_layer_norms(extra_path, m.cfg);

        std::ifstream test_i2s(i2s_path);
        if (test_i2s.good()) {
            m.layers = load_decomposed_layers_i2s(i2s_path);
        } else {
            m.layers = load_decomposed_layers(als_path);
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
    uint8_t has_i2s = ld.has_i2s ? 1 : 0;
    append(&has_raw, 1);

    if (ld.has_raw_weights) {
        uint32_t dw = (uint32_t)(ld.raw_weights.size() * sizeof(float));
        append(&dw, 4);
        append(ld.raw_weights.data(), dw);
        return buf;
    }

    append(&has_i2s, 1);

    if (ld.has_i2s) {
        uint32_t nb = (uint32_t)ld.i2s_blocks.size();
        append(&nb, 4);
        append(&ld.i2s_qk, 4);
        for (auto& blk : ld.i2s_blocks) {
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

    uint8_t has_i2s = buf[pos++];
    if (has_i2s) {
        ld.has_i2s = true;
        uint32_t nb = read32();
        ld.i2s_qk = (int32_t)read32();
        ld.i2s_blocks.resize(nb);
        for (uint32_t b = 0; b < nb; b++) {
            uint32_t ps = read32();
            ld.i2s_blocks[b].packed.resize(ps);
            std::memcpy(ld.i2s_blocks[b].packed.data(), buf + pos, ps); pos += ps;
            std::memcpy(&ld.i2s_blocks[b].scale, buf + pos, 4); pos += 4;
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
    if (!f) { Logger::error("Cannot write: {}", path); exit(1); }

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
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }

    uint32_t magic;
    f.read((char*)&magic, 4);
    if (magic != MOTE_MAGIC) {
        Logger::error("Bad MoTE magic: 0x{:x}", magic);
        exit(1);
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
    if (!f) { Logger::error("Cannot open: {}", path); exit(1); }

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
        exit(1);
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

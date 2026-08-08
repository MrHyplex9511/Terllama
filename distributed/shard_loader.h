/*
 * shard_loader.h — Shard-aware ALS model loader for distributed inference.
 *
 * Loads only the transformer-block range [start_layer, end_layer) from
 * model_decomposed.bin (plus embedding/final_norm when this shard owns the
 * first/last stage). Reuses the inline helpers in src/loader.h
 * (load_config, load_embedding, load_final_norm, load_layer_norms,
 * find_layer_index, parse_layer_type_data, sniff_als_layout) — those are
 * NOT modified. This file only adds the range-filtered layer reader and the
 * ShardModel bundle.
 *
 * Record framing (model_decomposed.bin, magic 0xDEADBEEF):
 *   u32 magic, u32 num_layers
 *   per record: u32 name_len, char name[name_len],
 *               u32 out_features, u32 in_features, then payload:
 *     - OLD ALS layout (x <= 32):  x == 0 → raw FP32
 *         (u32 data_len, float data[data_len/4]);  else x = num_terms and
 *         per term: i32 alpha_exp + packed 2-bit ternary codes
 *         (n_bytes = (out*in*2+7)/8).
 *     - NEW ALS layout (x > 32):   layer_type = x & 0xFF (0=single-term,
 *         1=RAW_FP32, 2=multi-term block), data_len = ((x>>8)&0xFFFFFF) |
 *         (len_hi<<24) where len_hi is one more byte on the stream, then
 *         data_len payload bytes.
 *
 * Skipped records are seeked past (never parsed into LayerData), so a shard
 * only materializes its own range in RAM.
 */
#pragma once

#include <httplib.h>
#include <json.hpp>
#include "protocol.h"
#include "loader.h"
#include "inference.h"
#include "core/logger.h"

#include <sys/stat.h>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cctype>

namespace tldist {

// ─── Transformer index extraction from a layer name ────────────────────────
// Returns the layer number N for names of the form "model.layers.<N>.<suffix>",
// or -1 when the name does not match the full pattern (e.g. "lm_head").
inline int parse_layer_index(const std::string& name) {
    const std::string pfx = "model.layers.";
    if (name.compare(0, pfx.size(), pfx) != 0) return -1;
    std::string rest = name.substr(pfx.size());
    size_t dot = rest.find('.');
    if (dot == std::string::npos || dot == 0) return -1;  // need "<N>." suffix
    std::string num = rest.substr(0, dot);
    if (num.empty() || !std::all_of(num.begin(), num.end(), ::isdigit)) return -1;
    return std::stoi(num);
}

// ─── Range-filtered ALS reader ─────────────────────────────────────────────
// Reads model_decomposed.bin sequentially, keeps records whose transformer
// block index is in [start_layer, end_layer), plus "lm_head" when
// include_lm_head. Records are kept in file order. Supports both the old
// (num_terms) and new (layer_type + data_len) ALS layouts for parity with
// load_decomposed_layers_als.
inline std::vector<LayerData> load_decomposed_layers_als_range(
        const std::string& path, int start_layer, int end_layer,
        bool include_lm_head) {
    if (start_layer < 0 || end_layer < start_layer) {
        throw std::runtime_error("Invalid shard layer range: [" +
                                 std::to_string(start_layer) + ", " +
                                 std::to_string(end_layer) + ")");
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        Logger::error("Cannot open: {}", path);
        throw std::runtime_error("Cannot open model file: " + path);
    }
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0xDEADBEEF) {
        Logger::error("Bad magic: 0x{:x}", magic);
        throw std::runtime_error("Bad magic in ALS model file");
    }
    uint32_t num_layers;
    f.read(reinterpret_cast<char*>(&num_layers), 4);

    std::vector<LayerData> kept;
    kept.reserve(end_layer - start_layer + (include_lm_head ? 1 : 0));

    for (uint32_t rec = 0; rec < num_layers; rec++) {
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        std::string name(name_len, '\0');
        f.read(&name[0], name_len);
        uint32_t out_f, in_f;
        f.read(reinterpret_cast<char*>(&out_f), 4);
        f.read(reinterpret_cast<char*>(&in_f), 4);
        uint32_t x;
        f.read(reinterpret_cast<char*>(&x), 4);

        int li = parse_layer_index(name);
        bool keep = (li >= start_layer && li < end_layer) ||
                    (include_lm_head && name == "lm_head");

        if (sniff_als_layout(f, x) == 1) {
            // ─── OLD ALS layout ─────────────────────────────────────────
            if (x == 0) {
                // Raw FP32: u32 data_len + float data
                uint32_t data_len;
                f.read(reinterpret_cast<char*>(&data_len), 4);
                if (keep) {
                    LayerData ld;
                    ld.name = std::move(name);
                    ld.out_features = (int32_t)out_f;
                    ld.in_features = (int32_t)in_f;
                    ld.has_raw_weights = true;
                    ld.raw_weights.resize(data_len / sizeof(float));
                    f.read(reinterpret_cast<char*>(ld.raw_weights.data()), data_len);
                    kept.push_back(std::move(ld));
                } else {
                    f.seekg((std::streamoff)data_len, std::ios::cur);
                }
                continue;
            }
            // x = num_terms; per term: i32 alpha_exp + packed ternary codes
            size_t n_elements = (size_t)out_f * in_f;
            size_t n_bytes = (n_elements * 2 + 7) / 8;
            if (keep) {
                LayerData ld;
                ld.name = std::move(name);
                ld.out_features = (int32_t)out_f;
                ld.in_features = (int32_t)in_f;
                ld.terms.resize(x);
                for (uint32_t t = 0; t < x; t++) {
                    auto& term = ld.terms[t];
                    f.read(reinterpret_cast<char*>(&term.alpha_exp), 4);
                    std::vector<uint8_t> packed(n_bytes);
                    f.read(reinterpret_cast<char*>(packed.data()), n_bytes);
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
                kept.push_back(std::move(ld));
            } else {
                // Skip all terms: (4 bytes alpha_exp + n_bytes) each
                f.seekg((std::streamoff)x * (4 + (std::streamoff)n_bytes),
                        std::ios::cur);
            }
            continue;
        }

        // ─── NEW ALS layout: u8 layer_type + u32 data_len ───────────────
        uint8_t layer_type = (uint8_t)(x & 0xFF);
        uint32_t data_len = (x >> 8) & 0x00FFFFFF;
        uint8_t len_hi;
        f.read(reinterpret_cast<char*>(&len_hi), 1);
        data_len |= (uint32_t(len_hi) << 24);

        if (keep) {
            if (layer_type > 2) {
                Logger::error("ALS layer {}: bad layer_type {}", name, (int)layer_type);
                throw std::runtime_error("ALS layer: bad layer_type in model file");
            }
            LayerData ld;
            ld.name = std::move(name);
            ld.out_features = (int32_t)out_f;
            ld.in_features = (int32_t)in_f;
            std::vector<uint8_t> data(data_len);
            f.read(reinterpret_cast<char*>(data.data()), data_len);
            parse_layer_type_data(ld, layer_type, data, /*build_combined=*/false);
            kept.push_back(std::move(ld));
        } else {
            f.seekg((std::streamoff)data_len, std::ios::cur);
        }
    }
    return kept;
}

// ─── Shard model bundle ────────────────────────────────────────────────────
struct ShardModel {
    ModelConfig cfg;
    std::vector<float> embedding;     // non-empty iff spec.is_first
    std::vector<float> final_norm;    // non-empty iff spec.is_last
    std::vector<NormWeights> layer_norms;  // FULL vector (indexed globally)
    std::vector<LayerData> layers;    // slice only (found by name at runtime)
    ShardSpec spec;
    RoPECache rope;
    KVCache kv;                       // FULL size (indexed by global layer idx)
};

// ─── Shard loader entry point ──────────────────────────────────────────────
inline ShardModel load_model_shard(const std::string& model_dir,
                                   const ShardSpec& in_spec) {
    ShardModel m;
    ShardSpec spec = in_spec;

    // ── Detect format ───────────────────────────────────────────────────
    std::string gguf_path = model_path_for(model_dir);

    if (!gguf_path.empty()) {
        // ── GGUF path (v1: memory-heavy full load, then filter) ─────────
        LoadedModel full = load_model_from(model_dir);
        m.cfg = full.cfg;
        m.embedding = full.embedding;
        m.final_norm = full.final_norm;
        m.layer_norms = full.layer_norms;
        for (auto& ld : full.layers) {
            int li = parse_layer_index(ld.name);
            bool keep = (li >= spec.start_layer && li < spec.end_layer) ||
                        (spec.is_last && ld.name == "lm_head");
            if (keep) m.layers.push_back(std::move(ld));
        }
    } else {
        // ── ALS (.bin) path ─────────────────────────────────────────────
        std::string extra_path = model_dir + "/model_extra.bin";
        std::string als_path = model_dir + "/model_decomposed.bin";

        struct stat st;
        if (stat(extra_path.c_str(), &st) != 0) {
            Logger::error("No model files found in {}", model_dir);
            throw std::runtime_error("No model files found in " + model_dir);
        }

        m.cfg = load_config(extra_path);

        // Validate the shard range against the real model before loading
        // anything heavy (fail fast on a bad spec).
        if (spec.start_layer < 0 || spec.end_layer < spec.start_layer ||
            spec.end_layer > m.cfg.num_hidden_layers) {
            throw std::runtime_error(
                "Invalid shard range [" + std::to_string(spec.start_layer) +
                ", " + std::to_string(spec.end_layer) + ") for model with " +
                std::to_string(m.cfg.num_hidden_layers) + " layers");
        }

        if (spec.is_first) {
            m.embedding = load_embedding(extra_path, m.cfg);
        }
        if (spec.is_last) {
            m.final_norm = load_final_norm(extra_path, m.cfg);
        }
        m.layer_norms = load_layer_norms(extra_path, m.cfg);
        m.layers = load_decomposed_layers_als_range(als_path, spec.start_layer,
                                                    spec.end_layer, spec.is_last);

        // ── Qwen3-style per-head Q/K RMSNorm + head_dim fixup ──────────
        // Mirrors loader.h load_model_from. q_norm/k_norm pseudo-layers are
        // only present in the slice when their layer N is in range, so the
        // lifting stays shard-local and consistent.
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
    }

    // ── Common post-processing ──────────────────────────────────────────
    // Self-consistent flags: the worker's forward path depends on is_first /
    // is_last matching the actual range, so derive them from the range even
    // if the coordinator's flags disagreed.
    spec.n_layers = m.cfg.num_hidden_layers;
    spec.is_first = (spec.start_layer == 0);
    spec.is_last = (spec.end_layer == m.cfg.num_hidden_layers);

    m.spec = spec;
    m.rope = build_rope_cache(m.cfg.max_position_embeddings, m.cfg.head_dim,
                              m.cfg.rope_theta);
    m.kv = KVCache(m.cfg.max_position_embeddings, m.cfg.num_hidden_layers,
                   m.cfg.num_key_value_heads, m.cfg.head_dim, m.cfg.hidden_size);

    Logger::info("shard loaded: layers [{}:{}), is_first={} is_last={}, {} layers in slice",
                 spec.start_layer, spec.end_layer, spec.is_first, spec.is_last,
                 m.layers.size());
    return m;
}

}  // namespace tldist

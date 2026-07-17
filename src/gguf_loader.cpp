/*
 * gguf_loader.cpp — GGUF format parser + Q2_0 decoder implementation
 *
 * Reads GGUF v3 files, extracts model config from metadata,
 * decodes Q2_0 quantized layers to I2_S format, and extracts
 * unquantized tensors (embedding, norms) for the inference engine.
 */
#include "gguf_loader.h"
#include "loader.h"
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <algorithm>
#include <sstream>

// ═══════════════════════════════════════════════════════════════════════════
// INTERNAL: read helpers (all GGUF is little-endian)
// ═══════════════════════════════════════════════════════════════════════════

// Note: x86 is LE, so memcpy is fine. On big-endian hosts we'd need byteswap.

static uint8_t  r8 (const uint8_t* p, size_t& off) { uint8_t  v; std::memcpy(&v, p+off, 1); off+=1; return v; }
static uint16_t r16(const uint8_t* p, size_t& off) { uint16_t v; std::memcpy(&v, p+off, 2); off+=2; return v; }
static uint32_t r32(const uint8_t* p, size_t& off) { uint32_t v; std::memcpy(&v, p+off, 4); off+=4; return v; }
static uint64_t r64(const uint8_t* p, size_t& off) { uint64_t v; std::memcpy(&v, p+off, 8); off+=8; return v; }
static float   rf32(const uint8_t* p, size_t& off) { float    v; std::memcpy(&v, p+off, 4); off+=4; return v; }

static std::string rstr(const uint8_t* p, size_t& off) {
    uint64_t len = r64(p, off);
    std::string s((const char*)(p + off), (size_t)len);
    off += (size_t)len;
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// PARSE GGUF FILE
// ═══════════════════════════════════════════════════════════════════════════

static GGUFFile parse_gguf_internal(const std::string& path) {
    GGUFFile gguf;

    // Read entire file into memory
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "GGUF: Cannot open " << path << std::endl;
        return gguf;
    }
    size_t file_size = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    gguf.file_data.resize(file_size);
    f.read(reinterpret_cast<char*>(gguf.file_data.data()), file_size);
    f.close();

    const uint8_t* p = gguf.file_data.data();
    size_t off = 0;

    // ── Header ───────────────────────────────────────────────────────────
    uint32_t magic = r32(p, off);
    if (magic != GGUF_MAGIC) {
        std::cerr << "GGUF: Bad magic 0x" << std::hex << magic
                  << " (expected 0x" << GGUF_MAGIC << ")" << std::dec << std::endl;
        return gguf;
    }
    uint32_t version = r32(p, off);
    if (version > GGUF_VERSION) {
        std::cerr << "GGUF: Unsupported version " << version
                  << " (max " << GGUF_VERSION << ")" << std::endl;
        return gguf;
    }
    uint64_t tensor_count      = r64(p, off);
    uint64_t metadata_kv_count = r64(p, off);

    // ── Metadata KV pairs ────────────────────────────────────────────────
    for (uint64_t i = 0; i < metadata_kv_count; i++) {
        std::string key = rstr(p, off);

        if (off + 4 > file_size) break;
        uint32_t val_type = r32(p, off);

        switch (val_type) {
            case GGUF_TYPE_UINT8: {
                uint8_t v = r8(p, off);
                gguf.metadata_int[key] = v;
                break;
            }
            case GGUF_TYPE_INT8: {
                int8_t v = (int8_t)r8(p, off);
                gguf.metadata_int[key] = v;
                break;
            }
            case GGUF_TYPE_UINT16: {
                uint16_t v = r16(p, off);
                gguf.metadata_int[key] = v;
                break;
            }
            case GGUF_TYPE_INT16: {
                uint16_t v = r16(p, off);
                gguf.metadata_int[key] = (int16_t)v;
                break;
            }
            case GGUF_TYPE_UINT32: {
                uint32_t v = r32(p, off);
                gguf.metadata_int[key] = v;
                break;
            }
            case GGUF_TYPE_INT32: {
                uint32_t v = r32(p, off);
                gguf.metadata_int[key] = (int32_t)v;
                break;
            }
            case GGUF_TYPE_UINT64: {
                uint64_t v = r64(p, off);
                gguf.metadata_int[key] = (int64_t)v;
                break;
            }
            case GGUF_TYPE_INT64: {
                uint64_t v = r64(p, off);
                gguf.metadata_int[key] = (int64_t)v;
                break;
            }
            case GGUF_TYPE_FLOAT32: {
                float v = rf32(p, off);
                gguf.metadata_float[key] = v;
                break;
            }
            case GGUF_TYPE_FLOAT64: {
                // Read as double, store as float
                uint64_t raw;
                std::memcpy(&raw, p + off, 8); off += 8;
                double dv;
                std::memcpy(&dv, &raw, sizeof(dv));
                gguf.metadata_float[key] = (float)dv;
                break;
            }
            case GGUF_TYPE_BOOL: {
                uint8_t v = r8(p, off);
                gguf.metadata_int[key] = v ? 1 : 0;
                break;
            }
            case GGUF_TYPE_STRING: {
                std::string v = rstr(p, off);
                gguf.metadata_str[key] = v;
                break;
            }
            case GGUF_TYPE_ARRAY: {
                if (off + 4 > file_size) break;
                uint32_t elem_type = r32(p, off);
                uint64_t elem_count = r64(p, off);
                // Skip array elements (we don't need arrays for config)
                for (uint64_t j = 0; j < elem_count; j++) {
                    switch (elem_type) {
                        case GGUF_TYPE_UINT8:
                        case GGUF_TYPE_INT8:   off += 1; break;
                        case GGUF_TYPE_UINT16:
                        case GGUF_TYPE_INT16:  off += 2; break;
                        case GGUF_TYPE_UINT32:
                        case GGUF_TYPE_INT32:
                        case GGUF_TYPE_FLOAT32: off += 4; break;
                        case GGUF_TYPE_UINT64:
                        case GGUF_TYPE_INT64:
                        case GGUF_TYPE_FLOAT64: off += 8; break;
                        case GGUF_TYPE_STRING: rstr(p, off); break;
                        default: off += 4; break;
                    }
                }
                break;
            }
            default:
                // Unknown type: skip 4 bytes and hope for the best
                off += 4;
                break;
        }
    }

    // ── Tensor info entries ──────────────────────────────────────────────
    for (uint64_t i = 0; i < tensor_count; i++) {
        GGUFTensorInfo ti;
        ti.name = rstr(p, off);

        if (off + 4 > file_size) break;
        uint32_t n_dims = r32(p, off);

        ti.dims.resize(n_dims);
        for (uint32_t d = 0; d < n_dims; d++) {
            ti.dims[d] = r64(p, off);
        }

        ti.type   = r32(p, off);
        ti.offset = r64(p, off);

        // Alignment padding (tensor info entries are 32-byte aligned)
        // Off is left at end of this entry; next entry starts at alignment

        gguf.tensors.push_back(ti);
    }

    gguf.valid = true;
    return gguf;
}

// ═══════════════════════════════════════════════════════════════════════════
// HELPER: find tensor by name
// ═══════════════════════════════════════════════════════════════════════════

static const GGUFTensorInfo* find_tensor(const std::vector<GGUFTensorInfo>& tensors,
                                          const std::string& name) {
    for (const auto& t : tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// EXTRACT UNQUANTIZED TENSOR (F32/F16) TO FLOAT VECTOR
// ═══════════════════════════════════════════════════════════════════════════

static bool extract_f32_tensor(const GGUFFile& gguf,
                                const GGUFTensorInfo& ti,
                                std::vector<float>& out) {
    if (ti.offset + 4 > gguf.file_data.size()) return false;
    const uint8_t* src = gguf.file_data.data() + (size_t)ti.offset;

    size_t n_elems = 1;
    for (auto d : ti.dims) n_elems *= (size_t)d;

    if (ti.type == GGML_TYPE_F32) {
        out.resize(n_elems);
        std::memcpy(out.data(), src, n_elems * 4);
    } else if (ti.type == GGML_TYPE_F16) {
        out.resize(n_elems);
        for (size_t i = 0; i < n_elems; i++) {
            uint16_t fp16;
            std::memcpy(&fp16, src + i * 2, 2);
            out[i] = fp16_to_fp32(fp16);
        }
    } else if (ti.type == GGML_TYPE_Q2_0 || ti.type == 42) {
        // Q2_0 (or TQ2_0_GS8) with g128: 34 bytes/block
        out.resize(n_elems);
        uint32_t qk = Q2_0_BLOCK_SIZE;
        uint32_t block_bytes = Q2_0_BLOCK_BYTES;
        uint32_t n_blocks = (uint32_t)((n_elems + qk - 1) / qk);
        for (uint32_t b = 0; b < n_blocks; b++) {
            uint16_t scale_fp16;
            std::memcpy(&scale_fp16, src + b * block_bytes, 2);
            float scale = fp16_to_fp32(scale_fp16);
            const uint8_t* codes = src + b * block_bytes + 2;
            uint32_t end = std::min(qk, (uint32_t)(n_elems - b * qk));
            for (uint32_t j = 0; j < end; j++) {
                uint8_t code = (codes[j / 4] >> (6 - (j % 4) * 2)) & 0x03;
                // code: 0→-1, 1→0, 2→+1
                float val = (code == 1) ? 0.0f : (code == 0 ? -1.0f : 1.0f);
                out[b * qk + j] = val * scale;
            }
        }
    } else {
        std::cerr << "GGUF: unexpected type " << ti.type
                  << " for unquantized tensor " << ti.name << std::endl;
        return false;
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// CONVERT Q2_0 TENSOR TO I2_S LAYER DATA
// ═══════════════════════════════════════════════════════════════════════════

static bool convert_q2_0_to_layer(const GGUFFile& gguf,
                                   const GGUFTensorInfo& ti,
                                   LayerData& layer) {
    if (ti.dims.size() < 2) return false;

    uint64_t out_features = ti.dims[0];  // rows
    uint64_t in_features  = ti.dims[1];  // cols

    layer.out_features = (int32_t)out_features;
    layer.in_features  = (int32_t)in_features;
    layer.has_i2s      = true;
    layer.num_terms    = 0;  // not using bitplane terms for I2_S path
    layer.i2s_qk       = (int32_t)Q2_0_BLOCK_SIZE;

    int qk = (int)Q2_0_BLOCK_SIZE;
    int n_blocks = (int)((in_features + qk - 1) / qk);
    int codes_per_block = qk / 4;  // 32

    layer.i2s_blocks.resize((size_t)out_features * n_blocks);

    const uint8_t* data_ptr = gguf.file_data.data() + (size_t)ti.offset;

    for (uint64_t row = 0; row < out_features; row++) {
        for (int b = 0; b < n_blocks; b++) {
            int block_idx = (int)(row * n_blocks + b);
            int block_start = b * qk;
            // int block_end = std::min(block_start + qk, (int)in_features);

            // Q2_0 block: 2 bytes FP16 scale + 32 bytes codes
            const uint8_t* q2_block = data_ptr + (row * n_blocks + b) * Q2_0_BLOCK_BYTES;

            uint8_t codes[32];
            float scale;
            decode_q2_0_block(q2_block, codes, &scale);

            auto& i2s = layer.i2s_blocks[block_idx];
            i2s.packed.assign(codes, codes + codes_per_block);
            i2s.scale = scale;
        }
    }

    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// BUILD LAYER NAME MAPPING: GGUF → Terllama convention
// ═══════════════════════════════════════════════════════════════════════════
// GGUF: blk.{layer}.attn_q.weight
// Terllama: model.layers.{layer}.self_attn.q_proj

static std::string gguf_name_to_terllama(const std::string& gguf_name) {
    // Remove trailing ".weight"
    std::string name = gguf_name;
    if (name.size() > 7 && name.substr(name.size() - 7) == ".weight") {
        name = name.substr(0, name.size() - 7);
    }

    // Special tensors (not per-layer)
    if (name == "token_embd")            return "token_embd";
    if (name == "output")                return "lm_head";
    if (name == "token_embd_norm")       return "token_embd_norm";

    // Per-layer tensors: blk.{N}.{type}
    // Extract layer number
    if (name.substr(0, 4) != "blk.") return name;

    size_t dot2 = name.find('.', 4);
    if (dot2 == std::string::npos) return name;

    std::string layer_num = name.substr(4, dot2 - 4);
    std::string weight_type = name.substr(dot2 + 1);

    // Map weight type to Terllama naming
    if (weight_type == "attn_norm")         return "model.layers." + layer_num + ".input_layernorm";
    if (weight_type == "ffn_norm")          return "model.layers." + layer_num + ".post_attention_layernorm";
    if (weight_type == "attn_q")            return "model.layers." + layer_num + ".self_attn.q_proj";
    if (weight_type == "attn_k")            return "model.layers." + layer_num + ".self_attn.k_proj";
    if (weight_type == "attn_v")            return "model.layers." + layer_num + ".self_attn.v_proj";
    if (weight_type == "attn_output")       return "model.layers." + layer_num + ".self_attn.o_proj";
    if (weight_type == "ffn_gate")          return "model.layers." + layer_num + ".mlp.gate_proj";
    if (weight_type == "ffn_up")            return "model.layers." + layer_num + ".mlp.up_proj";
    if (weight_type == "ffn_down")          return "model.layers." + layer_num + ".mlp.down_proj";

    return name;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

bool load_gguf_model(const std::string& path,
                     ModelConfig& cfg,
                     std::vector<float>& embedding,
                     std::vector<NormWeights>& layer_norms,
                     std::vector<float>& final_norm,
                     std::vector<LayerData>& layers) {
    // ── Parse GGUF file ──────────────────────────────────────────────────
    GGUFFile gguf = parse_gguf_internal(path);
    if (!gguf.valid) {
        std::cerr << "Failed to parse GGUF file: " << path << std::endl;
        return false;
    }

    // ── Extract model config from metadata ───────────────────────────────
    std::string arch = gguf.metadata_str.count("general.architecture")
        ? gguf.metadata_str["general.architecture"] : "qwen3";

    // Determine metadata key prefix
    std::string prefix = arch + ".";

    auto get_int = [&](const std::string& key, int64_t def) -> int64_t {
        std::string k = prefix + key;
        if (gguf.metadata_int.count(k)) return gguf.metadata_int[k];
        // Try without prefix
        if (gguf.metadata_int.count(key)) return gguf.metadata_int[key];
        return def;
    };

    auto get_float = [&](const std::string& key, float def) -> float {
        std::string k = prefix + key;
        if (gguf.metadata_float.count(k)) return gguf.metadata_float[k];
        if (gguf.metadata_float.count(key)) return gguf.metadata_float[key];
        return def;
    };

    int64_t block_count        = get_int("block_count", 0);
    int64_t context_length     = get_int("context_length", 2048);
    int64_t embedding_length   = get_int("embedding_length", 0);
    int64_t feed_forward_length = get_int("feed_forward_length", 0);
    int64_t head_count         = get_int("attention.head_count", get_int("head_count", 0));
    int64_t head_count_kv      = get_int("attention.head_count_kv", head_count);
    float rms_norm_eps         = get_float("attention.layer_norm_rms_epsilon", 1e-6f);
    float rope_freq_base       = get_float("rope.freq_base", 10000.0f);

    if (embedding_length == 0) {
        std::cerr << "GGUF: missing embedding_length in metadata" << std::endl;
        return false;
    }

    // ── Determine vocab_size from embedding tensor ───────────────────────
    int32_t vocab_size = 0;
    auto* embed_tensor = find_tensor(gguf.tensors, "token_embd.weight");
    if (embed_tensor && embed_tensor->dims.size() >= 2) {
        vocab_size = (int32_t)embed_tensor->dims[0];
    } else {
        std::cerr << "GGUF: token_embd.weight not found" << std::endl;
        return false;
    }

    // ── Populate ModelConfig ─────────────────────────────────────────────
    cfg.vocab_size             = vocab_size;
    cfg.hidden_size            = (int32_t)embedding_length;
    cfg.intermediate_size      = (int32_t)feed_forward_length;
    cfg.num_hidden_layers      = (int32_t)block_count;
    cfg.num_attention_heads    = (int32_t)head_count;
    cfg.num_key_value_heads    = (int32_t)head_count_kv;
    cfg.rms_norm_eps           = rms_norm_eps;
    cfg.rope_theta             = rope_freq_base;
    cfg.max_position_embeddings = (int32_t)context_length;
    cfg.head_dim               = (int32_t)(embedding_length / head_count);

    // ── Extract embedding ────────────────────────────────────────────────
    if (!extract_f32_tensor(gguf, *embed_tensor, embedding)) {
        std::cerr << "GGUF: failed to extract embedding" << std::endl;
        return false;
    }

    // ── Extract final norm ───────────────────────────────────────────────
    auto* fn_tensor = find_tensor(gguf.tensors, "token_embd_norm.weight");
    if (fn_tensor) {
        extract_f32_tensor(gguf, *fn_tensor, final_norm);
    } else {
        // Some models use "output_norm.weight" instead
        fn_tensor = find_tensor(gguf.tensors, "output_norm.weight");
        if (fn_tensor) extract_f32_tensor(gguf, *fn_tensor, final_norm);
        else final_norm.resize(cfg.hidden_size, 1.0f);
    }

    // ── Extract layer norms and quantized layers ─────────────────────────
    layer_norms.resize(cfg.num_hidden_layers);

    // Pre-allocate layers list (7 linear layers per transformer block + 1 lm_head)
    layers.clear();
    layers.reserve(cfg.num_hidden_layers * 7 + 1);

    // Collect all norm tensors and Q2_0 tensors, sorted by layer
    for (int i = 0; i < cfg.num_hidden_layers; i++) {
        std::string layer_str = std::to_string(i);

        // Input layernorm
        auto* in_norm = find_tensor(gguf.tensors, "blk." + layer_str + ".attn_norm.weight");
        if (!in_norm) in_norm = find_tensor(gguf.tensors,
            "blk." + layer_str + ".attn_norm.weight"); // fallback
        if (in_norm) {
            extract_f32_tensor(gguf, *in_norm, layer_norms[i].input_layernorm);
        } else {
            layer_norms[i].input_layernorm.resize(cfg.hidden_size, 1.0f);
        }

        // Post-attention layernorm
        auto* pa_norm = find_tensor(gguf.tensors, "blk." + layer_str + ".ffn_norm.weight");
        if (!pa_norm) pa_norm = find_tensor(gguf.tensors,
            "blk." + layer_str + ".ffn_norm.weight"); // fallback
        if (pa_norm) {
            extract_f32_tensor(gguf, *pa_norm, layer_norms[i].post_attention_layernorm);
        } else {
            layer_norms[i].post_attention_layernorm.resize(cfg.hidden_size, 1.0f);
        }

        // 7 quantized linear layers per transformer block
        // (names built dynamically since layer index is runtime)
    }

    // Second pass: collect Q2_0 tensors
    for (int i = 0; i < cfg.num_hidden_layers; i++) {
        std::string li = std::to_string(i);

        auto add_q2_layer = [&](const std::string& gguf_tensor_name,
                                 const std::string& terllama_name) {
            auto* ti = find_tensor(gguf.tensors, gguf_tensor_name);
            if (!ti) {
                std::cerr << "GGUF: missing tensor " << gguf_tensor_name << std::endl;
                return;
            }
            LayerData ld;
            ld.name = terllama_name;
            if (!convert_q2_0_to_layer(gguf, *ti, ld)) {
                std::cerr << "GGUF: failed to convert " << gguf_tensor_name << std::endl;
                return;
            }
            layers.push_back(std::move(ld));
        };

        // To avoid string concatenation in the lambda calls, precompute names
        std::string q = "blk." + li + ".attn_q.weight";
        std::string k = "blk." + li + ".attn_k.weight";
        std::string v = "blk." + li + ".attn_v.weight";
        std::string o = "blk." + li + ".attn_output.weight";
        std::string g = "blk." + li + ".ffn_gate.weight";
        std::string u = "blk." + li + ".ffn_up.weight";
        std::string d = "blk." + li + ".ffn_down.weight";

        std::string tq = "model.layers." + li + ".self_attn.q_proj";
        std::string tk = "model.layers." + li + ".self_attn.k_proj";
        std::string tv = "model.layers." + li + ".self_attn.v_proj";
        std::string to = "model.layers." + li + ".self_attn.o_proj";
        std::string tg = "model.layers." + li + ".mlp.gate_proj";
        std::string tu = "model.layers." + li + ".mlp.up_proj";
        std::string td = "model.layers." + li + ".mlp.down_proj";

        add_q2_layer(q, tq);
        add_q2_layer(k, tk);
        add_q2_layer(v, tv);
        add_q2_layer(o, to);
        add_q2_layer(g, tg);
        add_q2_layer(u, tu);
        add_q2_layer(d, td);
    }

    // ── Extract lm_head ──────────────────────────────────────────────────
    auto* lm_head_tensor = find_tensor(gguf.tensors, "output.weight");
    if (lm_head_tensor) {
        LayerData lm_ld;
        lm_ld.name = "lm_head";
        if (lm_head_tensor->type == GGML_TYPE_F32 || lm_head_tensor->type == GGML_TYPE_F16) {
            // Unquantized lm_head
            lm_ld.has_raw_weights = true;
            lm_ld.out_features = (int32_t)lm_head_tensor->dims[0];
            lm_ld.in_features  = (int32_t)lm_head_tensor->dims[1];
            extract_f32_tensor(gguf, *lm_head_tensor, lm_ld.raw_weights);
        } else {
            // Quantized lm_head
            convert_q2_0_to_layer(gguf, *lm_head_tensor, lm_ld);
        }
        layers.push_back(std::move(lm_ld));
    }

    std::cout << "  GGUF model loaded: " << arch
              << ", " << cfg.num_hidden_layers << " layers"
              << ", " << layers.size() << " linear layers" << std::endl;

    return true;
}

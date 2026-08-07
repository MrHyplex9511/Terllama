// ═══════════════════════════════════════════════════════════════════════════
// export.cpp — Native ALS model converter (Track C).
//
// Byte-compatible port of scripts/export_ternary_model_bitnet.py:
//   * model_decomposed.bin — magic 0xDEADBEEF, per-layer
//     [name_len u32][name][out_f u32][in_f u32][layer_type u8][data_len u32][data]
//     layer_type: 1 = RAW_FP32 (row-major fp32), 2 = multi-term ALS container
//     (pack_als_block_terms blob). Layers are emitted in PyTorch named_modules
//     order (self_attn q/k/v/o then mlp gate/up/down per layer, lm_head after
//     all layers, Qwen3 q_norm/k_norm RAW pseudo-layers appended last).
//   * model_extra.bin — 9 LE config fields, embedding [vocab,hidden] fp32,
//     final_norm [hidden] fp32, per-layer input_layernorm + post_attention_
//     layernorm [hidden] fp32 (matches loader.h load_config/load_embedding/
//     load_final_norm/load_layer_norms).
//
// Layer enumeration replicates get_model_layers(): only nn.Linear modules are
// exported (embed_tokens, RMSNorms, layer norms are skipped in the decomposed
// file); fused qkv_proj is split into q/k/v. The safetensors checkpoint stores
// tensor names lexicographically sorted, which does NOT match named_modules
// traversal, so the canonical module order is reconstructed explicitly.
// ═══════════════════════════════════════════════════════════════════════════

#include "convert/export.h"

#include "convert/als_decompose.h"
#include "convert/hf_download.h"
#include "convert/safetensors.h"
#include "core/logger.h"
#include <json.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

namespace terllama {
namespace {

// ── Small helpers ──────────────────────────────────────────────────────────

bool file_exists(const std::string& p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

// Recursive mkdir -p. Builds each prefix incrementally (keeping slashes),
// creating only the components that don't exist yet.
bool mkdir_p(const std::string& path) {
    if (path.empty()) return false;
    if (file_exists(path)) return true;
    std::string cur;
    for (size_t i = 0; i <= path.size(); i++) {
        cur += path[i];
        if (i == path.size() || path[i] == '/') {
            if (cur.empty() || cur == "/") continue;
            struct stat st;
            if (stat(cur.c_str(), &st) != 0) {
                if (mkdir(cur.c_str(), 0755) != 0) return false;
            }
        }
    }
    return true;
}

std::string read_file_str(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return std::string();
    std::string out((std::istreambuf_iterator<char>(f)),
                    std::istreambuf_iterator<char>());
    return out;
}

// ── Config ─────────────────────────────────────────────────────────────────

struct ModelCfg {
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    float rms_norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    int32_t max_position_embeddings = 0;
};

// Parse the 9 config fields written to model_extra.bin from config.json.
// Mirrors the Python reference: num_key_value_heads defaults to
// num_attention_heads, rope_theta defaults to 10000.0.
bool parse_config(const std::string& path, ModelCfg& cfg) {
    const std::string body = read_file_str(path);
    if (body.empty()) {
        Logger::error("config.json is empty or unreadable: {}", path);
        return false;
    }
    json j;
    try {
        j = json::parse(body);
    } catch (const std::exception& e) {
        Logger::error("config.json parse failed: {}", e.what());
        return false;
    }
    if (!j.is_object()) {
        Logger::error("config.json is not a JSON object{}", "");
        return false;
    }
    auto req_int = [&](const char* key, int32_t& out) -> bool {
        if (!j.contains(key) || !j[key].is_number_integer()) {
            Logger::error("config.json: missing/invalid '{}'", key);
            return false;
        }
        out = j[key].get<int32_t>();
        return true;
    };
    if (!req_int("vocab_size", cfg.vocab_size) ||
        !req_int("hidden_size", cfg.hidden_size) ||
        !req_int("intermediate_size", cfg.intermediate_size) ||
        !req_int("num_hidden_layers", cfg.num_hidden_layers) ||
        !req_int("num_attention_heads", cfg.num_attention_heads) ||
        !req_int("max_position_embeddings", cfg.max_position_embeddings)) {
        return false;
    }
    if (!j.contains("rms_norm_eps") || !j["rms_norm_eps"].is_number()) {
        Logger::error("config.json: missing/invalid 'rms_norm_eps'");
        return false;
    }
    cfg.rms_norm_eps = j["rms_norm_eps"].get<float>();
    if (j.contains("num_key_value_heads") && j["num_key_value_heads"].is_number_integer()) {
        cfg.num_key_value_heads = j["num_key_value_heads"].get<int32_t>();
    } else {
        cfg.num_key_value_heads = cfg.num_attention_heads;
    }
    if (j.contains("rope_theta") && j["rope_theta"].is_number()) {
        cfg.rope_theta = j["rope_theta"].get<float>();
    }
    if (cfg.vocab_size <= 0 || cfg.hidden_size <= 0 ||
        cfg.num_hidden_layers <= 0 || cfg.num_attention_heads <= 0 ||
        cfg.num_key_value_heads <= 0) {
        Logger::error("config.json: non-positive model dimensions{}", "");
        return false;
    }
    return true;
}

// ── Output layer record ────────────────────────────────────────────────────

struct OutLayer {
    std::string name;
    int32_t out_f = 0;
    int32_t in_f = 0;
    uint8_t layer_type = 0;
    std::vector<uint8_t> data;
};

void progress(int done, int total) {
    const int pct = total > 0 ? (int)(100.0 * done / total) : 0;
    printf("[PROGRESS] %d%%\n", pct);
    fflush(stdout);
}

// ── Layer enumeration ──────────────────────────────────────────────────────
// Matches the Python get_model_layers(): only Linear modules are exported to
// the decomposed file. A Linear is recognized by its name: it either contains
// one of the quantized projection suffixes, or is the top-level lm_head
// (embed_tokens and the various RMSNorm weights are skipped — they live in
// model_extra.bin / are pseudo-layers instead).

bool contains_any(const std::string& s,
                  const std::vector<const char*>& subs) {
    for (const char* sub : subs)
        if (s.find(sub) != std::string::npos) return true;
    return false;
}

bool is_linear_name(const std::string& n) {
    static const std::vector<const char*> quant = {
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", "qkv_proj"};
    return contains_any(n, quant) || n == "lm_head";
}

// Canonical per-layer module order (PyTorch registration order for the
// Llama/Qwen/SmolLM model family, matching named_modules traversal).
const char* k_layer_submodules[] = {
    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
    "self_attn.o_proj", "mlp.gate_proj",    "mlp.up_proj",
    "mlp.down_proj",
};

// Ordered Linear references: (module name without .weight, tensor).
struct LinearRef {
    std::string name;       // e.g. "model.layers.0.self_attn.q_proj"
    const STTensor* t = nullptr;
};

std::vector<LinearRef> enumerate_linears(
    const std::unordered_map<std::string, STTensor>& tensors,
    const ModelCfg& cfg, std::string& err) {
    std::vector<LinearRef> out;

    for (int32_t i = 0; i < cfg.num_hidden_layers; i++) {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        for (const char* sub : k_layer_submodules) {
            const std::string full = prefix + sub;
            // Fused qkv_proj (Phi-3 style) lives where q_proj would be.
            if (std::strcmp(sub, "self_attn.q_proj") == 0) {
                const std::string qkv = prefix + "self_attn.qkv_proj";
                if (tensors.count(qkv + ".weight")) {
                    out.push_back({qkv, &tensors.at(qkv + ".weight")});
                    continue;
                }
            }
            if (tensors.count(full + ".weight"))
                out.push_back({full, &tensors.at(full + ".weight")});
        }
    }

    // lm_head after all layers. For tied-embedding models the checkpoint may
    // omit lm_head.weight; synthesize it from the shared embedding matrix
    // (the reference model shares the same parameter object).
    if (tensors.count("lm_head.weight")) {
        out.push_back({"lm_head", &tensors.at("lm_head.weight")});
    } else if (tensors.count("model.embed_tokens.weight")) {
        Logger::info("lm_head.weight absent (tied embeddings); synthesizing "
                     "from model.embed_tokens.weight{}", "");
        out.push_back({"lm_head", &tensors.at("model.embed_tokens.weight")});
    } else {
        err = "no lm_head.weight and no model.embed_tokens.weight";
        return {};
    }
    return out;
}

// ── Binary writers ─────────────────────────────────────────────────────────

bool write_le_u32(std::ofstream& f, uint32_t v) {
    f.write(reinterpret_cast<const char*>(&v), 4);
    return !!f;
}

bool write_decomposed(const std::string& path,
                      const std::vector<OutLayer>& layers) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        Logger::error("Cannot open for write: {}", path);
        return false;
    }
    write_le_u32(f, 0xDEADBEEFu);
    write_le_u32(f, (uint32_t)layers.size());
    for (const OutLayer& L : layers) {
        write_le_u32(f, (uint32_t)L.name.size());
        f.write(L.name.data(), (std::streamsize)L.name.size());
        write_le_u32(f, (uint32_t)L.out_f);
        write_le_u32(f, (uint32_t)L.in_f);
        f.put((char)L.layer_type);
        write_le_u32(f, (uint32_t)L.data.size());
        f.write(reinterpret_cast<const char*>(L.data.data()),
                (std::streamsize)L.data.size());
    }
    f.close();
    return !!f;
}

// Write model_extra.bin: 9 LE config fields, embedding, final_norm, and the
// per-layer RMSNorm pairs. Mirrors loader.h read order exactly.
bool write_extra(const std::string& path, const ModelCfg& cfg,
                 const std::unordered_map<std::string, STTensor>& tensors,
                 std::string& err) {
    auto req = [&](const char* name) -> const STTensor* {
        auto it = tensors.find(std::string(name));
        if (it == tensors.end()) {
            err = std::string("missing required weight: ") + name;
            return nullptr;
        }
        return &it->second;
    };

    const STTensor* emb = req("model.embed_tokens.weight");
    if (!emb) return false;
    if ((int64_t)emb->data.size() != (int64_t)cfg.vocab_size * cfg.hidden_size) {
        err = "embed_tokens size mismatch vs config";
        return false;
    }

    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) {
        Logger::error("Cannot open for write: {}", path);
        return false;
    }

    // 9 config fields (LE): 6 x i32, 2 x f32, 1 x i32.
    const int32_t i_fields[] = {cfg.vocab_size,          cfg.hidden_size,
                                cfg.intermediate_size,   cfg.num_hidden_layers,
                                cfg.num_attention_heads, cfg.num_key_value_heads};
    for (int32_t v : i_fields) {
        f.write(reinterpret_cast<const char*>(&v), 4);
    }
    f.write(reinterpret_cast<const char*>(&cfg.rms_norm_eps), 4);
    f.write(reinterpret_cast<const char*>(&cfg.rope_theta), 4);
    f.write(reinterpret_cast<const char*>(&cfg.max_position_embeddings), 4);

    // Embedding [vocab, hidden] fp32 row-major.
    f.write(reinterpret_cast<const char*>(emb->data.data()),
            (std::streamsize)(emb->data.size() * sizeof(float)));

    // Final norm [hidden] fp32; zeros if absent (matches reference fallback).
    std::vector<float> final_norm((size_t)cfg.hidden_size, 0.0f);
    auto it_norm = tensors.find("model.norm.weight");
    if (it_norm != tensors.end()) {
        if ((int64_t)it_norm->second.data.size() != cfg.hidden_size) {
            err = "model.norm.weight size mismatch";
            return false;
        }
        final_norm = it_norm->second.data;
    } else {
        Logger::warn("model.norm.weight absent; writing zero final norm{}", "");
    }
    f.write(reinterpret_cast<const char*>(final_norm.data()),
            (std::streamsize)(final_norm.size() * sizeof(float)));

    // Per-layer input_layernorm + post_attention_layernorm (REQUIRED).
    for (int32_t i = 0; i < cfg.num_hidden_layers; i++) {
        const std::string pre =
            "model.layers." + std::to_string(i) + ".";
        const char* norm_names[2] = {"input_layernorm",
                                     "post_attention_layernorm"};
        for (const char* nn : norm_names) {
            const STTensor* t = req((pre + nn + ".weight").c_str());
            if (!t) return false;
            if ((int64_t)t->data.size() != cfg.hidden_size) {
                err = (pre + nn) + ".weight size mismatch";
                return false;
            }
            f.write(reinterpret_cast<const char*>(t->data.data()),
                    (std::streamsize)(t->data.size() * sizeof(float)));
        }
    }

    f.close();
    return !!f;
}

} // namespace

// ═══════════════════════════════════════════════════════════════════════════
// convert_model — full native pull + convert
// ═══════════════════════════════════════════════════════════════════════════
int convert_model(const std::string& repo_id, const std::string& outdir,
                  int num_terms, const std::string& format) {
#if !defined(TERLLAMA_HAVE_CURL)
    (void)repo_id; (void)outdir; (void)num_terms; (void)format;
    Logger::error("built without curl, cannot pull; use pre-converted weights{}", "");
    Logger::error("(rebuild with libcurl: TERLLAMA_HAVE_CURL){}", "");
    return 1;
#else
    if (num_terms < 1) {
        Logger::error("--terms must be >= 1 (got {})", num_terms);
        return 1;
    }
    if (format != "als" && format != "gguf") {
        Logger::error("unknown format '{}' (only 'als' is supported; 'gguf' "
                      "routes to als like the Python reference)", format);
        return 1;
    }

    progress(0, 100);
    Logger::info("Converting {} -> ALS per-block format ({} terms)", repo_id,
                 num_terms);

    if (!mkdir_p(outdir)) {
        Logger::error("cannot create output dir '{}'", outdir);
        return 1;
    }

    // ── 1. Download config.json + tokenizer files (config.json required).
    const std::string cfg_path = outdir + "/config.json";
    if (hf_download_file(repo_id, "config.json", outdir, "") != 0) {
        Logger::error("failed to download config.json from {}", repo_id);
        return 1;
    }
    const char* optional_files[] = {
        "tokenizer_config.json", "vocab.json",   "merges.txt",
        "special_tokens_map.json", "added_tokens.json",
    };
    for (const char* fn : optional_files) {
        if (hf_download_file(repo_id, fn, outdir, "") != 0) {
            Logger::warn("optional file '{}' not found on hub; skipping", fn);
        }
    }
    if (hf_download_file(repo_id, "tokenizer.json", outdir, "") != 0) {
        Logger::error("failed to download tokenizer.json from {} (required)", repo_id);
        return 1;
    }

    // ── 2. Config.
    ModelCfg cfg;
    if (!parse_config(cfg_path, cfg)) return 1;

    // ── 3. Weights: single model.safetensors or sharded set.
    const std::string cfg_body = read_file_str(cfg_path);
    std::vector<std::string> weight_files;
    if (should_use_index(cfg_body)) {
        Logger::info("Sharded checkpoint detected; downloading shards{}", "");
        if (hf_download_sharded(repo_id, outdir, "", weight_files) != 0) {
            Logger::error("sharded download failed for {}", repo_id);
            return 1;
        }
    } else {
        if (hf_download_file(repo_id, "model.safetensors", outdir, "") != 0) {
            Logger::error("failed to download model.safetensors from {}", repo_id);
            return 1;
        }
        weight_files.push_back(outdir + "/model.safetensors");
    }

    // ── 4. Load all tensors into one map (fp32). Each tensor lives in exactly
    // one shard (HF sharding never splits a single tensor across files).
    std::unordered_map<std::string, STTensor> tensors;
    for (const std::string& wf : weight_files) {
        std::vector<STTensor> parsed;
        if (!load_safetensors_file(wf, parsed)) {
            Logger::error("failed to parse safetensors '{}'", wf);
            return 1;
        }
        for (STTensor& t : parsed) {
            auto ins = tensors.emplace(t.name, std::move(t));
            if (!ins.second) {
                Logger::error("duplicate tensor name '{}' across weight files", t.name);
                return 1;
            }
        }
    }
    Logger::info("Loaded {} tensors from {} weight file(s)", tensors.size(),
                 weight_files.size());

    // ── 5. Enumerate Linear layers in named_modules order.
    std::string err;
    std::vector<LinearRef> linears = enumerate_linears(tensors, cfg, err);
    if (linears.empty()) {
        Logger::error("no layers enumerated: {}", err);
        return 1;
    }

    // Qwen3-style Q/K RMSNorm pseudo-layers: exported as RAW FP32 layers named
    // model.layers.<N>.self_attn.q_norm / .k_norm (appended AFTER all linears,
    // q_norms first then k_norms — matching the reference).
    const bool has_q_norm =
        tensors.count("model.layers.0.self_attn.q_norm.weight") > 0;
    const bool has_k_norm =
        tensors.count("model.layers.0.self_attn.k_norm.weight") > 0;
    int n_qk_norm_total = 0;
    if (has_q_norm) n_qk_norm_total += (int)cfg.num_hidden_layers;
    if (has_k_norm) n_qk_norm_total += (int)cfg.num_hidden_layers;

    const int total_layers = (int)linears.size() + n_qk_norm_total;
    int done = 0;

    // ── 6. Process each layer: RAW fp32 or ALS multi-term.
    std::vector<OutLayer> out_layers;
    int n_quantized = 0, n_raw = 0;
    const int head_dim = cfg.hidden_size / cfg.num_attention_heads;

    auto emit_raw = [&](const std::string& name, const STTensor& t) {
        OutLayer L;
        L.name = name;
        L.out_f = (int32_t)t.shape[0];
        L.in_f = (int32_t)(t.shape.size() > 1 ? t.shape[1] : t.shape[0]);
        L.layer_type = 1;
        L.data.resize(t.data.size() * sizeof(float));
        std::memcpy(L.data.data(), t.data.data(), L.data.size());
        out_layers.push_back(std::move(L));
        ++n_raw;
    };

    for (const LinearRef& ref : linears) {
        ++done;
        progress(done, total_layers);
        const STTensor& t = *ref.t;
        const int64_t out_f = t.shape[0];
        const int64_t in_f = t.shape.size() > 1 ? t.shape[1] : 1;

        // Fused qkv_proj -> q/k/v row split.
        if (ref.name.find("qkv_proj") != std::string::npos) {
            const int64_t q_dim = (int64_t)cfg.num_attention_heads * head_dim;
            const int64_t kv_dim = (int64_t)cfg.num_key_value_heads * head_dim;
            if (out_f != q_dim + 2 * kv_dim) {
                Logger::error("qkv_proj shape {}x{} does not match "
                              "q_dim+k_dim+v_dim ({}x{}+{})", out_f, in_f,
                              q_dim, kv_dim, kv_dim);
                return 1;
            }
            const char* parts[3] = {"q_proj", "k_proj", "v_proj"};
            const int64_t dims[3] = {q_dim, kv_dim, kv_dim};
            for (int p = 0; p < 3; p++) {
                OutLayer L;
                L.name = ref.name;
                const size_t pos = L.name.find("qkv_proj");
                L.name.replace(pos, 8, parts[p]);
                L.out_f = (int32_t)dims[p];
                L.in_f = (int32_t)in_f;
                const int64_t row_off = (p == 0) ? 0 : (p == 1) ? q_dim : q_dim + kv_dim;
                L.layer_type = 1;
                const int64_t nelem = dims[p] * in_f;
                L.data.resize((size_t)nelem * sizeof(float));
                std::memcpy(L.data.data(),
                            t.data.data() + row_off * in_f,
                            (size_t)nelem * sizeof(float));
                out_layers.push_back(std::move(L));
                ++n_raw;
            }
            continue;
        }

        // Quantized projection -> ALS multi-term container. Every Linear other
        // than lm_head carries one of the quantized suffixes (same set as the
        // Python QUANTIZED_LAYERS minus qkv_proj, handled above).
        if (contains_any(ref.name,
                         {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                          "up_proj", "down_proj"})) {
            std::vector<ALSTerm> terms =
                als_decompose(t.data.data(), (int)out_f, (int)in_f, num_terms);
            std::vector<uint8_t> blob =
                pack_als_block_terms(terms, (int)out_f, (int)in_f, 128);
            OutLayer L;
            L.name = ref.name;
            L.out_f = (int32_t)out_f;
            L.in_f = (int32_t)in_f;
            L.layer_type = 2;
            L.data = std::move(blob);
            out_layers.push_back(std::move(L));
            ++n_quantized;
            continue;
        }

        // Everything else (lm_head): RAW FP32.
        emit_raw(ref.name, t);
    }

    // ── 7. Q/K RMSNorm pseudo-layers (RAW FP32, shape [head_dim]).
    if (has_q_norm || has_k_norm) {
        const char* norm_attrs[2] = {"q_norm", "k_norm"};
        for (const char* attr : norm_attrs) {
            const std::string base = std::string("model.layers.");
            for (int32_t i = 0; i < cfg.num_hidden_layers; i++) {
                const std::string nm =
                    base + std::to_string(i) + ".self_attn." + attr;
                auto it = tensors.find(nm + ".weight");
                if (it == tensors.end()) continue;
                ++done;
                progress(done, total_layers);
                emit_raw(nm, it->second);
            }
        }
        Logger::info("Exported {} Qwen3 Q/K RMSNorm weights as RAW FP32",
                     n_qk_norm_total);
    }

    // ── 8. Write model_decomposed.bin.
    Logger::info("Writing model_decomposed.bin ({} layers)...", out_layers.size());
    const std::string dec_path = outdir + "/model_decomposed.bin";
    if (!write_decomposed(dec_path, out_layers)) return 1;

    // ── 9. Write model_extra.bin.
    Logger::info("Writing model_extra.bin...{}", "");
    if (!write_extra(outdir + "/model_extra.bin", cfg, tensors, err)) {
        Logger::error("model_extra.bin write failed: {}", err);
        return 1;
    }

    // ── 10. Summary.
    struct stat st;
    const long dec_size = (stat(dec_path.c_str(), &st) == 0) ? st.st_size : -1;
    Logger::info("ALS quantized layers: {}", n_quantized);
    Logger::info("RAW FP32 layers:      {}", n_raw);
    char mb_str[64];
    snprintf(mb_str, sizeof(mb_str), "%.1f",
             dec_size > 0 ? (double)dec_size / 1e6 : 0.0);
    Logger::info("model_decomposed.bin: {} MB", mb_str);
    progress(total_layers, total_layers);
    return 0;
#endif // TERLLAMA_HAVE_CURL
}

// ═══════════════════════════════════════════════════════════════════════════
// export_main — "convert" subcommand
// ═══════════════════════════════════════════════════════════════════════════
int export_main(int argc, char** argv) {
    std::string repo, outdir, format = "als";
    int terms = 12;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--model") {
            if (i + 1 >= argc) { Logger::error("--model requires an argument"); return 1; }
            repo = argv[++i];
        } else if (a == "--outdir" || a == "-o") {
            if (i + 1 >= argc) { Logger::error("--outdir requires an argument"); return 1; }
            outdir = argv[++i];
        } else if (a == "--terms") {
            if (i + 1 >= argc) { Logger::error("--terms requires an argument"); return 1; }
            try { terms = std::stoi(argv[++i]); }
            catch (...) { Logger::error("invalid --terms value"); return 1; }
        } else if (a == "--format" || a == "--fmt") {
            if (i + 1 >= argc) { Logger::error("--format requires an argument"); return 1; }
            format = argv[++i];
        } else if (a == "--rotate") {
            if (i + 1 >= argc) { Logger::error("--rotate requires an argument"); return 1; }
            ++i; // informational only (Python reference also ignores it)
        } else if (repo.empty()) {
            repo = a;
        } else {
            Logger::error("unexpected argument '{}'", a);
            return 1;
        }
    }
    if (repo.empty()) {
        Logger::error("Usage: {} convert --model <hf_repo> [--outdir <dir>] "
                      "[--terms N] [--format als|gguf]", argv[0]);
        return 1;
    }
    if (outdir.empty()) {
        const char* home = getenv("HOME");
        std::string slug = repo;
        for (char& c : slug)
            if (c == '/') c = '-';
        outdir = std::string(home ? home : "/root") + "/.terllama/models/" + slug;
    }
    return convert_model(repo, outdir, terms, format);
}

} // namespace terllama

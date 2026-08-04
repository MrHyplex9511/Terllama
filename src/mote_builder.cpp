/*
 * mote_builder.cpp — Build MoTE model from dense ternary model
 *
 * Converts a standard dense ternary FFN to Mixture of Ternary Experts.
 * ALS grouping: distribute bitplane terms round-robin across K experts.
 * Block grouping: replicate block-scaled terms with perturbed scales.
 *
 * Usage: terllama mote-build <input_model_dir> <output_path> --experts K --topk k
 */
#include "loader.h"
#include "model.h"
#include "core/logger.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>

// ═══════════════════════════════════════════════════════════════════════════
// ALS term distribution: round-robin across experts
// ═══════════════════════════════════════════════════════════════════════════
static void distribute_als_terms(const LayerData& src,
                                  int num_experts,
                                  std::vector<LayerData>& expert_layers) {
    expert_layers.resize(num_experts);
    for (int e = 0; e < num_experts; e++) {
        expert_layers[e].out_features = src.out_features;
        expert_layers[e].in_features = src.in_features;
        expert_layers[e].has_raw_weights = false;
        expert_layers[e].has_blocks = false;
    }

    // Distribute terms round-robin
    for (int t = 0; t < src.num_terms; t++) {
        int e = t % num_experts;
        const auto& st = src.terms[t];
        auto& et = expert_layers[e].terms.emplace_back();
        et.alpha_exp = st.alpha_exp;
        et.n_elements = st.n_elements;
        et.combined = st.combined;
    }

    // Update num_terms
    for (int e = 0; e < num_experts; e++) {
        expert_layers[e].num_terms = (int)expert_layers[e].terms.size();
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Block term distribution: replicate with scale perturbation
// ═══════════════════════════════════════════════════════════════════════════
static void distribute_block_terms(const LayerData& src,
                                   int num_experts,
                                   std::vector<LayerData>& expert_layers) {
    expert_layers.resize(num_experts);
    for (int e = 0; e < num_experts; e++) {
        expert_layers[e].out_features = src.out_features;
        expert_layers[e].in_features = src.in_features;
        expert_layers[e].has_blocks = true;
        expert_layers[e].block_qk = src.block_qk;
        expert_layers[e].has_raw_weights = false;

        // Replicate blocks with perturbed scales (single term set per expert)
        expert_layers[e].block_terms.resize(1);
        expert_layers[e].block_terms[0] = src.block_terms[0];
        for (size_t b = 0; b < expert_layers[e].block_terms[0].size(); b++) {
            // Perturb scale: multiply by (0.8 + 0.4 * e / (num_experts-1))
            float perturbation = 0.8f + 0.4f * (float)e / std::max(1, num_experts - 1);
            expert_layers[e].block_terms[0][b].scale =
                src.block_terms[0][b].scale * perturbation;
        }

        // Build combined[] bitplane for backward compat
        int words_per_row = (src.in_features + 15) / 16;
        size_t n_words = (size_t)src.out_features * words_per_row;
        BitplaneTerm term;
        term.alpha_exp = 0;
        term.n_elements = (size_t)src.out_features * src.in_features;
        term.combined.assign(n_words, 0);

        int qk = src.block_qk;
        int n_blocks = (src.in_features + qk - 1) / qk;
        std::vector<int8_t> decoded(qk);

        for (int row = 0; row < src.out_features; row++) {
            for (int b = 0; b < n_blocks; b++) {
                int block_idx = row * n_blocks + b;
                int block_start = b * qk;
                int block_end = std::min(block_start + qk, src.in_features);
                int block_size = block_end - block_start;

                decode_block_ternary(expert_layers[e].block_terms[0][block_idx].packed.data(),
                                     decoded.data(), qk);

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
        expert_layers[e].num_terms = 1;
        expert_layers[e].terms.push_back(term);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RAW FP32 distribution: replicate with Gaussian noise
// ═══════════════════════════════════════════════════════════════════════════
static void distribute_raw_weights(const LayerData& src,
                                    int num_experts,
                                    std::vector<LayerData>& expert_layers) {
    expert_layers.resize(num_experts);
    std::mt19937 rng(42);
    for (int e = 0; e < num_experts; e++) {
        expert_layers[e].out_features = src.out_features;
        expert_layers[e].in_features = src.in_features;
        expert_layers[e].has_raw_weights = true;
        expert_layers[e].has_blocks = false;
        expert_layers[e].raw_weights = src.raw_weights;

        // Add small noise for diversity
        if (num_experts > 1) {
            float noise_scale = 0.01f;
            std::normal_distribution<float> noise(0.0f, noise_scale);
            for (auto& w : expert_layers[e].raw_weights)
                w += noise(rng);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Distribute a LayerData across K experts based on its storage type
// ═══════════════════════════════════════════════════════════════════════════
static void distribute_layer(const LayerData& src,
                              int num_experts,
                              std::vector<LayerData>& experts) {
    if (src.has_raw_weights) {
        distribute_raw_weights(src, num_experts, experts);
    } else if (src.has_blocks) {
        distribute_block_terms(src, num_experts, experts);
    } else {
        distribute_als_terms(src, num_experts, experts);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN: mote-build command
// ═══════════════════════════════════════════════════════════════════════════
int cmd_mote_build(int argc, char** argv) {
    if (argc < 4) {
        Logger::error("Usage: {} mote-build <input_model_dir> <output_path> --experts K --topk k", argv[0] ? argv[0] : "terllama");
        Logger::error("  Converts a dense ternary model to MoTE format");
        Logger::error("  --experts K    Number of experts (default: 4)");
        Logger::error("  --topk k       Top-K experts per token (default: 1)");
        return 1;
    }

    std::string input_dir = argv[2];
    std::string output_path = argv[3];
    int num_experts = 4;
    int top_k = 1;

    for (int i = 4; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--experts" && i + 1 < argc) num_experts = std::stoi(argv[++i]);
        else if (a == "--topk" && i + 1 < argc) top_k = std::stoi(argv[++i]);
    }

    Logger::info("MoTE Build: input={} output={} experts={} top_k={}",
                 input_dir, output_path, num_experts, top_k);

    // ─── Load dense model ──────────────────────────────────────────────
    Logger::info("Loading dense model from {}...", input_dir);
    auto loaded = load_model_from(input_dir);
    auto& cfg = loaded.cfg;
    auto& layers = loaded.layers;

    int HS = cfg.hidden_size;
    int num_transformer_layers = cfg.num_hidden_layers;
    Logger::info("Model: {} layers, hidden={}, intermediate={}",
                 num_transformer_layers, HS, cfg.intermediate_size);

    // ─── Build MoTE layers ─────────────────────────────────────────────
    MoTEConfig mote_cfg;
    mote_cfg.num_experts = num_experts;
    mote_cfg.top_k = top_k;
    mote_cfg.use_shared_expert = true;

    std::vector<MoTELayerData> mote_layers(num_transformer_layers);

    for (int i = 0; i < num_transformer_layers; i++) {
        std::string prefix = "model.layers." + std::to_string(i);

        auto& ml = mote_layers[i];
        ml.num_experts = num_experts;
        ml.top_k = top_k;
        ml.is_mote = true;
        ml.router_scale = 1.0f;

        // Find FFN layers in loaded model
        int idx_g = find_layer_index(layers, prefix + ".mlp.gate_proj");
        int idx_u = find_layer_index(layers, prefix + ".mlp.up_proj");
        int idx_d = find_layer_index(layers, prefix + ".mlp.down_proj");

        // Shared expert = original FFN weights
        ml.gate_proj = layers[idx_g];
        ml.gate_proj.name = prefix + ".mote.gate_proj";
        ml.up_proj = layers[idx_u];
        ml.up_proj.name = prefix + ".mote.up_proj";
        ml.down_proj = layers[idx_d];
        ml.down_proj.name = prefix + ".mote.down_proj";

        // Distribute across experts
        distribute_layer(layers[idx_g], num_experts, ml.expert_gate);
        distribute_layer(layers[idx_u], num_experts, ml.expert_up);
        distribute_layer(layers[idx_d], num_experts, ml.expert_down);

        // Set expert names
        for (int e = 0; e < num_experts; e++) {
            std::string es = prefix + ".mote.expert." + std::to_string(e);
            ml.expert_gate[e].name = es + ".gate_proj";
            ml.expert_up[e].name = es + ".up_proj";
            ml.expert_down[e].name = es + ".down_proj";
        }

        // Initialize router weights: uniform distribution
        ml.router_weight.resize((size_t)HS * num_experts);
        for (int h = 0; h < HS; h++) {
            for (int k = 0; k < num_experts; k++) {
                // Uniform initial weights + small noise for differentiation
                ml.router_weight[h * num_experts + k] = 0.01f * (float)(k + 1) / num_experts;
            }
        }

        Logger::debug("  Layer {}: {} experts x {} ALS/block terms",
                      i, num_experts, ml.expert_gate[0].num_terms);
    }

    // ─── Save MoTE model ───────────────────────────────────────────────
    Logger::info("Saving MoTE model to {}...", output_path);
    save_mote_model(output_path, mote_cfg, mote_layers);

    // ─── Summary ───────────────────────────────────────────────────────
    Logger::info("═══════════════════════════════════════");
    Logger::info("MoTE Build Complete");
    Logger::info("  Input:       {}", input_dir);
    Logger::info("  Output:      {}", output_path);
    Logger::info("  Layers:      {}", num_transformer_layers);
    Logger::info("  Experts:     {}", num_experts);
    Logger::info("  Top-K:       {}", top_k);
    Logger::info("  Shared exp:  yes");
    Logger::info("  Hidden:      {}", HS);
    Logger::info("  Intermediate: {}", cfg.intermediate_size);
    Logger::info("═══════════════════════════════════════");

    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// mote-list command: list MoTE layers in a model file
// ═══════════════════════════════════════════════════════════════════════════
int cmd_mote_list(int argc, char** argv) {
    if (argc < 3) {
        Logger::error("Usage: {} mote-list <model_path>", argv[0] ? argv[0] : "terllama");
        return 1;
    }

    std::string path = argv[2];

    if (!is_mote_file(path)) {
        Logger::error("Not a MoTE file: {}", path);
        return 1;
    }

    MoTEConfig cfg = peek_mote_config(path);
    auto mote_layers = load_mote_layers(path);

    Logger::info("═══════════════════════════════════════");
    Logger::info("MoTE Model: {}", path);
    Logger::info("  Experts:     {}", cfg.num_experts);
    Logger::info("  Top-K:       {}", cfg.top_k);
    Logger::info("  Shared exp:  {}", cfg.use_shared_expert ? "yes" : "no");
    Logger::info("  Layers:      {}", mote_layers.size());
    Logger::info("───────────────────────────────────────");

    for (int i = 0; i < (int)mote_layers.size() && i < 5; i++) {
        auto& ml = mote_layers[i];
        std::string name = ml.gate_proj.name;
        // Strip to layer prefix
        auto p = name.rfind("mote");
        if (p != std::string::npos) name = name.substr(0, p + 4);
        else name = "layer." + std::to_string(i);

        Logger::info("  Layer {}: {} experts x top-{} | shared: {}x{} → {}",
                     i, ml.num_experts, ml.top_k,
                     ml.gate_proj.in_features, ml.gate_proj.out_features,
                     ml.down_proj.out_features);
        Logger::info("    Router: {}x{} = {} FP32 weights",
                     ml.router_weight.size() / ml.num_experts, ml.num_experts,
                     ml.router_weight.size());
        Logger::info("    Experts: {} {} {} ({} terms each)",
                     ml.expert_gate.size(),
                     ml.expert_gate[0].has_blocks ? "BLK" : (ml.expert_gate[0].has_raw_weights ? "RAW" : "ALS"),
                     ml.expert_gate[0].num_terms > 0 ? ("terms=" + std::to_string(ml.expert_gate[0].num_terms)) : "",
                     ml.expert_gate[0].num_terms);
    }

    if (mote_layers.size() > 5) {
        Logger::info("  ... and {} more layers", mote_layers.size() - 5);
    }

    Logger::info("═══════════════════════════════════════");
    return 0;
}

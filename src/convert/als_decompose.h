// ALS ternary decomposition + packing (Track B).
//
// C++ port of scripts/export_ternary_model_bitnet.py: _greedy_terms,
// als_decompose (Phase 1 greedy + Phase 2 refinement + joint per-block
// least-squares scales) and pack_als_block / pack_als_block_terms.
// Numerics are matched to the Python reference (bit-for-bit where the
// reference is deterministic; see als_decompose.cpp for the float64
// accumulation notes).
#pragma once

#include <cstdint>
#include <vector>

namespace terllama {

// One ALS term: per-block float32 scales (n_blocks) + full ternary matrix
// (out_f * in_f, row-major, values -1/0/1).
struct ALSTerm {
    std::vector<float> scales;    // n_blocks per-block least-squares scales
    std::vector<int8_t> ternary;  // out_f * in_f, values -1/0/1
};

// Full decomposition: returns the ALS terms (scales + ternary) exactly as the
// Python als_decompose does — per-term scales from the joint per-block
// ridge-regularized fp64 least-squares solve, ternary from greedy + 5
// refinement iterations. max_iter and qk match the script defaults.
std::vector<ALSTerm> als_decompose(const float* W, int out_f, int in_f,
                                   int num_terms, int max_iter = 5, int qk = 128);

// Per-term scale lists only: scales[term][block] (float32, cast from the fp64
// ridge-regularized least-squares solve). Same decomposition as als_decompose.
std::vector<std::vector<float>> als_decompose_scales(const float* W, int out_f,
                                                     int in_f, int num_terms);

// Pack a list of ALS terms into the layer_type=2 container blob:
//   [num_terms:u32][term0_len:u32][term0_data]...[termN_len:u32][termN_data]
std::vector<uint8_t> pack_als_block_terms(const std::vector<ALSTerm>& terms,
                                          int out_f, int in_f, int qk);

// Pack one ALS term into the per-row block blob:
//   per row: [32 code bytes + float32 scale(LE)] x n_blocks
// Codes are 2-bit/weight, 4 vals/byte MSB-first, mapping {-1,0,+1}->{0,1,2};
// tail blocks are zero-padded with code 1 (ternary 0).
std::vector<uint8_t> pack_als_block(const ALSTerm& term, int out_f, int in_f,
                                    int qk);

}  // namespace terllama

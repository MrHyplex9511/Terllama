// test_kernels.cpp — Catch2 tests for ternary kernel correctness
//
// Compares AVX2/NEON outputs against scalar reference on small matrices.
// Tests both bitplane (combined[]) and ALS block-scaled kernel paths.
#include "model.h"
#include "kernel_decl.h"
#include "catch_amalgamated.hpp"

#include <vector>
#include <cstdint>
#include <random>
#include <cmath>

// ─── Helpers ──────────────────────────────────────────────────────────────

static LayerData make_small_layer(int out_f, int in_f, int num_terms) {
    LayerData ld;
    ld.name = "test";
    ld.out_features = out_f;
    ld.in_features = in_f;
    ld.num_terms = num_terms;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 3);

    int words_per_row = (in_f + 15) / 16;

    for (int t = 0; t < num_terms; t++) {
        BitplaneTerm term;
        term.alpha_exp = -(t + 1);
        term.n_elements = (size_t)out_f * in_f;

        size_t n_words = (size_t)out_f * words_per_row;
        term.combined.assign(n_words, 0);

        for (int i = 0; i < out_f; i++) {
            for (int j = 0; j < in_f; j++) {
                int v = dist(rng);
                int word = j / 16;
                int bit = j % 16;
                size_t abs_word = (size_t)i * words_per_row + word;
                if (v == 1)
                    term.combined[abs_word] |= (1 << (bit + 16));       // +1
                else if (v == 2)
                    term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit); // -1
            }
        }
        ld.terms.push_back(std::move(term));
    }
    return ld;
}

static std::vector<float> make_input(int n) {
    std::vector<float> v(n);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n; i++) v[i] = dist(rng);
    return v;
}

static float max_diff(const float* a, const float* b, int n) {
    float d = 0;
    for (int i = 0; i < n; i++) d = std::max(d, std::abs(a[i] - b[i]));
    return d;
}

// ─── Tests ────────────────────────────────────────────────────────────────

TEST_CASE("Scalar kernel produces deterministic output", "[kernel][scalar]") {
    auto layer = make_small_layer(4, 8, 2);
    auto input = make_input(8);
    std::vector<float> out1(4), out2(4);

    const uint32_t* td[32]; int ae[32];
    int n = extract_term_data(layer, td, ae, 32);

    ternary_mul_scalar(td, ae, n, 4, 8, input.data(), out1.data());
    ternary_mul_scalar(td, ae, n, 4, 8, input.data(), out2.data());

    REQUIRE(max_diff(out1.data(), out2.data(), 4) == 0.0f);
}

TEST_CASE("Scalar kernel handles zero and negative values", "[kernel][scalar]") {
    auto layer = make_small_layer(2, 16, 1);
    std::vector<float> input(16, 0.0f);
    input[0] = 1.0f;
    input[1] = -1.0f;
    input[2] = 2.5f;

    std::vector<float> output(2);
    const uint32_t* td[32]; int ae[32];
    int n = extract_term_data(layer, td, ae, 32);
    ternary_mul_scalar(td, ae, n, 2, 16, input.data(), output.data());

    for (int i = 0; i < 2; i++) {
        REQUIRE_FALSE(std::isnan(output[i]));
        REQUIRE_FALSE(std::isinf(output[i]));
    }
}

TEST_CASE("extract_term_data returns 0 for empty layer", "[kernel]") {
    LayerData empty;
    empty.name = "empty";
    empty.out_features = 4;
    empty.in_features = 8;
    empty.num_terms = 0;

    const uint32_t* td[32] = {nullptr}; int ae[32] = {0};
    int n = extract_term_data(empty, td, ae, 32);
    REQUIRE(n == 0);
}

#if defined(__x86_64__) || defined(_M_X64)
TEST_CASE("AVX2 kernel matches scalar reference (small matrix)", "[kernel][avx2]") {
    auto layer = make_small_layer(8, 32, 3);
    auto input = make_input(32);

    std::vector<float> ref(8), test(8);
    const uint32_t* td[32]; int ae[32];
    int n = extract_term_data(layer, td, ae, 32);

    ternary_mul_scalar(td, ae, n, 8, 32, input.data(), ref.data());

    extern __attribute__((weak)) void ternary_mul_avx2(
        const uint32_t* const*, const int*, int, int, int,
        const float*, float*);

    if (!ternary_mul_avx2) {
        WARN("ternary_mul_avx2 not linked — skipping");
        return;
    }

    ternary_mul_avx2(td, ae, n, 8, 32, input.data(), test.data());

    float err = max_diff(ref.data(), test.data(), 8);
    CHECK(err < 1e-4f);
}
#endif

#if defined(__aarch64__)
TEST_CASE("NEON kernel matches scalar reference (small matrix)", "[kernel][neon]") {
    auto layer = make_small_layer(8, 32, 3);
    auto input = make_input(32);

    std::vector<float> ref(8), test(8);
    const uint32_t* td[32]; int ae[32];
    int n = extract_term_data(layer, td, ae, 32);

    ternary_mul_scalar(td, ae, n, 8, 32, input.data(), ref.data());

    extern __attribute__((weak)) void ternary_mul_neon(
        const uint32_t* const*, const int*, int, int, int,
        const float*, float*);

    if (!ternary_mul_neon) {
        WARN("ternary_mul_neon not linked — skipping");
        return;
    }

    ternary_mul_neon(td, ae, n, 8, 32, input.data(), test.data());

    float err = max_diff(ref.data(), test.data(), 8);
    CHECK(err < 1e-4f);
}
#endif

TEST_CASE("ALS block kernel produces NaN-free output", "[kernel][als]") {
    // Build a minimal ALS block-scaled layer (128-element blocks, codes:
    // {0=-1, 1=0, 2=+1} packed 4-per-byte, MSB first).
    constexpr int in_f = 128;   // one 128-element block per row
    constexpr int out_f = 2;
    LayerData ld;
    ld.name = "test_als";
    ld.out_features = out_f;
    ld.in_features = in_f;
    ld.num_terms = 1;
    ld.has_blocks = true;
    ld.block_qk = 128;
    ld.terms.clear();

    // term 0: one block per output row
    std::vector<BlockTerm> blocks(out_f);
    for (auto& blk : blocks) {
        blk.packed.assign(in_f / 4, 0);
        blk.scale = 0.5f;
        // pack alternating +1/-1 patterns: v=+1 → 2, v=-1 → 0
        for (int j = 0; j < in_f; j++) {
            int code = (j % 2 == 0) ? 2 : 0;
            blk.packed[j / 4] |= (uint8_t)code << (6 - 2 * (j % 4));
        }
    }
    ld.block_terms.push_back(std::move(blocks));

    auto input = make_input(in_f);
    std::vector<float> output(out_f, -999.0f);

    // Dispatch through the ALS block path
    ternary_linear_blocks(ld, input.data(), output.data(), CPUArch::X86_64_SCALAR);

    for (int i = 0; i < out_f; i++) {
        REQUIRE_FALSE(std::isnan(output[i]));
        REQUIRE_FALSE(std::isinf(output[i]));
    }
}

// test_inference.cpp — Catch2 tests for inference operations
//
// Tests sample_argmax, sample_multinomial, rms_norm, build_rope_cache.
#include "catch_amalgamated.hpp"

#include "model.h"
#include "inference.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

// ─── sample_argmax ────────────────────────────────────────────────────────

TEST_CASE("sample_argmax returns index of largest value", "[sample][argmax]") {
    float logits[] = {1.0f, 5.0f, 2.0f, 0.5f, 3.0f};
    int idx = sample_argmax(logits, 5);
    REQUIRE(idx == 1);  // 5.0 is at index 1
}

TEST_CASE("sample_argmax handles negative values", "[sample][argmax]") {
    float logits[] = {-10.0f, -1.0f, -100.0f};
    int idx = sample_argmax(logits, 3);
    REQUIRE(idx == 1);  // -1.0 is largest
}

TEST_CASE("sample_argmax returns first index on tie", "[sample][argmax]") {
    float logits[] = {3.0f, 1.0f, 3.0f, 2.0f};
    int idx = sample_argmax(logits, 4);
    REQUIRE(idx == 0);  // first occurrence of 3.0
}

TEST_CASE("sample_argmax single element", "[sample][argmax]") {
    float logits[] = {42.0f};
    int idx = sample_argmax(logits, 1);
    REQUIRE(idx == 0);
}

// ─── sample_multinomial ───────────────────────────────────────────────────

TEST_CASE("sample_multinomial with temperature=0 approximates argmax", "[sample][multinomial]") {
    float logits[] = {1.0f, 10.0f, 2.0f};
    srand(42);
    int idx = sample_multinomial(logits, 3, 0.001f);
    REQUIRE(idx == 1);  // 10.0 dominates → always index 1
}

TEST_CASE("sample_multinomial respects repetition penalty", "[sample][multinomial]") {
    // Token 0 has middling logit; tokens 1/2 dominate.
    // Heavy penalty on token 0 makes it negligible.
    float logits[] = {1.0f, 10.0f, 10.0f};
    srand(42);
    std::vector<int> recent = {0};
    int idx = sample_multinomial(logits, 3, 1.0f, recent, 10.0f);
    REQUIRE(idx != 0);
}

TEST_CASE("sample_multinomial with uniform distribution", "[sample][multinomial]") {
    float logits[] = {0.0f, 0.0f, 0.0f};
    srand(42);
    int idx = sample_multinomial(logits, 3, 1.0f);
    REQUIRE(idx >= 0);
    REQUIRE(idx < 3);
}

// ─── rms_norm ─────────────────────────────────────────────────────────────

TEST_CASE("rms_norm preserves sign of input", "[norm][rms]") {
    int n = 4;
    float x[] = {2.0f, -3.0f, 1.0f, -1.0f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;

    rms_norm(x, weight, n, eps);

    REQUIRE(x[0] > 0);
    REQUIRE(x[1] < 0);
    REQUIRE(x[2] > 0);
    REQUIRE(x[3] < 0);
}

TEST_CASE("rms_norm produces unit RMS", "[norm][rms]") {
    int n = 8;
    float x[] = {3.0f, -1.0f, 2.0f, -4.0f, 0.5f, -0.5f, 1.5f, -2.5f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;

    rms_norm(x, weight, n, eps);

    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float rms = std::sqrt(ss / n);
    REQUIRE(std::abs(rms - 1.0f) < 1e-5f);
}

TEST_CASE("rms_norm respects weight scaling", "[norm][rms]") {
    int n = 4;
    float x[] = {2.0f, 2.0f, 2.0f, 2.0f};
    float weight[] = {0.5f, 1.0f, 2.0f, 4.0f};
    float eps = 1e-6f;

    rms_norm(x, weight, n, eps);

    // After norm: x[i] = weight[i] * input[i] / rms
    // All inputs equal, so normed values should be proportional to weights
    float ratio = x[1] / x[0];
    REQUIRE(std::abs(ratio - 2.0f) < 1e-5f);  // weight[1]=1.0, weight[0]=0.5
    ratio = x[2] / x[0];
    REQUIRE(std::abs(ratio - 4.0f) < 1e-5f);
    ratio = x[3] / x[0];
    REQUIRE(std::abs(ratio - 8.0f) < 1e-5f);
}

TEST_CASE("rms_norm handles zero input gracefully", "[norm][rms]") {
    int n = 4;
    float x[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float weight[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float eps = 1e-6f;

    rms_norm(x, weight, n, eps);

    // With eps, RMS = sqrt(0/n + eps) = sqrt(eps), so x[i] = w[i] * 0 / sqrt(eps) = 0
    for (int i = 0; i < n; i++) {
        REQUIRE(x[i] == 0.0f);
    }
}

TEST_CASE("rms_norm single element", "[norm][rms]") {
    int n = 1;
    float x[] = {5.0f};
    float weight[] = {2.0f};
    float eps = 1e-6f;

    rms_norm(x, weight, n, eps);
    // RMS = sqrt(25/1 + eps) ≈ 5.0, so x[0] = 2.0 * 5.0 / 5.0 = 2.0
    REQUIRE(std::abs(x[0] - 2.0f) < 1e-5f);
}

// ─── build_rope_cache ─────────────────────────────────────────────────────

TEST_CASE("build_rope_cache produces non-zero sin/cos arrays", "[rope]") {
    auto cache = build_rope_cache(10, 64, 10000.0f);

    REQUIRE(cache.max_seq_len == 10);
    REQUIRE(cache.head_dim == 64);
    REQUIRE(cache.sin.size() == (size_t)10 * 32);
    REQUIRE(cache.cos.size() == (size_t)10 * 32);
}

TEST_CASE("build_rope_cache sin and cos satisfy sin^2 + cos^2 = 1", "[rope]") {
    auto cache = build_rope_cache(4, 32, 10000.0f);

    int hd2 = 16;
    for (int pos = 0; pos < 4; pos++) {
        for (int j = 0; j < hd2; j++) {
            float s = cache.sin[pos * hd2 + j];
            float c = cache.cos[pos * hd2 + j];
            float ssc = s*s + c*c;
            REQUIRE(std::abs(ssc - 1.0f) < 1e-5f);
        }
    }
}

TEST_CASE("build_rope_cache zero position gives cos=1 sin=0", "[rope]") {
    auto cache = build_rope_cache(1, 32, 10000.0f);

    int hd2 = 16;
    for (int j = 0; j < hd2; j++) {
        REQUIRE(std::abs(cache.sin[j] - 0.0f) < 1e-6f);
        REQUIRE(std::abs(cache.cos[j] - 1.0f) < 1e-6f);
    }
}

TEST_CASE("build_rope_cache larger theta gives slower frequencies", "[rope]") {
    auto cache_low  = build_rope_cache(4, 32, 1000.0f);
    auto cache_high = build_rope_cache(4, 32, 10000.0f);

    int hd2 = 16;
    // Sin is monotonic increasing on [0, π/2]. At position 1, all angles
    // are ≤ 1.0 rad (< π/2), so sin(angle) tracks frequency monotonically.
    int pos = 1;
    for (int j = 1; j < hd2; j++) {  // j=0: freq=1.0 for both → identical
        float sl = std::abs(cache_low.sin[pos * hd2 + j]);
        float sh = std::abs(cache_high.sin[pos * hd2 + j]);
        // Lower theta → higher freq → larger angle → larger sin
        REQUIRE(sl >= sh);
    }
}

TEST_CASE("RoPE sin/cos indices are within bounds for large sequences", "[rope]") {
    int max_seq = 2048;
    int head_dim = 128;
    auto cache = build_rope_cache(max_seq, head_dim, 10000.0f);

    REQUIRE(cache.sin.size() == (size_t)max_seq * (head_dim / 2));
    REQUIRE(cache.cos.size() == (size_t)max_seq * (head_dim / 2));

    // Spot-check last position
    int hd2 = head_dim / 2;
    int last_idx = (max_seq - 1) * hd2;
    float s = cache.sin[last_idx + hd2 - 1];
    float c = cache.cos[last_idx + hd2 - 1];
    float ssc = s*s + c*c;
    REQUIRE(std::abs(ssc - 1.0f) < 1e-5f);
}

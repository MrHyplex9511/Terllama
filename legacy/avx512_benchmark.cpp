/*
 * avx512_benchmark.cpp - Synthetic benchmark for AVX-512 ternary kernel.
 * Generates random weights/inputs, tests correctness + throughput.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -mavx512f -mavx512dq -fopenmp \
 *       -o avx512_benchmark avx512_benchmark.cpp -lm
 *
 * Run:
 *   ./avx512_benchmark
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <omp.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

// ─── Bitplane format ───────────────────────────────────────────────────────
struct BitplaneTerm {
    int32_t alpha_exp;
    std::vector<uint16_t> nz;    // non-zero bitplane
    std::vector<uint16_t> neg;   // negative bitplane
    size_t n_elements;
};

struct LayerData {
    int32_t out_features, in_features;
    int32_t num_terms;
    std::vector<BitplaneTerm> terms;
};

// ─── Generate synthetic ternary layer ──────────────────────────────────────
LayerData generate_layer(int out_f, int in_f, int N, float sparsity=0.5f, unsigned seed=42) {
    LayerData ld;
    ld.out_features = out_f;
    ld.in_features = in_f;
    ld.num_terms = N;
    ld.terms.resize(N);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // Generate a random matrix W, decompose into ternary terms
    // Each term: random ternary
    for (int t = 0; t < N; t++) {
        size_t n_elements = (size_t)out_f * in_f;
        size_t n_words = (n_elements + 15) / 16;
        uint16_t alpha_exp = t;  // decreasing significance
        ld.terms[t].alpha_exp = alpha_exp;
        ld.terms[t].n_elements = n_elements;
        ld.terms[t].nz.resize(n_words, 0);
        ld.terms[t].neg.resize(n_words, 0);

        // Fill with random ternary {-1,0,+1} at given sparsity
        for (size_t pos = 0; pos < n_elements; pos++) {
            float v = dist(rng);
            int8_t tv = 0;
            if (v > sparsity) tv = 1;
            else if (v < -sparsity) tv = -1;
            if (tv != 0) {
                int word = pos / 16;
                int bit = pos % 16;
                ld.terms[t].nz[word] |= (1 << bit);
                if (tv == -1) ld.terms[t].neg[word] |= (1 << bit);
            }
        }
    }
    return ld;
}

// ─── Input vector ─────────────────────────────────────────────────────────
std::vector<float> generate_input(int n, unsigned seed=123) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(n);
    for (int i = 0; i < n; i++) x[i] = dist(rng);
    return x;
}

// ═══════════════════════════════════════════════════════════════════════════
// NAIVE KERNEL (reference)
// ═══════════════════════════════════════════════════════════════════════════
void ternary_linear_naive(const LayerData& layer, const float* input, float* output) {
    const int32_t out_f = layer.out_features;
    const int32_t in_f = layer.in_features;
    std::fill(output, output + out_f, 0.0f);

    for (int t = 0; t < layer.num_terms; t++) {
        int32_t ae = layer.terms[t].alpha_exp;
        if (ae == -128) continue;
        const auto& nz = layer.terms[t].nz;
        const auto& neg = layer.terms[t].neg;
        int words_per_row = (in_f + 15) / 16;

        for (int i = 0; i < out_f; i++) {
            float sum = 0.0f;
            size_t word_base = (size_t)i * words_per_row;
            for (int w = 0; w < words_per_row; w++) {
                uint16_t nzw = nz[word_base + w];
                uint16_t negw = neg[word_base + w];
                for (int k = 0; k < 16; k++) {
                    int pos = w * 16 + k;
                    if (pos >= in_f) break;
                    if ((nzw >> k) & 1) {
                        sum += ((negw >> k) & 1) ? -input[pos] : input[pos];
                    }
                }
            }
            output[i] += std::ldexp(sum, ae);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AVX-512 KERNEL
// ═══════════════════════════════════════════════════════════════════════════
#ifdef __AVX512F__
void ternary_linear_avx512(const LayerData& layer, const float* input, float* output) {
    const int32_t out_f = layer.out_features;
    const int32_t in_f = layer.in_features;
    std::fill(output, output + out_f, 0.0f);

    for (int t = 0; t < layer.num_terms; t++) {
        int32_t ae = layer.terms[t].alpha_exp;
        if (ae == -128) continue;
        const auto& nz = layer.terms[t].nz;
        const auto& neg = layer.terms[t].neg;
        int words_per_row = (in_f + 15) / 16;
        int rem = in_f % 16;
        int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
        uint16_t tail_mask = rem > 0 ? (uint16_t)((1 << rem) - 1) : 0;

        #pragma omp parallel for
        for (int i = 0; i < out_f; i++) {
            __m512 sum = _mm512_setzero_ps();
            size_t word_base = (size_t)i * words_per_row;

            for (int w = 0; w < full_words; w++) {
                __m512 v = _mm512_loadu_ps(&input[w * 16]);
                uint16_t nzw = nz[word_base + w];
                uint16_t negw = neg[word_base + w];
                __mmask16 mask_add = nzw & ~negw;
                __mmask16 mask_sub = nzw & negw;
                sum = _mm512_mask_add_ps(sum, mask_add, sum, v);
                sum = _mm512_mask_sub_ps(sum, mask_sub, sum, v);
            }

            if (rem > 0) {
                int w = full_words;
                __m512 v = _mm512_loadu_ps(&input[w * 16]);
                uint16_t nzw = nz[word_base + w] & tail_mask;
                uint16_t negw = neg[word_base + w] & tail_mask;
                __mmask16 mask_add = nzw & ~negw;
                __mmask16 mask_sub = nzw & negw;
                sum = _mm512_mask_add_ps(sum, mask_add, sum, v);
                sum = _mm512_mask_sub_ps(sum, mask_sub, sum, v);
            }

            float s = _mm512_reduce_add_ps(sum);
            output[i] += std::ldexp(s, ae);
        }
    }
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// MSE between two arrays
// ═══════════════════════════════════════════════════════════════════════════
double compute_mse(const float* a, const float* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        sum += d * d;
    }
    return sum / n;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
int main() {
    std::cout << "═══ AVX-512 Ternary Kernel Benchmark ═══" << std::endl;

#ifdef __AVX512F__
    std::cout << "✅ AVX-512: available" << std::endl;
#else
    std::cout << "❌ AVX-512: NOT available" << std::endl;
    return 1;
#endif

    // Test configurations: (name, out_f, in_f, num_terms)
    struct TestCase {
        std::string name; int out_f; int in_f; int N;
    };
    std::vector<TestCase> tests = {
        {"q_proj  [2048x2048]",   2048, 2048, 12},
        {"o_proj  [2048x2048]",   2048, 2048, 12},
        {"k_proj  [256x2048]",     256, 2048, 12},
        {"v_proj  [256x2048]",     256, 2048, 12},
        {"gate_proj [5632x2048]", 5632, 2048, 12},
        {"down_proj [2048x5632]", 2048, 5632, 12},
        {"lm_head [576x96000]",    576, 96000, 15},
        {"large [4096x4096]",     4096, 4096, 12},
        {"huge [1024x65536]",     1024, 65536, 10},
    };

    // Warmup
    std::cout << "\nWarming up..." << std::endl;
    {
        auto wl = generate_layer(128, 128, 5);
        auto wi = generate_input(128);
        std::vector<float> wo_avx(128), wo_naive(128);
        ternary_linear_avx512(wl, wi.data(), wo_avx.data());
        ternary_linear_naive(wl, wi.data(), wo_naive.data());
        std::cout << "  Warmup OK, MSE=" << compute_mse(wo_avx.data(), wo_naive.data(), 128) << std::endl;
    }

    std::cout << "\n─── Results ──────────────────────────────────" << std::endl;
    std::cout << std::left << std::setw(28) << "Layer"
              << std::right
              << std::setw(10) << "AVX-512"
              << std::setw(10) << "Naive"
              << std::setw(10) << "Speedup"
              << std::setw(14) << "MSE"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    for (const auto& tc : tests) {
        auto layer = generate_layer(tc.out_f, tc.in_f, tc.N, 0.5f, 42);
        auto input = generate_input(tc.in_f);
        std::vector<float> out_avx(tc.out_f, 0), out_naive(tc.out_f, 0);

        // Warmup runs
        ternary_linear_avx512(layer, input.data(), out_avx.data());
        ternary_linear_naive(layer, input.data(), out_naive.data());

        // Benchmark AVX-512 (10 runs)
        auto t0 = std::chrono::high_resolution_clock::now();
        int reps = std::max(1, 100000 / (tc.out_f * tc.in_f / 1000));
        reps = std::min(reps, 100);
        for (int r = 0; r < reps; r++) {
            std::fill(out_avx.begin(), out_avx.end(), 0.0f);
            ternary_linear_avx512(layer, input.data(), out_avx.data());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt_avx = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

        // Benchmark naive
        t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; r++) {
            std::fill(out_naive.begin(), out_naive.end(), 0.0f);
            ternary_linear_naive(layer, input.data(), out_naive.data());
        }
        t1 = std::chrono::high_resolution_clock::now();
        double dt_naive = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;

        double speedup = dt_naive / dt_avx;
        double mse = compute_mse(out_avx.data(), out_naive.data(), tc.out_f);

        std::cout << std::left << std::setw(28) << tc.name
                  << std::right
                  << std::setw(9) << std::fixed << std::setprecision(2) << dt_avx << "ms"
                  << std::setw(9) << std::fixed << std::setprecision(2) << dt_naive << "ms"
                  << std::setw(9) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(14) << std::scientific << std::setprecision(2) << mse
                  << std::endl;
    }

    // ═══ Model-level estimate ═══
    std::cout << "\n─── Estimated throughput (1.1B model) ────" << std::endl;
    // Rough model: 22 layers × (q+k+v+o+gate+up+down) + lm_head = 155 layers
    // Each layer is ~7M ternary MAC operations per term
    // Total: ~155 × 7M × 12 = ~13B ternary MACs per forward pass
    // Using lm_head time as bottleneck limiter
    
    // Count total ternary MACs for 1.1B model
    struct ModelLayer {
        std::string name; int out_f; int in_f; int N; int count;
    };
    std::vector<ModelLayer> model = {
        {"q_proj",   2048, 2048, 12, 22},
        {"k_proj",   256,  2048, 12, 22},
        {"v_proj",   256,  2048, 12, 22},
        {"o_proj",   2048, 2048, 12, 22},
        {"gate_proj",5632, 2048, 12, 22},
        {"up_proj",  5632, 2048, 12, 22},
        {"down_proj",2048, 5632, 12, 22},
        {"lm_head",  32000,2048, 12, 1},
    };
    
    double total_ms_avx = 0, total_ms_naive = 0;
    for (const auto& ml : model) {
        auto layer = generate_layer(ml.out_f, ml.in_f, ml.N, 0.5f, 42);
        auto input = generate_input(ml.in_f);
        std::vector<float> out(ml.out_f);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        ternary_linear_avx512(layer, input.data(), out.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms_avx += dt * ml.count;
        
        t0 = std::chrono::high_resolution_clock::now();
        ternary_linear_naive(layer, input.data(), out.data());
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms_naive += dt * ml.count;
    }
    
    // Overhead: RMSNorm, RoPE, softmax, KV cache (~20% of ML time)
    double overhead = total_ms_avx * 0.3;
    double ms_per_token = total_ms_avx + overhead;
    double ms_per_token_naive = total_ms_naive + total_ms_naive * 0.3;
    
    std::cout << "  Matrix ops (AVX-512): " << std::fixed << std::setprecision(1) 
              << total_ms_avx << " ms/token" << std::endl;
    std::cout << "  With overhead:         " << ms_per_token << " ms/token" << std::endl;
    std::cout << "  Tok/s (AVX-512):       " << (1000.0 / ms_per_token) << std::endl;
    std::cout << "  Tok/s (naive):         " << (1000.0 / ms_per_token_naive) << std::endl;
    std::cout << "  Speedup:               " << (ms_per_token_naive / ms_per_token) << "×" << std::endl;

    // OMP threads
    #pragma omp parallel
    {
        #pragma omp master
        std::cout << "\n  OpenMP threads: " << omp_get_num_threads() << std::endl;
    }

    std::cout << "\n═══ Done ═══" << std::endl;
    return 0;
}

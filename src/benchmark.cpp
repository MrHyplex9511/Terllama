/*
 * benchmark.cpp — Kernel benchmark for Terllama
 *
 * Test kernels on representative matrix shapes.
 * Report throughput, MSE, dispatch.
 *
 * Build:
 *   g++ -std=c++17 -O3 -march=native -mavx512f -mavx512dq -fopenmp \
 *       -o benchmark benchmark.cpp -lm
 *
 * Run:
 *   ./benchmark              # auto-detect + test all supported kernels
 *   TERLLAMA_ARCH=avx2 ./benchmark  # force a specific kernel
 */
#include "model.h"
#include "kernel_decl.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <random>
#ifdef _OPENMP
#include <omp.h>
#endif

// ═══════════════════════════════════════════════════════════════════════════
// SYNTHETIC LAYER GENERATOR
// ═══════════════════════════════════════════════════════════════════════════
// LayerData with random ternary weights for benchmarking.
LayerData make_synthetic_layer(const std::string& name,
                                int out_f, int in_f, int num_terms) {
    LayerData ld;
    ld.name = name;
    ld.out_features = out_f;
    ld.in_features = in_f;
    ld.num_terms = num_terms;

    // Ternary distribution: ~50% zeros, ~25% +1, ~25% -1
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, 3);

    int words_per_row = (in_f + 15) / 16;
    size_t n_words = (size_t)out_f * words_per_row;

    for (int t = 0; t < num_terms; t++) {
        BitplaneTerm term;
        term.alpha_exp = -(t + 1);  // decreasing alpha
        term.combined.assign(n_words, 0);
        term.n_elements = (size_t)out_f * in_f;

        for (int i = 0; i < out_f; i++) {
            for (int j = 0; j < in_f; j++) {
                int v = dist(rng);
                int word = j / 16;
                int bit = j % 16;
                size_t abs_word = (size_t)i * words_per_row + word;
                if (v == 1) term.combined[abs_word] |= (1 << (bit + 16));        // +1
                else if (v == 2) term.combined[abs_word] |= (1 << (bit + 16)) | (1 << bit);  // -1
                // v == 0 or 3 → 0 (skip)
            }
        }
        ld.terms.push_back(std::move(term));
    }
    return ld;
}

// ═══════════════════════════════════════════════════════════════════════════
// TEST CONFIG
// ═══════════════════════════════════════════════════════════════════════════
struct TestCase {
    const char* name;
    int out_f, in_f, terms;
};

static const TestCase tests[] = {
    {"q_proj    [576x576]",    576,   576,   10},
    {"k_proj    [192x576]",    192,   576,   10},
    {"v_proj    [192x576]",    192,   576,   10},
    {"o_proj    [576x576]",    576,   576,   12},
    {"gate_proj [1536x576]",  1536,  576,   10},
    {"up_proj   [1536x576]",  1536,  576,   10},
    {"down_proj [576x1536]",   576,  1536,   10},
    {"lm_head   [49152x576]", 49152,  576,   15},
    {"large     [4096x4096]",  4096, 4096,   12},
    {"huge      [1024x65536]",1024, 65536,   15},
};

static const int num_tests = sizeof(tests) / sizeof(tests[0]);

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL DISPATCH TABLE
// ═══════════════════════════════════════════════════════════════════════════
struct KernelEntry {
    CPUArch arch;
    const char* label;
    void (*func)(const uint32_t* const*, const int*, int, int, int,
                 const float*, float*);
};

// Weak declarations for optional kernels
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((weak)) void ternary_mul_avx2(const uint32_t* const*, const int*, int, int, int, const float*, float*);
#elif defined(__aarch64__) || defined(_M_ARM64)
__attribute__((weak)) void ternary_mul_neon(const uint32_t* const*, const int*, int, int, int, const float*, float*);
#endif

// ═══════════════════════════════════════════════════════════════════════════
// BENCHMARK A SINGLE KERNEL
// ═══════════════════════════════════════════════════════════════════════════
double bench_kernel(void (*kernel)(const uint32_t* const*, const int*, int, int, int,
                                    const float*, float*),
                     const LayerData& layer, const float* input, float* output,
                     int warmup, int iterations) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
        const uint32_t* td[32]; int ae[32];
        int na = 0;
        for (int t = 0; t < layer.num_terms && t < 32; t++) {
            if (layer.terms[t].alpha_exp == -128) continue;
            ae[na] = layer.terms[t].alpha_exp;
            td[na] = layer.terms[t].combined.data();
            na++;
        }
        kernel(td, ae, na, layer.out_features, layer.in_features, input, output);
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        const uint32_t* td[32]; int ae[32];
        int na = 0;
        for (int t = 0; t < layer.num_terms && t < 32; t++) {
            if (layer.terms[t].alpha_exp == -128) continue;
            ae[na] = layer.terms[t].alpha_exp;
            td[na] = layer.terms[t].combined.data();
            na++;
        }
        kernel(td, ae, na, layer.out_features, layer.in_features, input, output);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iterations;
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════
int main() {
#ifdef _OPENMP
    #pragma omp parallel
    {
        #pragma omp master
        std::cout << "OpenMP threads: " << omp_get_num_threads() << "\n";
    }
#endif

    // Build kernel table
    struct TestKernel {
        CPUArch arch;
        const char* name;
        void (*func)(const uint32_t* const*, const int*, int, int, int,
                     const float*, float*);
    };
    std::vector<TestKernel> kernels;

    // Always available
    kernels.push_back({CPUArch::X86_64_SCALAR, "SCALAR", ternary_mul_scalar});

    // Register kernels per arch. Uses weak symbols; null if not linked.
#if defined(__x86_64__) || defined(_M_X64)
    if (ternary_mul_avx2)
        kernels.push_back({CPUArch::X86_64_AVX2, "AVX2", ternary_mul_avx2});
#elif defined(__aarch64__) || defined(_M_ARM64)
    if (ternary_mul_neon)
        kernels.push_back({CPUArch::ARM64_NEON, "NEON", ternary_mul_neon});
#endif

    std::cout << "\n═══ Terllama Multi-Kernel Benchmark ═══\n";
    std::cout << "CPU: " << cpu_arch_name(detect_cpu_arch()) << "\n";
    std::cout << "Kernels tested: " << kernels.size() << "\n\n";

    // Column header
    std::cout << std::left << std::setw(24) << "Layer";
    for (auto& k : kernels) {
        std::cout << std::setw(12) << k.name;
    }
    std::cout << "  Best\n";
    std::cout << std::string(24 + 12 * kernels.size() + 6, '-') << "\n";

    // Results table
    double best_total = 0.0, scalar_total = 0.0;

    int full_bench = getenv("TERLLAMA_FULL_BENCH") ? 1 : 0;

    for (int ti = 0; ti < num_tests; ti++) {
        auto& tc = tests[ti];
        // Skip large tests unless TERLLAMA_FULL_BENCH=1
        if ((size_t)tc.out_f * tc.in_f > 10000000 && !full_bench) continue;

        auto layer = make_synthetic_layer(tc.name, tc.out_f, tc.in_f, tc.terms);
        int out_f = tc.out_f, in_f = tc.in_f;

        std::vector<float> input(in_f);
        std::mt19937 rng(ti + 100);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        for (int i = 0; i < in_f; i++) input[i] = fdist(rng);

        std::vector<float> output(out_f);

        int warmup = 2, iterations = 5;
        // Reduce iterations for large matrices
        size_t nops = (size_t)out_f * in_f;
        if (nops > 1000000)   { warmup = 1; iterations = 3; }
        if (nops > 5000000)   { warmup = 1; iterations = 2; }
        if (nops > 10000000)  { warmup = 1; iterations = 1; }

        // Benchmark kernels
        std::cout << std::left << std::setw(24) << tc.name;

        double best_time = 1e30;
        double scal_ms = 0;
        const char* best_name = "";
        for (auto& k : kernels) {
            double ms = bench_kernel(k.func, layer, input.data(), output.data(),
                                     warmup, iterations);
            if (k.arch == CPUArch::X86_64_SCALAR) scal_ms = ms;
            if (ms < best_time) {
                best_time = ms;
                best_name = k.name;
            }

            std::cout << std::right << std::fixed << std::setprecision(1)
                      << std::setw(12) << ms;
        }
        std::cout << "  " << best_name << "\n";

        best_total += best_time;
        scalar_total += scal_ms;
    }

    std::cout << "\n─── Summary ───\n";
    std::cout << "  Scalar total: " << std::fixed << std::setprecision(1)
              << scalar_total << " ms\n";
    std::cout << "  Best total:   " << std::fixed << std::setprecision(1)
              << best_total << " ms\n";
    std::cout << "  Speedup:      " << std::fixed << std::setprecision(2)
              << (scalar_total / best_total) << "\xC3\x97\n";
    std::cout << "  Auto-detect:  " << cpu_arch_name(detect_cpu_arch()) << "\n";

    // ─── Validation ─────────────────────────────────────────────────────
    std::cout << "\n─── Kernel Validation (MSE vs Scalar) ───\n";
    for (int ti = 0; ti < 3 && ti < num_tests; ti++) {
        auto& tc = tests[ti];
        size_t nops_v = (size_t)tc.out_f * tc.in_f;
        if (nops_v > 10000000 && !full_bench) continue;
        auto layer = make_synthetic_layer(tc.name, tc.out_f, tc.in_f, tc.terms);
        int in_f = tc.in_f;
        std::vector<float> input(in_f);
        std::mt19937 rng(ti + 200);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        for (int i = 0; i < in_f; i++) input[i] = fdist(rng);

        std::cout << tc.name << ":\n";
        float max_err = validate_all_kernels(layer, input.data(), nullptr, "scalar");
        (void)max_err;
    }

    // ─── Dispatch ───────────────────────────────────────────────────────────
    std::cout << "\n─── Dispatch Test ───\n";
    for (int ti = 0; ti < num_tests; ti++) {
        auto& tc = tests[ti];
        size_t nops_d = (size_t)tc.out_f * tc.in_f;
        if (nops_d > 10000000 && !full_bench) continue;
        auto layer = make_synthetic_layer(tc.name, tc.out_f, tc.in_f, tc.terms);
        int in_f = tc.in_f;
        std::vector<float> input(in_f);
        std::mt19937 rng(ti + 300);
        std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
        for (int i = 0; i < in_f; i++) input[i] = fdist(rng);
        std::vector<float> out_auto(tc.out_f), out_scalar(tc.out_f);

        // Dispatch (auto-detect)
        ternary_linear(layer, input.data(), out_auto.data(), CPUArch::UNKNOWN);
        // Scalar reference
        ternary_linear(layer, input.data(), out_scalar.data(), CPUArch::X86_64_SCALAR);

        float max_err = 0;
        for (int i = 0; i < tc.out_f; i++) {
            float d = std::abs(out_auto[i] - out_scalar[i]);
            if (d > max_err) max_err = d;
        }
        std::cout << "  " << tc.name << " dispatch vs scalar: max_err = "
                  << std::scientific << max_err << (max_err < 1e-4f ? "  OK" : "  MISMATCH!")
                  << "\n";
    }

    std::cout << "\nDone.\n";
    return 0;
}

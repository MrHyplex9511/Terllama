/*
 * Compare OLD (term-major, sep arrays) vs NEW (row-major, fused combined).
 * Both use AVX-512 + OpenMP.
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
#include <immintrin.h>
#include <omp.h>

// ─── OLD format: separate nz/neg arrays ────────────────────────────────────
struct LayerDataOld {
    int32_t out_f, in_f, N;
    struct { int32_t ae; std::vector<uint16_t> nz, neg; } terms[32];
    int n_active;
};

// ─── NEW format: fused combined ────────────────────────────────────────────
struct LayerDataNew {
    int32_t out_f, in_f, N;
    struct { int32_t ae; size_t stride; const uint32_t* data; } terms[32];
    std::vector<uint32_t> storage; // backing memory
    int n_active;
};

// ─── Generate identical test data ──────────────────────────────────────────
LayerDataOld gen_old(int out_f, int in_f, int N, float sp=0.5f, unsigned seed=42) {
    LayerDataOld ld; ld.out_f = out_f; ld.in_f = in_f; ld.N = N; ld.n_active = 0;
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1,1);
    int wp = (in_f + 15)/16; size_t nw = (size_t)out_f * wp;
    for (int t = 0; t < N; t++) {
        auto& term = ld.terms[ld.n_active++];
        term.ae = t;
        term.nz.assign(nw, 0); term.neg.assign(nw, 0);
        for (int i = 0; i < out_f; i++) {
            for (int j = 0; j < in_f; j++) {
                float v = d(rng);
                int8_t tv = 0;
                if (v > sp) tv = 1; else if (v < -sp) tv = -1;
                if (tv != 0) {
                    int w = j/16, b = j%16; size_t aw = (size_t)i*wp + w;
                    term.nz[aw] |= (uint16_t)(1<<b);
                    if (tv == -1) term.neg[aw] |= (uint16_t)(1<<b);
                }
            }
        }
    }
    return ld;
}

LayerDataNew gen_new(int out_f, int in_f, int N, float sp=0.5f, unsigned seed=42) {
    auto old = gen_old(out_f, in_f, N, sp, seed);
    LayerDataNew ld; ld.out_f = out_f; ld.in_f = in_f; ld.N = N; ld.n_active = old.n_active;
    int wp = (in_f + 15)/16; size_t nw = (size_t)out_f * wp;
    ld.storage.resize(nw * old.n_active);
    for (int t = 0; t < old.n_active; t++) {
        ld.terms[t].ae = old.terms[t].ae;
        ld.terms[t].stride = wp;
        ld.terms[t].data = &ld.storage[t * nw];
        for (size_t k = 0; k < nw; k++) {
            ld.storage[t * nw + k] = ((uint32_t)old.terms[t].nz[k] << 16) | old.terms[t].neg[k];
        }
    }
    return ld;
}

std::vector<float> gen_input(int n, unsigned seed=123) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1,1);
    std::vector<float> x(n); for (int i=0;i<n;i++) x[i]=d(rng); return x;
}

// ═══ OLD kernel (term-major, sep arrays, input re-read per term) ═══
void kernel_old(const LayerDataOld& ld, const float* in, float* out) {
    int wp = (ld.in_f + 15)/16, rem = ld.in_f % 16;
    int fw = rem ? wp-1 : wp; uint16_t tm = rem ? (uint16_t)((1<<rem)-1) : 0;
    std::fill(out, out+ld.out_f, 0.0f);
    for (int t = 0; t < ld.n_active; t++) {
        int ae = ld.terms[t].ae;
        const auto& nz = ld.terms[t].nz;
        const auto& neg = ld.terms[t].neg;
        #pragma omp parallel for
        for (int i = 0; i < ld.out_f; i++) {
            __m512 sum = _mm512_setzero_ps();
            size_t wb = (size_t)i * wp;
            for (int w = 0; w < fw; w++) {
                __m512 v = _mm512_loadu_ps(&in[w*16]);
                uint16_t nzw = nz[wb+w], negw = neg[wb+w];
                __mmask16 ma = nzw & ~negw, ms = nzw & negw;
                sum = _mm512_mask_add_ps(sum, ma, sum, v);
                sum = _mm512_mask_sub_ps(sum, ms, sum, v);
            }
            if (rem) {
                int w = fw; __m512 v = _mm512_loadu_ps(&in[w*16]);
                uint16_t nzw = nz[wb+w]&tm, negw = neg[wb+w]&tm;
                __mmask16 ma = nzw & ~negw, ms = nzw & negw;
                sum = _mm512_mask_add_ps(sum, ma, sum, v);
                sum = _mm512_mask_sub_ps(sum, ms, sum, v);
            }
            out[i] += std::ldexp(_mm512_reduce_add_ps(sum), ae);
        }
    }
}

// ═══ NEW kernel (row-major, fused combined, input read ONCE) ═══
void kernel_new(const LayerDataNew& ld, const float* in, float* out) {
    int wp = (ld.in_f + 15)/16, rem = ld.in_f % 16;
    int fw = rem ? wp-1 : wp; uint32_t tm = rem ? (uint32_t)((1<<rem)-1) : 0;
    std::fill(out, out+ld.out_f, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < ld.out_f; i++) {
        // Per-term SIMD accumulators (stay in zmm registers)
        __m512 vacc[32];
        for (int t = 0; t < ld.n_active; t++) vacc[t] = _mm512_setzero_ps();

        for (int w = 0; w < fw; w++) {
            __m512 v = _mm512_loadu_ps(&in[w*16]);
            for (int t = 0; t < ld.n_active; t++) {
                uint32_t c = ld.terms[t].data[i * ld.terms[t].stride + w];
                uint16_t nzw = c>>16, negw = c&0xFFFF;
                __mmask16 ma = nzw & ~negw, ms = nzw & negw;
                vacc[t] = _mm512_mask_add_ps(vacc[t], ma, vacc[t], v);
                vacc[t] = _mm512_mask_sub_ps(vacc[t], ms, vacc[t], v);
            }
        }
        if (rem) {
            __m512 v = _mm512_loadu_ps(&in[fw*16]);
            for (int t = 0; t < ld.n_active; t++) {
                uint32_t c = ld.terms[t].data[i * ld.terms[t].stride + fw] & (tm | (tm<<16));
                uint16_t nzw = c>>16, negw = c&0xFFFF;
                __mmask16 ma = nzw & ~negw, ms = nzw & negw;
                vacc[t] = _mm512_mask_add_ps(vacc[t], ma, vacc[t], v);
                vacc[t] = _mm512_mask_sub_ps(vacc[t], ms, vacc[t], v);
            }
        }
        // Reduce once per term (not per-word!)
        float r = 0;
        for (int t = 0; t < ld.n_active; t++) {
            r += std::ldexp(_mm512_reduce_add_ps(vacc[t]), ld.terms[t].ae);
        }
        out[i] = r;
    }
}

// ═══ MAIN ═══
int main() {
    std::cout << "═══ Memory Coalescing Benchmark ═══\n";
    #pragma omp parallel
    {
        #pragma omp master
        std::cout << "  OpenMP threads: " << omp_get_num_threads() << "\n";
    }

    struct Test { const char* name; int of, inf; int N; };
    Test tests[] = {
        {"q_proj [2048x2048]", 2048, 2048, 12},
        {"gate_proj [5632x2048]", 5632, 2048, 12},
        {"lm_head [576x96000]", 576, 96000, 15},
        {"large [4096x4096]", 4096, 4096, 12},
        {"huge [1024x65536]", 1024, 65536, 10},
    };

    std::cout << "\n" << std::left << std::setw(28) << "Layer"
              << std::right << std::setw(12) << "OLD (ms)" << std::setw(12) << "NEW (ms)"
              << std::setw(10) << "Speedup" << std::setw(14) << "MSE" << "\n"
              << std::string(76, '-') << "\n";

    for (auto& tc : tests) {
        auto old = gen_old(tc.of, tc.inf, tc.N);
        auto nw = gen_new(tc.of, tc.inf, tc.N);
        auto inp = gen_input(tc.inf);
        std::vector<float> oo(tc.of), on(tc.of);

        // Warmup
        kernel_old(old, inp.data(), oo.data());
        kernel_new(nw, inp.data(), on.data());

        // Benchmark OLD
        int reps = std::max(1, std::min(50, 50000 / (tc.of * tc.inf / 1000)));
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; r++) { kernel_old(old, inp.data(), oo.data()); }
        auto t1 = std::chrono::high_resolution_clock::now();
        double dt_old = std::chrono::duration<double,std::milli>(t1-t0).count()/reps;

        // Benchmark NEW
        t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < reps; r++) { kernel_new(nw, inp.data(), on.data()); }
        t1 = std::chrono::high_resolution_clock::now();
        double dt_new = std::chrono::duration<double,std::milli>(t1-t0).count()/reps;

        double sp = dt_old / dt_new;
        double mse = 0;
        for (int i = 0; i < tc.of; i++) { double d = oo[i]-on[i]; mse += d*d; }
        mse /= tc.of;

        std::cout << std::left << std::setw(28) << tc.name << std::right
                  << std::setw(11) << std::fixed << std::setprecision(2) << dt_old << "ms"
                  << std::setw(11) << std::fixed << std::setprecision(2) << dt_new << "ms"
                  << std::setw(9) << std::fixed << std::setprecision(2) << sp << "×"
                  << std::setw(14) << std::scientific << std::setprecision(2) << mse << "\n";
    }

    // Estimated full-model throughput
    std::cout << "\n─── Estimated 1.1B model throughput ───\n";
    struct { const char* name; int of, inf, N, count; } model[] = {
        {"q_proj", 2048, 2048, 12, 22},
        {"k_proj", 256, 2048, 12, 22},
        {"v_proj", 256, 2048, 12, 22},
        {"o_proj", 2048, 2048, 12, 22},
        {"gate_proj", 5632, 2048, 12, 22},
        {"up_proj", 5632, 2048, 12, 22},
        {"down_proj", 2048, 5632, 12, 22},
        {"lm_head", 32000, 2048, 12, 1},
    };
    double new_total = 0, old_total = 0;
    for (auto& ml : model) {
        auto old = gen_old(ml.of, ml.inf, ml.N);
        auto nw = gen_new(ml.of, ml.inf, ml.N);
        auto inp = gen_input(ml.inf);
        std::vector<float> oo(ml.of);
        auto t0 = std::chrono::high_resolution_clock::now();
        kernel_old(old, inp.data(), oo.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        old_total += std::chrono::duration<double,std::milli>(t1-t0).count() * ml.count;
        t0 = std::chrono::high_resolution_clock::now();
        kernel_new(nw, inp.data(), oo.data());
        t1 = std::chrono::high_resolution_clock::now();
        new_total += std::chrono::duration<double,std::milli>(t1-t0).count() * ml.count;
    }
    double ovh = 0.3; // RMSNorm + RoPE + softmax overhead
    std::cout << "  OLD: " << std::fixed << std::setprecision(1) << old_total << " ms, "
              << (1000.0/(old_total*(1+ovh))) << " tok/s\n";
    std::cout << "  NEW: " << std::fixed << std::setprecision(1) << new_total << " ms, "
              << (1000.0/(new_total*(1+ovh))) << " tok/s\n";
    std::cout << "  Speedup: " << (old_total/new_total) << "×\n";
    return 0;
}

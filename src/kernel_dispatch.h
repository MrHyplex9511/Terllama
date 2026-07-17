/*
 * kernel_dispatch.h: CPU auto-detection + arch-specific kernels
 *
 * Compile all kernels with __attribute__((target(...))).
 * Runtime dispatch via __builtin_cpu_supports().
 *
 * Architectures:
 *   AVX-512  → 512-bit (Skylake-X+)
 *   AVX2     → 256-bit + FMA (Haswell+)
 *   AVX      → 256-bit (Sandy Bridge+)
 *   SSE4.2   → 128-bit (Nehalem+)
 *   NEON     → 128-bit (ARM64 / Apple Silicon)
 *   Scalar   → Pure C++ fallback (any CPU)
 */
#pragma once
#include "model.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstdint>

// Arch-specific intrinsics headers
// #pragma GCC target includes ALL intrinsic decls regardless
// of compile-time -march. Per-function __attribute__((target(...)))
// generates correct code for each ISA variant.
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC push_options
        #pragma GCC target("sse4.2,avx,avx2,fma,avx512f,avx512dq,avx512vl,avx512bw")
    #endif
    #include <immintrin.h>
    #if defined(__GNUC__) || defined(__clang__)
        #pragma GCC pop_options
    #endif
#endif
#if defined(__aarch64__)
    #include <arm_neon.h>    // ARM NEON intrinsics
#endif

// ═══════════════════════════════════════════════════════════════════════════
// CPU ARCHITECTURE DETECTION
// ═══════════════════════════════════════════════════════════════════════════
inline CPUArch detect_cpu_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    // Check in descending order of capability
    #if defined(__GNUC__) || defined(__clang__)
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq"))
            return CPUArch::X86_64_AVX512;
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            return CPUArch::X86_64_AVX2;
        if (__builtin_cpu_supports("avx"))
            return CPUArch::X86_64_AVX;
        if (__builtin_cpu_supports("sse4.2"))
            return CPUArch::X86_64_SSE42;
    #endif
    return CPUArch::X86_64_SCALAR;
#elif defined(__aarch64__) || defined(_M_ARM64)
    // All ARM64 CPUs have NEON
    return CPUArch::ARM64_NEON;
#else
    return CPUArch::X86_64_SCALAR;
#endif
}

// ═══════════════════════════════════════════════════════════════════════════
// KERNEL COMMON HELPERS
// ═══════════════════════════════════════════════════════════════════════════
// Extract per-term data from LayerData into flat arrays for kernel dispatch.
// Returns number of active (non-zero-alpha) terms.
inline int extract_term_data(const LayerData& layer,
                             const uint32_t** term_data_out,
                             int* alpha_exps_out,
                             int max_terms) {
    int n = 0;
    for (int t = 0; t < layer.num_terms && t < max_terms; t++) {
        int32_t ae = layer.terms[t].alpha_exp;
        if (ae == -128) continue;
        alpha_exps_out[n] = ae;
        term_data_out[n] = layer.terms[t].combined.data();
        n++;
    }
    return n;
}

// ═══════════════════════════════════════════════════════════════════════════
// SCALAR KERNEL (pure C++, any CPU)
// ═══════════════════════════════════════════════════════════════════════════
inline void ternary_mul_scalar(const uint32_t* const* term_data,
                                const int* alpha_exps,
                                int n_active, int out_f, int in_f,
                                const float* input, float* output) {
    int words_per_row = (in_f + 15) / 16;
    std::fill(output, output + out_f, 0.0f);

    for (int t = 0; t < n_active; t++) {
        int ae = alpha_exps[t];
        if (ae == -128) continue;
        const uint32_t* comb = term_data[t];
        size_t stride = (size_t)words_per_row;

        for (int i = 0; i < out_f; i++) {
            float sum = 0.0f;
            size_t base = (size_t)i * stride;
            for (int j = 0; j < in_f; j++) {
                int word = j / 16;
                int bit = j % 16;
                uint32_t c = comb[base + word];
                bool is_nz = (c >> (bit + 16)) & 1;
                bool is_neg = (c >> bit) & 1;
                if (is_nz) sum += is_neg ? -input[j] : input[j];
            }
            output[i] += std::ldexp(sum, ae);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// x86-64 KERNELS (target-annotated for per-function ISA generation)
// ═══════════════════════════════════════════════════════════════════════════
#if defined(__x86_64__) || defined(_M_X64)

// ─── SSE4.2 (128-bit SIMD) ───────────────────────────────────────────────
__attribute__((target("sse4.2")))
inline void ternary_mul_sse42(const uint32_t* const* term_data,
                               const int* alpha_exps,
                               int n_active, int out_f, int in_f,
                               const float* input, float* output) {
    int words_per_row = (in_f + 15) / 16;
    int rem = in_f % 16;
    int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
    uint32_t tail_mask = rem > 0 ? (uint32_t)((1 << rem) - 1) : 0;

    // Per-term stride: words per row
    size_t stride = (size_t)words_per_row;

    #pragma omp parallel for
    for (int i = 0; i < out_f; i++) {
        __m128 vacc[32];
        for (int t = 0; t < n_active; t++) vacc[t] = _mm_setzero_ps();

        for (int w = 0; w < full_words; w++) {
            // Load all 16 input elements
            __m128 v0 = _mm_loadu_ps(&input[w * 16 + 0]);
            __m128 v1 = _mm_loadu_ps(&input[w * 16 + 4]);
            __m128 v2 = _mm_loadu_ps(&input[w * 16 + 8]);
            __m128 v3 = _mm_loadu_ps(&input[w * 16 + 12]);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + w];
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                // Chunk 0: bits [0..3]
                { uint32_t add_bits = (nzw & ~negw) & 0xF;
                  uint32_t sub_bits = (nzw & negw) & 0xF;
                  // Broadcast each bit to full dword mask for blendv
                  __m128i addm = _mm_set_epi32(
                      add_bits & 8 ? -1 : 0, add_bits & 4 ? -1 : 0,
                      add_bits & 2 ? -1 : 0, add_bits & 1 ? -1 : 0);
                  __m128i subm = _mm_set_epi32(
                      sub_bits & 8 ? -1 : 0, sub_bits & 4 ? -1 : 0,
                      sub_bits & 2 ? -1 : 0, sub_bits & 1 ? -1 : 0);
                  __m128 zero = _mm_setzero_ps();
                  vacc[t] = _mm_add_ps(vacc[t], _mm_blendv_ps(zero, v0, _mm_castsi128_ps(addm)));
                  vacc[t] = _mm_sub_ps(vacc[t], _mm_blendv_ps(zero, v0, _mm_castsi128_ps(subm)));
                }
                // Chunk 1: bits [4..7]
                { uint32_t add_bits = ((nzw & ~negw) >> 4) & 0xF;
                  uint32_t sub_bits = ((nzw & negw) >> 4) & 0xF;
                  __m128i addm = _mm_set_epi32(
                      add_bits & 8 ? -1 : 0, add_bits & 4 ? -1 : 0,
                      add_bits & 2 ? -1 : 0, add_bits & 1 ? -1 : 0);
                  __m128i subm = _mm_set_epi32(
                      sub_bits & 8 ? -1 : 0, sub_bits & 4 ? -1 : 0,
                      sub_bits & 2 ? -1 : 0, sub_bits & 1 ? -1 : 0);
                  __m128 zero = _mm_setzero_ps();
                  vacc[t] = _mm_add_ps(vacc[t], _mm_blendv_ps(zero, v1, _mm_castsi128_ps(addm)));
                  vacc[t] = _mm_sub_ps(vacc[t], _mm_blendv_ps(zero, v1, _mm_castsi128_ps(subm)));
                }
                // Chunk 2: bits [8..11]
                { uint32_t add_bits = ((nzw & ~negw) >> 8) & 0xF;
                  uint32_t sub_bits = ((nzw & negw) >> 8) & 0xF;
                  __m128i addm = _mm_set_epi32(
                      add_bits & 8 ? -1 : 0, add_bits & 4 ? -1 : 0,
                      add_bits & 2 ? -1 : 0, add_bits & 1 ? -1 : 0);
                  __m128i subm = _mm_set_epi32(
                      sub_bits & 8 ? -1 : 0, sub_bits & 4 ? -1 : 0,
                      sub_bits & 2 ? -1 : 0, sub_bits & 1 ? -1 : 0);
                  __m128 zero = _mm_setzero_ps();
                  vacc[t] = _mm_add_ps(vacc[t], _mm_blendv_ps(zero, v2, _mm_castsi128_ps(addm)));
                  vacc[t] = _mm_sub_ps(vacc[t], _mm_blendv_ps(zero, v2, _mm_castsi128_ps(subm)));
                }
                // Chunk 3: bits [12..15]
                { uint32_t add_bits = ((nzw & ~negw) >> 12) & 0xF;
                  uint32_t sub_bits = ((nzw & negw) >> 12) & 0xF;
                  __m128i addm = _mm_set_epi32(
                      add_bits & 8 ? -1 : 0, add_bits & 4 ? -1 : 0,
                      add_bits & 2 ? -1 : 0, add_bits & 1 ? -1 : 0);
                  __m128i subm = _mm_set_epi32(
                      sub_bits & 8 ? -1 : 0, sub_bits & 4 ? -1 : 0,
                      sub_bits & 2 ? -1 : 0, sub_bits & 1 ? -1 : 0);
                  __m128 zero = _mm_setzero_ps();
                  vacc[t] = _mm_add_ps(vacc[t], _mm_blendv_ps(zero, v3, _mm_castsi128_ps(addm)));
                  vacc[t] = _mm_sub_ps(vacc[t], _mm_blendv_ps(zero, v3, _mm_castsi128_ps(subm)));
                }
            }
        }

        // Tail word
        if (rem > 0) {
            __m128 v0 = _mm_loadu_ps(&input[full_words * 16 + 0]);
            __m128 v1 = _mm_loadu_ps(&input[full_words * 16 + 4]);
            __m128 v2 = _mm_loadu_ps(&input[full_words * 16 + 8]);
            __m128 v3 = (rem > 12) ? _mm_loadu_ps(&input[full_words * 16 + 12]) : _mm_setzero_ps();
            uint32_t tm = tail_mask;

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + full_words] & (tm | (tm << 16));
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                for (int ch = 0; ch < 4; ch++) {
                    int sh = ch * 4;
                    __m128 vv = ch==0? v0 : ch==1? v1 : ch==2? v2 : v3;
                    int chbits = rem - sh;
                    if (chbits <= 0) break;
                    if (chbits > 4) chbits = 4;
                    uint32_t cm = (1 << chbits) - 1;
                    uint32_t add_bits = ((nzw & ~negw) >> sh) & cm;
                    uint32_t sub_bits = ((nzw & negw) >> sh) & cm;
                    __m128i addm = _mm_set_epi32(
                        (chbits>3 && (add_bits&8)) ? -1 : 0,
                        (chbits>2 && (add_bits&4)) ? -1 : 0,
                        (chbits>1 && (add_bits&2)) ? -1 : 0,
                        (chbits>0 && (add_bits&1)) ? -1 : 0);
                    __m128i subm = _mm_set_epi32(
                        (chbits>3 && (sub_bits&8)) ? -1 : 0,
                        (chbits>2 && (sub_bits&4)) ? -1 : 0,
                        (chbits>1 && (sub_bits&2)) ? -1 : 0,
                        (chbits>0 && (sub_bits&1)) ? -1 : 0);
                    __m128 zero = _mm_setzero_ps();
                    vacc[t] = _mm_add_ps(vacc[t], _mm_blendv_ps(zero, vv, _mm_castsi128_ps(addm)));
                    vacc[t] = _mm_sub_ps(vacc[t], _mm_blendv_ps(zero, vv, _mm_castsi128_ps(subm)));
                }
            }
        }

        float result = 0.0f;
        for (int t = 0; t < n_active; t++) {
            // Horizontal add: extract 4 floats and sum
            float buf[4];
            _mm_storeu_ps(buf, vacc[t]);
            result += std::ldexp(buf[0] + buf[1] + buf[2] + buf[3], alpha_exps[t]);
        }
        output[i] = result;
    }
}

// ─── AVX (256-bit SIMD, no FMA) ──────────────────────────────────────────
__attribute__((target("avx")))
inline void ternary_mul_avx(const uint32_t* const* term_data,
                             const int* alpha_exps,
                             int n_active, int out_f, int in_f,
                             const float* input, float* output) {
    int words_per_row = (in_f + 15) / 16;
    int rem = in_f % 16;
    int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
    uint32_t tail_mask = rem > 0 ? (uint32_t)((1 << rem) - 1) : 0;
    size_t stride = (size_t)words_per_row;

    #pragma omp parallel for
    for (int i = 0; i < out_f; i++) {
        __m256 vacc[32];
        for (int t = 0; t < n_active; t++) vacc[t] = _mm256_setzero_ps();

        for (int w = 0; w < full_words; w++) {
            __m256 v0 = _mm256_loadu_ps(&input[w * 16 + 0]);
            __m256 v1 = _mm256_loadu_ps(&input[w * 16 + 8]);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + w];
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                // Low 8 bits → v0
                { uint32_t abits = (nzw & ~negw) & 0xFF;
                  uint32_t sbits = (nzw & negw) & 0xFF;
                  __m256i addm = _mm256_set_epi32(
                      abits&128?-1:0, abits&64?-1:0, abits&32?-1:0, abits&16?-1:0,
                      abits&8?-1:0, abits&4?-1:0, abits&2?-1:0, abits&1?-1:0);
                  __m256i subm = _mm256_set_epi32(
                      sbits&128?-1:0, sbits&64?-1:0, sbits&32?-1:0, sbits&16?-1:0,
                      sbits&8?-1:0, sbits&4?-1:0, sbits&2?-1:0, sbits&1?-1:0);
                  __m256 zero = _mm256_setzero_ps();
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(subm)));
                }
                // High 8 bits → v1
                { uint32_t abits = ((nzw & ~negw) >> 8) & 0xFF;
                  uint32_t sbits = ((nzw & negw) >> 8) & 0xFF;
                  __m256i addm = _mm256_set_epi32(
                      abits&128?-1:0, abits&64?-1:0, abits&32?-1:0, abits&16?-1:0,
                      abits&8?-1:0, abits&4?-1:0, abits&2?-1:0, abits&1?-1:0);
                  __m256i subm = _mm256_set_epi32(
                      sbits&128?-1:0, sbits&64?-1:0, sbits&32?-1:0, sbits&16?-1:0,
                      sbits&8?-1:0, sbits&4?-1:0, sbits&2?-1:0, sbits&1?-1:0);
                  __m256 zero = _mm256_setzero_ps();
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(subm)));
                }
            }
        }

        // Tail word
        if (rem > 0) {
            uint32_t tm = tail_mask;
            __m256 v0 = _mm256_loadu_ps(&input[full_words * 16 + 0]);
            __m256 v1 = (rem > 8) ? _mm256_loadu_ps(&input[full_words * 16 + 8]) : _mm256_setzero_ps();

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + full_words] & (tm | (tm << 16));
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                // Low chunk (up to 8 bits from bit 0)
                { int chbits = rem > 8 ? 8 : rem;
                  uint32_t cm = (1 << chbits) - 1;
                  uint32_t abits = ((nzw & ~negw) >> 0) & cm;
                  uint32_t sbits = ((nzw & negw) >> 0) & cm;
                  int a[8]={0}, sb[8]={0};
                  for (int b = 0; b < chbits; b++) {
                      a[b] = (abits & (1 << b)) ? -1 : 0;
                      sb[b] = (sbits & (1 << b)) ? -1 : 0;
                  }
                  __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                  __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
                  __m256 zero = _mm256_setzero_ps();
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(subm)));
                }
                // High chunk (if rem > 8)
                if (rem > 8) {
                    int chbits = rem - 8;
                    uint32_t cm = (1 << chbits) - 1;
                    uint32_t abits = ((nzw & ~negw) >> 8) & cm;
                    uint32_t sbits = ((nzw & negw) >> 8) & cm;
                    int a[8]={0}, sb[8]={0};
                    for (int b = 0; b < chbits; b++) {
                        a[b] = (abits & (1 << b)) ? -1 : 0;
                        sb[b] = (sbits & (1 << b)) ? -1 : 0;
                    }
                    __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                    __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
                    __m256 zero = _mm256_setzero_ps();
                    vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(addm)));
                    vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(subm)));
                }
            }
        }

        float result = 0.0f;
        for (int t = 0; t < n_active; t++) {
            float buf[8];
            _mm256_storeu_ps(buf, vacc[t]);
            float s = 0.0f;
            for (int k = 0; k < 8; k++) s += buf[k];
            result += std::ldexp(s, alpha_exps[t]);
        }
        output[i] = result;
    }
}

// ─── AVX2+FMA (256-bit SIMD, uses FMA for RoPE if applicable) ────────────
// Ternary kernel uses add/sub only, no FMA. AVX2 has better
// _mm256_blendv_ps throughput and general perf.
__attribute__((target("avx2,fma")))
inline void ternary_mul_avx2(const uint32_t* const* term_data,
                              const int* alpha_exps,
                              int n_active, int out_f, int in_f,
                              const float* input, float* output) {
    // Same algorithm as AVX. AVX2+FMA target guarantees CPU capability.
    // Identical _mm256_blendv_ps and setzero. RoPE/attention uses FMA
    // (handled separately).
    int words_per_row = (in_f + 15) / 16;
    int rem = in_f % 16;
    int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
    uint32_t tail_mask = rem > 0 ? (uint32_t)((1 << rem) - 1) : 0;
    size_t stride = (size_t)words_per_row;

    #pragma omp parallel for
    for (int i = 0; i < out_f; i++) {
        __m256 vacc[32];
        for (int t = 0; t < n_active; t++) vacc[t] = _mm256_setzero_ps();

        for (int w = 0; w < full_words; w++) {
            __m256 v0 = _mm256_loadu_ps(&input[w * 16 + 0]);
            __m256 v1 = _mm256_loadu_ps(&input[w * 16 + 8]);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + w];
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                // Low 8 bits → v0
                { uint32_t abits = (nzw & ~negw) & 0xFF;
                  uint32_t sbits = (nzw & negw) & 0xFF;
                  int a[8], sb[8];
                  for (int b = 0; b < 8; b++) {
                      a[b] = (abits & (1 << b)) ? -1 : 0;
                      sb[b] = (sbits & (1 << b)) ? -1 : 0;
                  }
                  __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                  __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
                  __m256 zero = _mm256_setzero_ps();
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(subm)));
                }
                // High 8 bits → v1
                { uint32_t abits = ((nzw & ~negw) >> 8) & 0xFF;
                  uint32_t sbits = ((nzw & negw) >> 8) & 0xFF;
                  int a[8], sb[8];
                  for (int b = 0; b < 8; b++) {
                      a[b] = (abits & (1 << b)) ? -1 : 0;
                      sb[b] = (sbits & (1 << b)) ? -1 : 0;
                  }
                  __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                  __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
                  __m256 zero = _mm256_setzero_ps();
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(subm)));
                }
            }
        }

        // Tail word
        if (rem > 0) {
            uint32_t tm = tail_mask;
            __m256 v0 = _mm256_loadu_ps(&input[full_words * 16 + 0]);
            __m256 v1 = (rem > 8) ? _mm256_loadu_ps(&input[full_words * 16 + 8]) : _mm256_setzero_ps();

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + full_words] & (tm | (tm << 16));
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                // Low chunk
                { int chbits = rem > 8 ? 8 : rem;
                  uint32_t cm = (1 << chbits) - 1;
                  uint32_t abits = ((nzw & ~negw) >> 0) & cm;
                  uint32_t sbits = ((nzw & negw) >> 0) & cm;
                  int a[8]={0}, sb[8]={0};
                  for (int b = 0; b < chbits; b++) {
                      a[b] = (abits & (1 << b)) ? -1 : 0;
                      sb[b] = (sbits & (1 << b)) ? -1 : 0;
                  }
                  __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                  __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
                  __m256 zero = _mm256_setzero_ps();
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(subm)));
                }
                if (rem > 8) {
                    int chbits = rem - 8;
                    uint32_t cm = (1 << chbits) - 1;
                    uint32_t abits = ((nzw & ~negw) >> 8) & cm;
                    uint32_t sbits = ((nzw & negw) >> 8) & cm;
                    int a[8]={0}, sb[8]={0};
                    for (int b = 0; b < chbits; b++) {
                        a[b] = (abits & (1 << b)) ? -1 : 0;
                        sb[b] = (sbits & (1 << b)) ? -1 : 0;
                    }
                    __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                    __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
                    __m256 zero = _mm256_setzero_ps();
                    vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(addm)));
                    vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(subm)));
                }
            }
        }

        float result = 0.0f;
        for (int t = 0; t < n_active; t++) {
            float buf[8];
            _mm256_storeu_ps(buf, vacc[t]);
            float s = 0;
            for (int k = 0; k < 8; k++) s += buf[k];
            result += std::ldexp(s, alpha_exps[t]);
        }
        output[i] = result;
    }
}

// ─── AVX-512 (512-bit SIMD) ──────────────────────────────────────────────
__attribute__((target("avx512f,avx512dq")))
inline void ternary_mul_avx512(const uint32_t* const* term_data,
                                const int* alpha_exps,
                                int n_active, int out_f, int in_f,
                                const float* input, float* output) {
    int words_per_row = (in_f + 15) / 16;
    int rem = in_f % 16;
    int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
    uint32_t tail_mask = rem > 0 ? (uint32_t)((1 << rem) - 1) : 0;
    size_t stride = (size_t)words_per_row;

    #pragma omp parallel for
    for (int i = 0; i < out_f; i++) {
        __m512 vacc[32];
        for (int t = 0; t < n_active; t++) vacc[t] = _mm512_setzero_ps();

        for (int w = 0; w < full_words; w++) {
            __m512 v = _mm512_loadu_ps(&input[w * 16]);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + w];
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;
                __mmask16 ma = (__mmask16)(nzw & ~negw);
                __mmask16 ms = (__mmask16)(nzw & negw);
                vacc[t] = _mm512_mask_add_ps(vacc[t], ma, vacc[t], v);
                vacc[t] = _mm512_mask_sub_ps(vacc[t], ms, vacc[t], v);
            }
        }

        if (rem > 0) {
            __m512 v = _mm512_loadu_ps(&input[full_words * 16]);
            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + full_words] & (tail_mask | (tail_mask << 16));
                __mmask16 ma = (__mmask16)((c >> 16) & ~(uint16_t)(c & 0xFFFF));
                __mmask16 ms = (__mmask16)((c >> 16) & (uint16_t)(c & 0xFFFF));
                vacc[t] = _mm512_mask_add_ps(vacc[t], ma, vacc[t], v);
                vacc[t] = _mm512_mask_sub_ps(vacc[t], ms, vacc[t], v);
            }
        }

        float result = 0.0f;
        for (int t = 0; t < n_active; t++) {
            result += std::ldexp(_mm512_reduce_add_ps(vacc[t]), alpha_exps[t]);
        }
        output[i] = result;
    }
}

#endif // x86-64

// ═══════════════════════════════════════════════════════════════════════════
// ARM64 NEON KERNEL (128-bit SIMD)
// ═══════════════════════════════════════════════════════════════════════════
#if defined(__aarch64__)
inline void ternary_mul_neon(const uint32_t* const* term_data,
                              const int* alpha_exps,
                              int n_active, int out_f, int in_f,
                              const float* input, float* output) {
    int words_per_row = (in_f + 15) / 16;
    int rem = in_f % 16;
    int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
    uint32_t tail_mask = rem > 0 ? (uint32_t)((1 << rem) - 1) : 0;
    size_t stride = (size_t)words_per_row;

    #pragma omp parallel for
    for (int i = 0; i < out_f; i++) {
        float32x4_t vacc[32];
        for (int t = 0; t < n_active; t++) vacc[t] = vdupq_n_f32(0.0f);

        for (int w = 0; w < full_words; w++) {
            float32x4_t v0 = vld1q_f32(&input[w * 16 + 0]);
            float32x4_t v1 = vld1q_f32(&input[w * 16 + 4]);
            float32x4_t v2 = vld1q_f32(&input[w * 16 + 8]);
            float32x4_t v3 = vld1q_f32(&input[w * 16 + 12]);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + w];
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                // Chunk 0: bits [0..3]
                { uint32_t add_bits = (nzw & ~negw) & 0xF;
                  uint32_t sub_bits = (nzw & negw) & 0xF;
                  uint32_t addm[4] = {add_bits&1?0xFFFFFFFF:0, add_bits&2?0xFFFFFFFF:0,
                                       add_bits&4?0xFFFFFFFF:0, add_bits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sub_bits&1?0xFFFFFFFF:0, sub_bits&2?0xFFFFFFFF:0,
                                       sub_bits&4?0xFFFFFFFF:0, sub_bits&8?0xFFFFFFFF:0};
                  uint32x4_t am = vld1q_u32(addm);
                  uint32x4_t sm = vld1q_u32(subm);
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(am, vreinterpretq_u32_f32(v0))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(sm, vreinterpretq_u32_f32(v0))));
                }
                // Chunk 1: bits [4..7]
                { uint32_t add_bits = ((nzw & ~negw) >> 4) & 0xF;
                  uint32_t sub_bits = ((nzw & negw) >> 4) & 0xF;
                  float32x4_t vv = v1;
                  uint32_t addm[4] = {add_bits&1?0xFFFFFFFF:0, add_bits&2?0xFFFFFFFF:0,
                                       add_bits&4?0xFFFFFFFF:0, add_bits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sub_bits&1?0xFFFFFFFF:0, sub_bits&2?0xFFFFFFFF:0,
                                       sub_bits&4?0xFFFFFFFF:0, sub_bits&8?0xFFFFFFFF:0};
                  uint32x4_t am = vld1q_u32(addm);
                  uint32x4_t sm = vld1q_u32(subm);
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(am, vreinterpretq_u32_f32(vv))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(sm, vreinterpretq_u32_f32(vv))));
                }
                // Chunk 2: bits [8..11]
                { uint32_t add_bits = ((nzw & ~negw) >> 8) & 0xF;
                  uint32_t sub_bits = ((nzw & negw) >> 8) & 0xF;
                  float32x4_t vv = v2;
                  uint32_t addm[4] = {add_bits&1?0xFFFFFFFF:0, add_bits&2?0xFFFFFFFF:0,
                                       add_bits&4?0xFFFFFFFF:0, add_bits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sub_bits&1?0xFFFFFFFF:0, sub_bits&2?0xFFFFFFFF:0,
                                       sub_bits&4?0xFFFFFFFF:0, sub_bits&8?0xFFFFFFFF:0};
                  uint32x4_t am = vld1q_u32(addm);
                  uint32x4_t sm = vld1q_u32(subm);
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(am, vreinterpretq_u32_f32(vv))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(sm, vreinterpretq_u32_f32(vv))));
                }
                // Chunk 3: bits [12..15]
                { uint32_t add_bits = ((nzw & ~negw) >> 12) & 0xF;
                  uint32_t sub_bits = ((nzw & negw) >> 12) & 0xF;
                  float32x4_t vv = v3;
                  uint32_t addm[4] = {add_bits&1?0xFFFFFFFF:0, add_bits&2?0xFFFFFFFF:0,
                                       add_bits&4?0xFFFFFFFF:0, add_bits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sub_bits&1?0xFFFFFFFF:0, sub_bits&2?0xFFFFFFFF:0,
                                       sub_bits&4?0xFFFFFFFF:0, sub_bits&8?0xFFFFFFFF:0};
                  uint32x4_t am = vld1q_u32(addm);
                  uint32x4_t sm = vld1q_u32(subm);
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(am, vreinterpretq_u32_f32(vv))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(sm, vreinterpretq_u32_f32(vv))));
                }
            }
        }

        // Tail word
        if (rem > 0) {
            float32x4_t v0 = vld1q_f32(&input[full_words * 16 + 0]);
            float32x4_t v1 = (rem > 4) ? vld1q_f32(&input[full_words * 16 + 4]) : vdupq_n_f32(0.0f);
            float32x4_t v2 = (rem > 8) ? vld1q_f32(&input[full_words * 16 + 8]) : vdupq_n_f32(0.0f);
            float32x4_t v3 = (rem > 12) ? vld1q_f32(&input[full_words * 16 + 12]) : vdupq_n_f32(0.0f);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + full_words] & (tail_mask | (tail_mask << 16));
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                float32x4_t vv[4] = {v0, v1, v2, v3};
                for (int ch = 0; ch < 4; ch++) {
                    int sh = ch * 4;
                    int chbits = rem - sh;
                    if (chbits <= 0) break;
                    if (chbits > 4) chbits = 4;
                    uint32_t cm = (1 << chbits) - 1;
                    uint32_t add_bits = ((nzw & ~negw) >> sh) & cm;
                    uint32_t sub_bits = ((nzw & negw) >> sh) & cm;
                    uint32_t addm[4] = {(chbits>0&&add_bits&1)?0xFFFFFFFF:0,
                                        (chbits>1&&add_bits&2)?0xFFFFFFFF:0,
                                        (chbits>2&&add_bits&4)?0xFFFFFFFF:0,
                                        (chbits>3&&add_bits&8)?0xFFFFFFFF:0};
                    uint32_t subm[4] = {(chbits>0&&sub_bits&1)?0xFFFFFFFF:0,
                                        (chbits>1&&sub_bits&2)?0xFFFFFFFF:0,
                                        (chbits>2&&sub_bits&4)?0xFFFFFFFF:0,
                                        (chbits>3&&sub_bits&8)?0xFFFFFFFF:0};
                    uint32x4_t am = vld1q_u32(addm);
                    uint32x4_t sm = vld1q_u32(subm);
                    vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(am, vreinterpretq_u32_f32(vv[ch]))));
                    vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(sm, vreinterpretq_u32_f32(vv[ch]))));
                }
            }
        }

        float result = 0.0f;
        for (int t = 0; t < n_active; t++) {
            float s = vaddvq_f32(vacc[t]);  // NEON horizontal add
            result += std::ldexp(s, alpha_exps[t]);
        }
        output[i] = result;
    }
}
#endif // __aarch64__

// ═══════════════════════════════════════════════════════════════════════════
// DISPATCHER
// ═══════════════════════════════════════════════════════════════════════════
// Dispatch to optimal kernel for detected CPU architecture.
// Override arch for testing.
inline void ternary_linear(const LayerData& layer,
                            const float* input,
                            float* output,
                            CPUArch override_arch) {
    const int max_terms = 32;
    const uint32_t* term_data[32];
    int alpha_exps[32];

    int n_active = extract_term_data(layer, term_data, alpha_exps, max_terms);
    if (n_active == 0) {
        std::fill(output, output + layer.out_features, 0.0f);
        return;
    }

    CPUArch arch = (override_arch != CPUArch::UNKNOWN) ? override_arch : detect_cpu_arch();

    switch (arch) {
#if defined(__x86_64__) || defined(_M_X64)
        case CPUArch::X86_64_AVX512:
            ternary_mul_avx512(term_data, alpha_exps, n_active,
                               layer.out_features, layer.in_features, input, output);
            return;
        case CPUArch::X86_64_AVX2:
            ternary_mul_avx2(term_data, alpha_exps, n_active,
                             layer.out_features, layer.in_features, input, output);
            return;
        case CPUArch::X86_64_AVX:
            ternary_mul_avx(term_data, alpha_exps, n_active,
                            layer.out_features, layer.in_features, input, output);
            return;
        case CPUArch::X86_64_SSE42:
            ternary_mul_sse42(term_data, alpha_exps, n_active,
                              layer.out_features, layer.in_features, input, output);
            return;
#endif
#if defined(__aarch64__)
        case CPUArch::ARM64_NEON:
            ternary_mul_neon(term_data, alpha_exps, n_active,
                             layer.out_features, layer.in_features, input, output);
            return;
#endif
        default:
            ternary_mul_scalar(term_data, alpha_exps, n_active,
                               layer.out_features, layer.in_features, input, output);
            return;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-KERNEL VALIDATOR (for testing)
// ═══════════════════════════════════════════════════════════════════════════
// Runs all available kernels on the same input and compares outputs.
// Returns max absolute error across all outputs.
inline float validate_all_kernels(const LayerData& layer,
                                   const float* input,
                                   float* output_reference,
                                   const char* reference_name) {
    struct KernelInfo { CPUArch arch; const char* name; };
    KernelInfo all[] = {
        {CPUArch::X86_64_SCALAR, "scalar"},
#if defined(__x86_64__) || defined(_M_X64)
        {CPUArch::X86_64_SSE42,  "sse42"},
        {CPUArch::X86_64_AVX,    "avx"},
        {CPUArch::X86_64_AVX2,   "avx2"},
        {CPUArch::X86_64_AVX512, "avx512"},
#endif
#if defined(__aarch64__)
        {CPUArch::ARM64_NEON,    "neon"},
#endif
    };

    const int max_terms = 32;
    const uint32_t* term_data[32];
    int alpha_exps[32];
    int n_active = extract_term_data(layer, term_data, alpha_exps, max_terms);
    if (n_active == 0) return 0.0f;

    int out_f = layer.out_features;
    int in_f = layer.in_features;
    std::vector<float> ref(out_f);

    // Compute reference
    ternary_mul_scalar(term_data, alpha_exps, n_active, out_f, in_f, input, ref.data());

    float max_err = 0.0f;
    for (auto& k : all) {
        if (k.arch == CPUArch::X86_64_SCALAR) continue; // skip self
        std::vector<float> test(out_f);
        switch (k.arch) {
#if defined(__x86_64__) || defined(_M_X64)
            case CPUArch::X86_64_SSE42:
                ternary_mul_sse42(term_data, alpha_exps, n_active, out_f, in_f, input, test.data()); break;
            case CPUArch::X86_64_AVX:
                ternary_mul_avx(term_data, alpha_exps, n_active, out_f, in_f, input, test.data()); break;
            case CPUArch::X86_64_AVX2:
                ternary_mul_avx2(term_data, alpha_exps, n_active, out_f, in_f, input, test.data()); break;
            case CPUArch::X86_64_AVX512:
                ternary_mul_avx512(term_data, alpha_exps, n_active, out_f, in_f, input, test.data()); break;
#endif
#if defined(__aarch64__)
            case CPUArch::ARM64_NEON:
                ternary_mul_neon(term_data, alpha_exps, n_active, out_f, in_f, input, test.data()); break;
#endif
            default: continue;
        }
        float err = 0.0f;
        for (int i = 0; i < out_f; i++) {
            float d = std::abs(ref[i] - test[i]);
            if (d > err) err = d;
        }
        printf("  %-8s vs scalar: max_err = %8.2e  %s\n",
               k.name, err, err < 1e-4f ? "OK" : "MISMATCH!");
        if (err > max_err) max_err = err;

        // If caller provided reference output buffer, fill it
        if (output_reference && strcmp(k.name, reference_name) == 0) {
            std::copy(test.begin(), test.end(), output_reference);
        }
    }
    return max_err;
}

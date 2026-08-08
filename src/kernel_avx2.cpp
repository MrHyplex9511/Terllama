/*
 * kernel_avx2.cpp: AVX2+FMA 256-bit SIMD ternary kernels
 *
 * Compile: g++ -c -O3 -mavx2 -mfma kernel_avx2.cpp
 *
 * Two kernels:
 *   ternary_mul_avx2()       bitplane combined[] format (backward compat)
 *   ternary_mul_avx2_blocks()  block-scaled direct path + INT8 quant + activation-parallel tiling
 */
#include "kernel_decl.h"
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>

// Horizontal sum of 8 int32 lanes (AVX2).
static inline int32_t hsum_epi32(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(2, 3, 0, 1)));
    lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, _MM_SHUFFLE(1, 0, 3, 2)));
    return _mm_cvtsi128_si32(lo);
}

// 256-entry LUT: 8-bit mask byte -> __m256i sign mask with lane i = -1 iff
// bit i of the byte is set. One vpshufb/load replaces the per-word scalar
// bit-test + _mm256_set_epi32 construction.
struct TernaryMaskTable {
    alignas(32) __m256i m[256];
    TernaryMaskTable() {
        for (int v = 0; v < 256; v++) {
            int a[8];
            for (int b = 0; b < 8; b++) a[b] = (v & (1 << b)) ? -1 : 0;
            m[v] = _mm256_set_epi32(a[7], a[6], a[5], a[4], a[3], a[2], a[1], a[0]);
        }
    }
};

// ═══════════════════════════════════════════════════════════════════════════
// BITPLANE KERNEL - bitplane combined[] format (backward compat)
// ═══════════════════════════════════════════════════════════════════════════
#if defined(__x86_64__) || defined(_M_X64)
void ternary_mul_avx2(const uint32_t* const* term_data, const int* alpha_exps,
                      int n_active, int out_f, int in_f,
                      const float* input, float* output) {
    int words_per_row = (in_f + 15) / 16;
    int rem = in_f % 16;
    int full_words = rem > 0 ? words_per_row - 1 : words_per_row;
    uint32_t tail_mask = rem > 0 ? (uint32_t)((1 << rem) - 1) : 0;
    size_t stride = (size_t)words_per_row;

    // alpha_exps is constant per call: precompute 2^alpha once, multiply
    // instead of a std::ldexp libm call per (row, term).
    float alpha_scale[32];
    for (int t = 0; t < n_active; t++) alpha_scale[t] = std::ldexp(1.0f, alpha_exps[t]);

    // Skip OMP team creation for tiny GEMMs (fork/join cost dominates).
    #pragma omp parallel for if((int64_t)out_f * in_f > 65536)
    for (int i = 0; i < out_f; i++) {
        __m256 vacc[32];
        for (int t = 0; t < n_active; t++) vacc[t] = _mm256_setzero_ps();
        __m256 zero = _mm256_setzero_ps();
        static thread_local const TernaryMaskTable mask_tab;

        for (int w = 0; w < full_words; w++) {
            __m256 v0 = _mm256_loadu_ps(&input[w * 16 + 0]);
            __m256 v1 = _mm256_loadu_ps(&input[w * 16 + 8]);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + w];
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                { uint32_t abits = (nzw & ~negw) & 0xFF;
                  uint32_t sbits = (nzw & negw) & 0xFF;
                  __m256i addm = _mm256_load_si256(&mask_tab.m[abits]);
                  __m256i subm = _mm256_load_si256(&mask_tab.m[sbits]);
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(subm)));
                }
                { uint32_t abits = ((nzw & ~negw) >> 8) & 0xFF;
                  uint32_t sbits = ((nzw & negw) >> 8) & 0xFF;
                  __m256i addm = _mm256_load_si256(&mask_tab.m[abits]);
                  __m256i subm = _mm256_load_si256(&mask_tab.m[sbits]);
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v1, _mm256_castsi256_ps(subm)));
                }
            }
        }

        if (rem > 0) {
            __m256 v0 = _mm256_loadu_ps(&input[full_words * 16 + 0]);
            __m256 v1 = (rem > 8) ? _mm256_loadu_ps(&input[full_words * 16 + 8]) : _mm256_setzero_ps();

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + full_words] & (tail_mask | (tail_mask << 16));
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                { int chbits = rem > 8 ? 8 : rem;
                  uint32_t cm = (1 << chbits) - 1;
                  uint32_t abits = ((nzw & ~negw) >> 0) & cm;
                  uint32_t sbits = ((nzw & negw) >> 0) & cm;
                  __m256i addm = _mm256_load_si256(&mask_tab.m[abits]);
                  __m256i subm = _mm256_load_si256(&mask_tab.m[sbits]);
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(subm)));
                }
                if (rem > 8) {
                    int chbits = rem - 8;
                    uint32_t cm = (1 << chbits) - 1;
                    uint32_t abits = ((nzw & ~negw) >> 8) & cm;
                    uint32_t sbits = ((nzw & negw) >> 8) & cm;
                    __m256i addm = _mm256_load_si256(&mask_tab.m[abits]);
                    __m256i subm = _mm256_load_si256(&mask_tab.m[sbits]);
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
            result += s * alpha_scale[t];
        }
        output[i] = result;
    }
}
#endif // __x86_64__ || _M_X64

// ═══════════════════════════════════════════════════════════════════════════
// INT8 ACTIVATION QUANTIZATION
// ═══════════════════════════════════════════════════════════════════════════
// FP32 -> INT8 quantize, return scale.
inline float quantize_activations_to_i8(const float* x, int n, int8_t* x_q) {
    // Vectorized abs-max scan (8 lanes at a time).
    __m256 vmax = _mm256_setzero_ps();
    const __m256 sign_mask = _mm256_set1_ps(-0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&x[i]);
        vmax = _mm256_max_ps(vmax, _mm256_andnot_ps(sign_mask, v));
    }
    float max_val = 0.0f;
    for (; i < n; i++) max_val = std::max(max_val, std::abs(x[i]));
    __m128 lo = _mm256_castps256_ps128(vmax), hi = _mm256_extractf128_ps(vmax, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(2, 3, 0, 1)));
    lo = _mm_max_ps(lo, _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 3, 2)));
    float simd_max = _mm_cvtss_f32(lo);
    if (simd_max > max_val) max_val = simd_max;

    if (max_val < 1e-10f) {
        std::memset(x_q, 0, n);
        return 1.0f;
    }
    float scale = max_val / 127.0f;

    // Vectorized quantize. _mm256_cvttps_epi32 truncates toward zero,
    // matching the (int) cast; clamp to [-128, 127] then pack to int8.
    const __m256 svec = _mm256_set1_ps(scale);
    const __m256i lo_c = _mm256_set1_epi32(-128);
    const __m256i hi_c = _mm256_set1_epi32(127);
    i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 f = _mm256_div_ps(_mm256_loadu_ps(&x[i]), svec);
        __m256i v = _mm256_cvttps_epi32(f);
        v = _mm256_max_epi32(v, lo_c);
        v = _mm256_min_epi32(v, hi_c);
        // Pack low+high 128-bit halves (both already clamped int8 values).
        __m128i lo = _mm256_castsi256_si128(v);
        __m128i hi = _mm256_extracti128_si256(v, 1);
        __m128i i16 = _mm_packs_epi32(lo, hi);
        __m128i i8v = _mm_packs_epi16(i16, i16);
        _mm_storel_epi64((__m128i*)&x_q[i], i8v);
    }
    for (; i < n; i++) {
        int v = (int)(x[i] / scale);
        x_q[i] = (int8_t)std::clamp(v, -128, 127);
    }
    return scale;
}

// ═══════════════════════════════════════════════════════════════════════════
// BLOCK-SCALED TERNARY DECODE (AVX2)
// ═══════════════════════════════════════════════════════════════════════════
// 128 packed weights -> int8 ternary
inline void decode_block_ternary_avx2(const uint8_t* packed, int8_t* ternary) {
    // 32 bytes -> 128 ternary values, 4 per byte
    for (int i = 0; i < 32; i++) {
        uint8_t byte = packed[i];
        ternary[i*4 + 0] = (int8_t)(((byte >> 6) & 0x03) - 1);
        ternary[i*4 + 1] = (int8_t)(((byte >> 4) & 0x03) - 1);
        ternary[i*4 + 2] = (int8_t)(((byte >> 2) & 0x03) - 1);
        ternary[i*4 + 3] = (int8_t)((byte & 0x03) - 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// BLOCK-SCALED KERNEL - ACTIVATION-PARALLEL TILING (AVX2)
// ═══════════════════════════════════════════════════════════════════════════
// 128 columns per tile. INT8 quantize activations, decode weights to int8,
// then INT8 dot-product with per-block scale dequant.
#define COL_BLOCK_SIZE 128

void ternary_mul_avx2_blocks(const uint8_t* const* const* term_block_data,
                             const float* const* const* term_block_scales,
                             int n_terms, int out_f, int in_f, int n_blocks,
                             const float* input, float* output) {
    // INT8 quantized activation buffer
    int8_t* input_i8 = (int8_t*)alloca(in_f * sizeof(int8_t));
    int8_t* decoded_w = (int8_t*)alloca(COL_BLOCK_SIZE * sizeof(int8_t));

    // FP32 activations -> INT8
    float act_scale = quantize_activations_to_i8(input, in_f, input_i8);
    // Precompute reciprocal once; per-row divide becomes a multiply.
    float inv_act_scale = 1.0f / act_scale;

    // Skip OMP team creation for tiny GEMMs (fork/join cost dominates).
    #pragma omp parallel for if((int64_t)out_f * in_f > 65536)
    for (int row = 0; row < out_f; row++) {
        float sum = 0.0f;

        for (int t = 0; t < n_terms; t++) {
            const uint8_t* row_data = term_block_data[t][row];
            const float* row_scales = term_block_scales[t][row];

            for (int b = 0; b < n_blocks; b++) {
                const uint8_t* packed = row_data + b * (COL_BLOCK_SIZE / 4);
                float w_scale = row_scales[b];
                int block_start = b * COL_BLOCK_SIZE;
                int block_end = std::min(block_start + COL_BLOCK_SIZE, in_f);
                int block_size = block_end - block_start;

                // Decode codes -> int8
                decode_block_ternary_avx2(packed, decoded_w);

                // INT8 dot product via _mm256_maddubs_epi16 (4x lanes).
                // maddubs treats operand a as UNSIGNED bytes; activations can
                // be negative so bias them by +128 and subtract 128*sum(w).
                // All arithmetic is exact int32, so results stay bit-identical
                // to the scalar dot (block_size < 32 falls through to scalar).
                const __m256i ones8  = _mm256_set1_epi8(1);
                const __m256i ones16 = _mm256_set1_epi16(1);
                const __m256i bias   = _mm256_set1_epi8(-128); // +128 unsigned
                __m256i dot32 = _mm256_setzero_si256();
                __m256i sw32  = _mm256_setzero_si256();
                int j = 0;
                for (; j + 32 <= block_size; j += 32) {
                    __m256i u = _mm256_loadu_si256((const __m256i*)&input_i8[block_start + j]);
                    u = _mm256_add_epi8(u, bias);
                    __m256i w = _mm256_loadu_si256((const __m256i*)&decoded_w[j]);
                    __m256i p  = _mm256_maddubs_epi16(u, w);       // (a+128)*w pairs
                    __m256i pw = _mm256_maddubs_epi16(ones8, w);   // sum of w per pair
                    dot32 = _mm256_add_epi32(dot32, _mm256_madd_epi16(p, ones16));
                    sw32  = _mm256_add_epi32(sw32, _mm256_madd_epi16(pw, ones16));
                }
                int32_t dot = hsum_epi32(dot32) - 128 * hsum_epi32(sw32);
                for (; j < block_size; j++)
                    dot += (int32_t)input_i8[block_start + j] * (int32_t)decoded_w[j];
                sum += (float)dot * w_scale;
            }
        }
        output[row] = sum * inv_act_scale;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FAIRYFUSE AVX-512 BLOCK-SCALED KERNEL
// ═══════════════════════════════════════════════════════════════════════════
// Key insight: process ternary weights directly without INT8 quantization.
// Decode 2-bit codes (0=-1, 1=0, 2=+1) to AVX-512 masks and use
// _mm512_mask_add_ps / _mm512_mask_sub_ps — zero multiplications in loop.
//
// 16 input elements processed per AVX-512 iteration.  4 bytes of packed
// data provide 16 ternary codes.  No intermediate int8 decode needed.
// ~3× fewer memory ops vs old decode+dot approach.
#if defined(__x86_64__) || defined(_M_X64)
// Reverse the order of the four 2-bit ternary codes within a byte.
// The pack contract (loader.h, pack_weights.py) is MSB-first:
//   elem0 at bits 7:6 ... elem3 at bits 1:0.
// On little-endian, pext pulls bit 0 (elem3) out first, so codes must be
// reordered to ascending element order before the pext below, otherwise each
// 4-element group comes out reversed and weights are scrambled.
static inline uint8_t reverse_2bit_codes(uint8_t x) {
    return (uint8_t)(((x & 0x03u) << 6) |
                     ((x & 0x0Cu) << 2) |
                     ((x & 0x30u) >> 2) |
                     ((x & 0xC0u) >> 6));
}

__attribute__((target("avx512f,avx512dq,bmi2")))
void ternary_mul_avx512_fairyfuse(
    const uint8_t* const* const* term_block_data,
    const float* const* const* term_block_scales,
    int n_terms, int out_f, int in_f, int n_blocks,
    const float* input, float* output) {

    // Skip OMP team creation for tiny GEMMs (fork/join cost dominates).
    #pragma omp parallel for if((int64_t)out_f * in_f > 65536)
    for (int row = 0; row < out_f; row++) {
        __m512 vacc = _mm512_setzero_ps();

        for (int t = 0; t < n_terms; t++) {
            const uint8_t* row_data = term_block_data[t][row];
            const float* row_scales = term_block_scales[t][row];

            for (int b = 0; b < n_blocks; b++) {
                const uint8_t* packed = row_data + b * (COL_BLOCK_SIZE / 4);
                float w_scale = row_scales[b];
                int block_start = b * COL_BLOCK_SIZE;
                int block_size = std::min(COL_BLOCK_SIZE, in_f - block_start);

                // Accumulate block in FP32, apply w_scale at block end.
                __m512 bacc = _mm512_setzero_ps();
                int nb = (block_size + 15) / 16;

                for (int j = 0; j < nb; j++) {
                    int base = j * 16;
                    int rem = block_size - base;
                    int chunk = rem > 16 ? 16 : rem;

                    // Load 4 bytes from packed block data
                    uint32_t codes32;
                    std::memcpy(&codes32, &packed[j * 4], 4);
                    // Reorder each byte to ascending element order
                    codes32 = (uint32_t)reverse_2bit_codes((uint8_t)(codes32 & 0xFF)) |
                              ((uint32_t)reverse_2bit_codes((uint8_t)((codes32 >> 8) & 0xFF)) << 8) |
                              ((uint32_t)reverse_2bit_codes((uint8_t)((codes32 >> 16) & 0xFF)) << 16) |
                              ((uint32_t)reverse_2bit_codes((uint8_t)((codes32 >> 24) & 0xFF)) << 24);

                    // Decode 16 codes into add/sub masks in one go via BMI2 pext.
                    uint32_t lo_bits = _pext_u32(codes32, 0x55555555u);
                    uint32_t hi_bits = _pext_u32(codes32, 0xAAAAAAAAu);
                    uint16_t add_mask = (uint16_t)(hi_bits & ~lo_bits);
                    uint16_t sub_mask = (uint16_t)(~(hi_bits | lo_bits) & 0xFFFFu);

                    // Mask the load for the last chunk to avoid reading past end
                    __mmask16 ld_mask = (chunk == 16) ? (__mmask16)-1
                                                      : (__mmask16)((1 << chunk) - 1);
                    __m512 zero = _mm512_setzero_ps();
                    __m512 vin  = _mm512_mask_loadu_ps(zero, ld_mask, &input[block_start + base]);
                    __m512 tmp  = _mm512_mask_add_ps(zero, (__mmask16)add_mask, zero, vin);
                    tmp         = _mm512_mask_sub_ps(tmp,  (__mmask16)sub_mask, zero, vin);
                    bacc = _mm512_add_ps(bacc, tmp);
                }

                // Fused multiply-add: vacc += w_scale * bacc
                vacc = _mm512_fmadd_ps(_mm512_set1_ps(w_scale), bacc, vacc);
            }
        }

        output[row] = _mm512_reduce_add_ps(vacc);
    }
}
#endif

// ═══════════════════════════════════════════════════════════════════════════
// SCALAR BLOCK-SCALED KERNEL (for validation)
// ═══════════════════════════════════════════════════════════════════════════
void ternary_mul_scalar_blocks(const uint8_t* const* const* term_block_data,
                               const float* const* const* term_block_scales,
                               int n_terms, int out_f, int in_f, int n_blocks,
                               const float* input, float* output) {
    #pragma omp parallel for
    for (int row = 0; row < out_f; row++) {
        float sum = 0.0f;
        for (int t = 0; t < n_terms; t++) {
            const uint8_t* row_data = term_block_data[t][row];
            const float* row_scales = term_block_scales[t][row];
            for (int b = 0; b < n_blocks; b++) {
                const uint8_t* packed = row_data + b * (COL_BLOCK_SIZE / 4);
                float w_scale = row_scales[b];
                int block_start = b * COL_BLOCK_SIZE;
                int block_end = std::min(block_start + COL_BLOCK_SIZE, in_f);
                int block_size = block_end - block_start;

                // Decode codes and dot product
                float block_sum = 0.0f;
                for (int j = 0; j < block_size; j += 4) {
                    uint8_t byte = packed[j / 4];
                    int8_t t0 = ((byte >> 6) & 0x03) - 1;
                    int8_t t1 = ((byte >> 4) & 0x03) - 1;
                    int8_t t2 = ((byte >> 2) & 0x03) - 1;
                    int8_t t3 = (byte & 0x03) - 1;
                    if (j+0 < block_size) block_sum += input[block_start + j + 0] * t0;
                    if (j+1 < block_size) block_sum += input[block_start + j + 1] * t1;
                    if (j+2 < block_size) block_sum += input[block_start + j + 2] * t2;
                    if (j+3 < block_size) block_sum += input[block_start + j + 3] * t3;
                }
                sum += block_sum * w_scale;
            }
        }
        output[row] = sum;
    }
}

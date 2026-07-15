/*
 * kernel_avx2.cpp: AVX2+FMA 256-bit SIMD ternary kernels
 *
 * Compile: g++ -c -O3 -mavx2 -mfma kernel_avx2.cpp
 *
 * Two kernels:
 *   ternary_mul_avx2()       bitplane combined[] format (backward compat)
 *   ternary_mul_avx2_i2s()   I2_S direct path + INT8 quant + activation-parallel tiling
 */
#include "kernel_decl.h"
#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdint>

// ═══════════════════════════════════════════════════════════════════════════
// BITPLANE KERNEL - bitplane combined[] format (backward compat)
// ═══════════════════════════════════════════════════════════════════════════
void ternary_mul_avx2(const uint32_t* const* term_data, const int* alpha_exps,
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
        __m256 zero = _mm256_setzero_ps();

        for (int w = 0; w < full_words; w++) {
            __m256 v0 = _mm256_loadu_ps(&input[w * 16 + 0]);
            __m256 v1 = _mm256_loadu_ps(&input[w * 16 + 8]);

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + w];
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                { uint32_t abits = (nzw & ~negw) & 0xFF;
                  uint32_t sbits = (nzw & negw) & 0xFF;
                  int a[8], sb[8];
                  for (int b = 0; b < 8; b++) {
                      a[b] = (abits & (1 << b)) ? -1 : 0;
                      sb[b] = (sbits & (1 << b)) ? -1 : 0;
                  }
                  __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                  __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
                  vacc[t] = _mm256_add_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(addm)));
                  vacc[t] = _mm256_sub_ps(vacc[t], _mm256_blendv_ps(zero, v0, _mm256_castsi256_ps(subm)));
                }
                { uint32_t abits = ((nzw & ~negw) >> 8) & 0xFF;
                  uint32_t sbits = ((nzw & negw) >> 8) & 0xFF;
                  int a[8], sb[8];
                  for (int b = 0; b < 8; b++) {
                      a[b] = (abits & (1 << b)) ? -1 : 0;
                      sb[b] = (sbits & (1 << b)) ? -1 : 0;
                  }
                  __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                  __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
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
                  int a[8]={0}, sb[8]={0};
                  for (int b = 0; b < chbits; b++) {
                      a[b] = (abits & (1 << b)) ? -1 : 0;
                      sb[b] = (sbits & (1 << b)) ? -1 : 0;
                  }
                  __m256i addm = _mm256_set_epi32(a[7],a[6],a[5],a[4],a[3],a[2],a[1],a[0]);
                  __m256i subm = _mm256_set_epi32(sb[7],sb[6],sb[5],sb[4],sb[3],sb[2],sb[1],sb[0]);
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

// ═══════════════════════════════════════════════════════════════════════════
// INT8 ACTIVATION QUANTIZATION
// ═══════════════════════════════════════════════════════════════════════════
// FP32 -> INT8 quantize, return scale.
inline float quantize_activations_to_i8(const float* x, int n, int8_t* x_q) {
    float max_val = 0.0f;
    for (int i = 0; i < n; i++) max_val = std::max(max_val, std::abs(x[i]));
    if (max_val < 1e-10f) {
        std::memset(x_q, 0, n);
        return 1.0f;
    }
    float scale = max_val / 127.0f;
    for (int i = 0; i < n; i++) {
        int v = (int)(x[i] / scale);
        x_q[i] = (int8_t)std::clamp(v, -128, 127);
    }
    return scale;
}

// ═══════════════════════════════════════════════════════════════════════════
// I2_S DECODE (AVX2)
// ═══════════════════════════════════════════════════════════════════════════
// 128 I2_S packed weights -> int8 ternary
inline void decode_i2s_block_avx2(const uint8_t* packed, int8_t* ternary) {
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
// I2_S KERNEL - ACTIVATION-PARALLEL TILING
// ═══════════════════════════════════════════════════════════════════════════
// 128 columns per tile. Unpack weights once, reuse across 4 rows.
#define COL_BLOCK_SIZE 128
#define ROW_BLOCK_SIZE 4

void ternary_mul_avx2_i2s(const uint8_t* const* i2s_block_data,
                          const float* const* i2s_block_scales,
                          int out_f, int in_f, int n_blocks,
                          const float* input, float* output) {
    // INT8 quantized activation buffer
    int8_t* input_i8 = (int8_t*)alloca(in_f * sizeof(int8_t));
    int8_t* decoded_w = (int8_t*)alloca(COL_BLOCK_SIZE * sizeof(int8_t));

    // FP32 activations -> INT8
    float act_scale = quantize_activations_to_i8(input, in_f, input_i8);

    #pragma omp parallel for
    for (int row = 0; row < out_f; row++) {
        const uint8_t* row_data = i2s_block_data[row];  // row's packed I2_S data
        const float* row_scales = i2s_block_scales[row];  // row's per-block scales
        float sum = 0.0f;

        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* packed = row_data + b * (COL_BLOCK_SIZE / 4);
            float w_scale = row_scales[b];
            int block_start = b * COL_BLOCK_SIZE;
            int block_end = std::min(block_start + COL_BLOCK_SIZE, in_f);
            int block_size = block_end - block_start;

            // I2_S decode
            decode_i2s_block_avx2(packed, decoded_w);

            // INT8 dot product for this block
            int32_t dot = 0;
            for (int j = 0; j < block_size; j++) {
                dot += (int32_t)input_i8[block_start + j] * (int32_t)decoded_w[j];
            }

            // Dequantize: apply per-block scale
            sum += (float)dot * w_scale;
        }

        // Dequantize activation scale
        output[row] = sum / act_scale;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SCALAR I2_S KERNEL (for validation)
// ═══════════════════════════════════════════════════════════════════════════
void ternary_mul_scalar_i2s(const uint8_t* const* i2s_block_data,
                            const float* const* i2s_block_scales,
                            int out_f, int in_f, int n_blocks,
                            const float* input, float* output) {
    #pragma omp parallel for
    for (int row = 0; row < out_f; row++) {
        const uint8_t* row_data = i2s_block_data[row];
        const float* row_scales = i2s_block_scales[row];
        float sum = 0.0f;
        for (int b = 0; b < n_blocks; b++) {
            const uint8_t* packed = row_data + b * (COL_BLOCK_SIZE / 4);
            float w_scale = row_scales[b];
            int block_start = b * COL_BLOCK_SIZE;
            int block_end = std::min(block_start + COL_BLOCK_SIZE, in_f);
            int block_size = block_end - block_start;

            // Decode I2_S and dot product
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
        output[row] = sum;
    }
}

/*
 * kernel_neon.cpp: ARM64 NEON (128-bit SIMD) ternary kernel
 *
 * Compile: g++ -c -O3 -march=armv8-a+fp+simd kernel_neon.cpp
 *
 * Emulate masked add/sub via bitwise AND with mask. NEON has no
 * hardware masking like AVX-512. Process 16 input elements as
 * 4 × float32x4_t chunks.
 */
#include "kernel_decl.h"
#include <arm_neon.h>
#include <cmath>
#include <algorithm>
#include <cstring>

void ternary_mul_neon(const uint32_t* const* term_data, const int* alpha_exps,
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
                { uint32_t abits = (nzw & ~negw) & 0xF;
                  uint32_t sbits = (nzw & negw) & 0xF;
                  uint32_t addm[4] = {abits&1?0xFFFFFFFF:0, abits&2?0xFFFFFFFF:0,
                                       abits&4?0xFFFFFFFF:0, abits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sbits&1?0xFFFFFFFF:0, sbits&2?0xFFFFFFFF:0,
                                       sbits&4?0xFFFFFFFF:0, sbits&8?0xFFFFFFFF:0};
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(addm), vreinterpretq_u32_f32(v0))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(subm), vreinterpretq_u32_f32(v0))));
                }
                // Chunk 1: bits [4..7]
                { uint32_t abits = ((nzw & ~negw) >> 4) & 0xF;
                  uint32_t sbits = ((nzw & negw) >> 4) & 0xF;
                  uint32_t addm[4] = {abits&1?0xFFFFFFFF:0, abits&2?0xFFFFFFFF:0,
                                       abits&4?0xFFFFFFFF:0, abits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sbits&1?0xFFFFFFFF:0, sbits&2?0xFFFFFFFF:0,
                                       sbits&4?0xFFFFFFFF:0, sbits&8?0xFFFFFFFF:0};
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(addm), vreinterpretq_u32_f32(v1))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(subm), vreinterpretq_u32_f32(v1))));
                }
                // Chunk 2: bits [8..11]
                { uint32_t abits = ((nzw & ~negw) >> 8) & 0xF;
                  uint32_t sbits = ((nzw & negw) >> 8) & 0xF;
                  uint32_t addm[4] = {abits&1?0xFFFFFFFF:0, abits&2?0xFFFFFFFF:0,
                                       abits&4?0xFFFFFFFF:0, abits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sbits&1?0xFFFFFFFF:0, sbits&2?0xFFFFFFFF:0,
                                       sbits&4?0xFFFFFFFF:0, sbits&8?0xFFFFFFFF:0};
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(addm), vreinterpretq_u32_f32(v2))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(subm), vreinterpretq_u32_f32(v2))));
                }
                // Chunk 3: bits [12..15]
                { uint32_t abits = ((nzw & ~negw) >> 12) & 0xF;
                  uint32_t sbits = ((nzw & negw) >> 12) & 0xF;
                  uint32_t addm[4] = {abits&1?0xFFFFFFFF:0, abits&2?0xFFFFFFFF:0,
                                       abits&4?0xFFFFFFFF:0, abits&8?0xFFFFFFFF:0};
                  uint32_t subm[4] = {sbits&1?0xFFFFFFFF:0, sbits&2?0xFFFFFFFF:0,
                                       sbits&4?0xFFFFFFFF:0, sbits&8?0xFFFFFFFF:0};
                  vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(addm), vreinterpretq_u32_f32(v3))));
                  vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(subm), vreinterpretq_u32_f32(v3))));
                }
            }
        }

        if (rem > 0) {
            float32x4_t vv[4] = {
                vld1q_f32(&input[full_words * 16 + 0]),
                (rem > 4)  ? vld1q_f32(&input[full_words * 16 + 4])  : vdupq_n_f32(0.0f),
                (rem > 8)  ? vld1q_f32(&input[full_words * 16 + 8])  : vdupq_n_f32(0.0f),
                (rem > 12) ? vld1q_f32(&input[full_words * 16 + 12]) : vdupq_n_f32(0.0f),
            };

            for (int t = 0; t < n_active; t++) {
                uint32_t c = term_data[t][i * stride + full_words] & (tail_mask | (tail_mask << 16));
                uint16_t nzw = c >> 16;
                uint16_t negw = c & 0xFFFF;

                for (int ch = 0; ch < 4; ch++) {
                    int sh = ch * 4;
                    int chbits = rem - sh;
                    if (chbits <= 0) break;
                    if (chbits > 4) chbits = 4;
                    uint32_t cm = (1 << chbits) - 1;
                    uint32_t abits = ((nzw & ~negw) >> sh) & cm;
                    uint32_t sbits = ((nzw & negw) >> sh) & cm;
                    uint32_t addm[4] = {(chbits>0&&abits&1)?0xFFFFFFFF:0,
                                        (chbits>1&&abits&2)?0xFFFFFFFF:0,
                                        (chbits>2&&abits&4)?0xFFFFFFFF:0,
                                        (chbits>3&&abits&8)?0xFFFFFFFF:0};
                    uint32_t subm[4] = {(chbits>0&&sbits&1)?0xFFFFFFFF:0,
                                        (chbits>1&&sbits&2)?0xFFFFFFFF:0,
                                        (chbits>2&&sbits&4)?0xFFFFFFFF:0,
                                        (chbits>3&&sbits&8)?0xFFFFFFFF:0};
                    vacc[t] = vaddq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(addm), vreinterpretq_u32_f32(vv[ch]))));
                    vacc[t] = vsubq_f32(vacc[t], vreinterpretq_f32_u32(vandq_u32(vld1q_u32(subm), vreinterpretq_u32_f32(vv[ch]))));
                }
            }
        }

        float result = 0.0f;
        for (int t = 0; t < n_active; t++) {
            result += std::ldexp(vaddvq_f32(vacc[t]), alpha_exps[t]);
        }
        output[i] = result;
    }
}

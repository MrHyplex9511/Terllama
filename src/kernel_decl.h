/*
 * kernel_decl.h — Ternary kernel function declarations
 *
 * Compile each arch .cpp with its own -march flags.
 * dispatcher.cpp selects the best one at runtime.
 */
#pragma once
#include "model.h"
#include <cstdint>

// Kernel interface:
//   term_data[t] = pointer to combined[] array for term t
//   alpha_exps[t] = alpha exponent for term t
//   n_active = number of active terms
//   out_f = number of output neurons (rows)
//   in_f = number of input elements (columns)
//   input = input vector (in_f floats)
//   output = output vector (out_f floats)

void ternary_mul_scalar(const uint32_t* const* term_data, const int* alpha_exps,
                        int n_active, int out_f, int in_f,
                        const float* input, float* output);

#if defined(__x86_64__) || defined(_M_X64)
void ternary_mul_avx2(const uint32_t* const* term_data, const int* alpha_exps,
                      int n_active, int out_f, int in_f,
                      const float* input, float* output);
#elif defined(__aarch64__) || defined(_M_ARM64)
void ternary_mul_neon(const uint32_t* const* term_data, const int* alpha_exps,
                      int n_active, int out_f, int in_f,
                      const float* input, float* output);
#endif

int extract_term_data(const LayerData& layer,
                      const uint32_t** term_data_out,
                      int* alpha_exps_out, int max_terms);

void ternary_mul_avx2_i2s(const uint8_t* const* i2s_block_data,
                          const float* const* i2s_block_scales,
                          int out_f, int in_f, int n_blocks,
                          const float* input, float* output);
void ternary_mul_scalar_i2s(const uint8_t* const* i2s_block_data,
                            const float* const* i2s_block_scales,
                            int out_f, int in_f, int n_blocks,
                            const float* input, float* output);

CPUArch detect_cpu_arch();
void ternary_linear(const LayerData& layer, const float* input, float* output,
                    CPUArch override_arch = CPUArch::UNKNOWN);
void ternary_linear_i2s(const LayerData& layer, const float* input, float* output,
                        CPUArch override_arch = CPUArch::UNKNOWN);

float validate_all_kernels(const LayerData& layer, const float* input,
                           float* output_reference = nullptr,
                           const char* reference_name = "scalar");

// ─── MoTE kernel ───────────────────────────────────────────────────────────
void mote_ternary_linear(const MoTELayerData& mote, const float* x,
                          float* output, int hidden_size,
                          int intermediate_size);

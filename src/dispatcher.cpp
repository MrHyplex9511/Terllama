/*
 * dispatcher.cpp — CPU detection + kernel dispatch
 *
 * Weak symbols let the linker skip missing kernel .o files.
 * Null function pointer → skip.
 */
#include "kernel_decl.h"
#include "model.h"
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio>

// Weak declarations: nullptr if kernel .o not linked
#if defined(__x86_64__) || defined(_M_X64)
__attribute__((weak)) void ternary_mul_avx2(const uint32_t* const*, const int*, int, int, int, const float*, float*);
#elif defined(__aarch64__) || defined(_M_ARM64)
__attribute__((weak)) void ternary_mul_neon(const uint32_t* const*, const int*, int, int, int, const float*, float*);
#endif

CPUArch detect_cpu_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            if (ternary_mul_avx2) return CPUArch::X86_64_AVX2;
    #endif
    return CPUArch::X86_64_SCALAR;
#elif defined(__aarch64__) || defined(_M_ARM64)
    return ternary_mul_neon ? CPUArch::ARM64_NEON : CPUArch::ARM64_SCALAR;
#else
    return CPUArch::X86_64_SCALAR;
#endif
}

int extract_term_data(const LayerData& layer,
                      const uint32_t** term_data_out,
                      int* alpha_exps_out, int max_terms) {
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

void ternary_linear(const LayerData& layer, const float* input, float* output,
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
        case CPUArch::X86_64_AVX2:
            if (ternary_mul_avx2) { ternary_mul_avx2(term_data, alpha_exps, n_active, layer.out_features, layer.in_features, input, output); return; }
            break;
#elif defined(__aarch64__) || defined(_M_ARM64)
        case CPUArch::ARM64_NEON:
            if (ternary_mul_neon) { ternary_mul_neon(term_data, alpha_exps, n_active, layer.out_features, layer.in_features, input, output); return; }
            break;
#endif
        case CPUArch::X86_64_SCALAR:
        case CPUArch::ARM64_SCALAR:
        default:
            ternary_mul_scalar(term_data, alpha_exps, n_active,
                               layer.out_features, layer.in_features, input, output);
            return;
    }
    ternary_mul_scalar(term_data, alpha_exps, n_active,
                       layer.out_features, layer.in_features, input, output);
}

__attribute__((weak)) void ternary_mul_avx2_i2s(
    const uint8_t* const*, const float* const*, int, int, int, const float*, float*);
__attribute__((weak)) void ternary_mul_scalar_i2s(
    const uint8_t* const*, const float* const*, int, int, int, const float*, float*);

void ternary_linear_i2s(const LayerData& layer, const float* input, float* output,
                        CPUArch override_arch) {
    if (!layer.has_i2s || layer.i2s_blocks.empty() || layer.in_features == 0) {
        ternary_linear(layer, input, output, override_arch);
        return;
    }

    int n_blocks = (layer.in_features + layer.i2s_qk - 1) / layer.i2s_qk;

    std::vector<const uint8_t*> block_data(layer.i2s_blocks.size());
    std::vector<const float*> block_scales(layer.i2s_blocks.size());
    for (size_t i = 0; i < layer.i2s_blocks.size(); i++) {
        block_data[i] = layer.i2s_blocks[i].packed.data();
        block_scales[i] = &layer.i2s_blocks[i].scale;
    }

    CPUArch arch = (override_arch != CPUArch::UNKNOWN) ? override_arch : detect_cpu_arch();

#if defined(__x86_64__) || defined(_M_X64)
    if (arch == CPUArch::X86_64_AVX2) {
        if (ternary_mul_avx2_i2s) {
            ternary_mul_avx2_i2s(block_data.data(), block_scales.data(),
                                 layer.out_features, layer.in_features, n_blocks,
                                 input, output);
            return;
        }
    }
#endif
    if (ternary_mul_scalar_i2s) {
        ternary_mul_scalar_i2s(block_data.data(), block_scales.data(),
                               layer.out_features, layer.in_features, n_blocks,
                               input, output);
        return;
    }
    ternary_linear(layer, input, output, override_arch);
}

float validate_all_kernels(const LayerData& layer, const float* input,
                           float* output_reference, const char* reference_name) {
    struct K { CPUArch arch; const char* name; void (*func)(); };
    K trials[] = {
#if defined(__x86_64__) || defined(_M_X64)
        {CPUArch::X86_64_AVX2, "avx2", (void(*)())ternary_mul_avx2},
#elif defined(__aarch64__) || defined(_M_ARM64)
        {CPUArch::ARM64_NEON,  "neon",  (void(*)())ternary_mul_neon},
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
    ternary_mul_scalar(term_data, alpha_exps, n_active, out_f, in_f, input, ref.data());

    float max_err = 0.0f;
    for (auto& k : trials) {
        if (!k.func) continue;
        std::vector<float> test(out_f);
        using KFn = void(*)(const uint32_t* const*, const int*, int, int, int,
                            const float*, float*);
        reinterpret_cast<KFn>(k.func)(term_data, alpha_exps, n_active,
                                       out_f, in_f, input, test.data());

        float err = 0.0f;
        for (int i = 0; i < out_f; i++) {
            float d = std::abs(ref[i] - test[i]);
            if (d > err) err = d;
        }
        printf("  %-8s vs scalar: max_err = %8.2e  %s\n",
               k.name, err, err < 1e-4f ? "OK" : "MISMATCH!");
        if (err > max_err) max_err = err;

        if (output_reference && strcmp(k.name, reference_name) == 0) {
            std::copy(test.begin(), test.end(), output_reference);
        }
    }
    return max_err;
}

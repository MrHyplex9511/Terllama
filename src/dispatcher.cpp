/*
 * dispatcher.cpp — CPU detection + kernel dispatch
 *
 * Weak symbols let the linker skip missing kernel .o files.
 * Null function pointer → skip.
 */
#include "kernel_decl.h"
#include "model.h"
#include "core/logger.h"
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

// Use CPU detection from kernel_dispatch.h's inline variants for fine-grained ISA
// but keep the dispatch functions here for block-scaled and MoTE-specific logic
CPUArch detect_cpu_arch() {
#if defined(__x86_64__) || defined(_M_X64)
    #if defined(__GNUC__) || defined(__clang__)
        if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq"))
            return CPUArch::X86_64_AVX512;
        if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
            if (ternary_mul_avx2) return CPUArch::X86_64_AVX2;
        if (__builtin_cpu_supports("avx"))
            return CPUArch::X86_64_AVX;
        if (__builtin_cpu_supports("sse4.2"))
            return CPUArch::X86_64_SSE42;
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
        case CPUArch::X86_64_AVX512:
        case CPUArch::X86_64_AVX2:
            if (ternary_mul_avx2) { ternary_mul_avx2(term_data, alpha_exps, n_active, layer.out_features, layer.in_features, input, output); return; }
            break;
#elif defined(__aarch64__) || defined(_M_ARM64)
        case CPUArch::ARM64_NEON:
            if (ternary_mul_neon) { ternary_mul_neon(term_data, alpha_exps, n_active, layer.out_features, layer.in_features, input, output); return; }
            break;
#endif
        case CPUArch::X86_64_AVX:
        case CPUArch::X86_64_SSE42:
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

__attribute__((weak)) void ternary_mul_avx2_blocks(
    const uint8_t* const* const*, const float* const* const*, int, int, int, int, const float*, float*);
__attribute__((weak)) void ternary_mul_avx512_fairyfuse(
    const uint8_t* const* const*, const float* const* const*, int, int, int, int, const float*, float*);
__attribute__((weak)) void ternary_mul_scalar_blocks(
    const uint8_t* const* const*, const float* const* const*, int, int, int, int, const float*, float*);

void ternary_linear_blocks(const LayerData& layer, const float* input, float* output,
                           CPUArch override_arch) {
    if (!layer.has_blocks || layer.block_terms.empty() || layer.in_features == 0) {
        ternary_linear(layer, input, output, override_arch);
        return;
    }

    // Multi-term layers are a set of block-scaled term sets; single-term
    // layers are a 1-element set (all term sets share out_f × n_blocks shape).
    int n_terms = (int)layer.block_terms.size();
    int n_blocks = (layer.in_features + layer.block_qk - 1) / layer.block_qk;
    int codes_per_block = layer.block_qk / 4;  // 32 for qk=128

    // ─── Lazy contiguous layout cache ────────────────────────────────────
    // Build the packed/scale buffers once and reuse for every token.
    // Previously this rebuilt + re-memcpy'd the whole weight matrix on every
    // matmul call (211 calls/token, ~40MB of copies) — a major decode cost.
    // Multi-term: builds one contiguous set per term.
    if (!layer.term_cache_built) {
        layer.term_contig_data.resize(n_terms);
        layer.term_contig_scales.resize(n_terms);
        layer.term_block_data.resize(n_terms);
        layer.term_block_scales.resize(n_terms);

        for (int t = 0; t < n_terms; t++) {
            const std::vector<BlockTerm>& blocks = layer.block_terms[t];
            layer.term_contig_data[t].assign((size_t)layer.out_features * n_blocks * codes_per_block, 0);
            layer.term_contig_scales[t].assign((size_t)layer.out_features * n_blocks, 0.0f);
            layer.term_block_data[t].resize(layer.out_features);
            layer.term_block_scales[t].resize(layer.out_features);
            for (int row = 0; row < layer.out_features; row++) {
                for (int b = 0; b < n_blocks; b++) {
                    size_t idx = (size_t)row * n_blocks + b;
                    std::memcpy(&layer.term_contig_data[t][(size_t)row * n_blocks * codes_per_block + b * codes_per_block],
                                blocks[idx].packed.data(), (size_t)codes_per_block);
                    layer.term_contig_scales[t][(size_t)row * n_blocks + b] = blocks[idx].scale;
                }
                layer.term_block_data[t][row] = &layer.term_contig_data[t][(size_t)row * n_blocks * codes_per_block];
                layer.term_block_scales[t][row] = &layer.term_contig_scales[t][(size_t)row * n_blocks];
            }
        }
        layer.term_cache_built = true;
    }

    CPUArch arch = (override_arch != CPUArch::UNKNOWN) ? override_arch : detect_cpu_arch();

    // Build per-term pointer arrays for the kernel (term-major).
    // The kernels expect: term_block_data[t][row] → packed codes for row.
    const uint8_t* const* term_data[32];
    const float* const* term_scales[32];
    for (int t = 0; t < n_terms && t < 32; t++) {
        term_data[t] = layer.term_block_data[t].data();
        term_scales[t] = layer.term_block_scales[t].data();
    }

#if defined(__x86_64__) || defined(_M_X64)
    // FairyFuse AVX-512 path — zero-multiplication masked ternary kernel
    if (arch == CPUArch::X86_64_AVX512) {
        if (ternary_mul_avx512_fairyfuse) {
            ternary_mul_avx512_fairyfuse(term_data, term_scales,
                                          n_terms, layer.out_features, layer.in_features, n_blocks,
                                          input, output);
            return;
        }
    }
    if (arch == CPUArch::X86_64_AVX2) {
        if (ternary_mul_avx2_blocks) {
            ternary_mul_avx2_blocks(term_data, term_scales,
                                    n_terms, layer.out_features, layer.in_features, n_blocks,
                                    input, output);
            return;
        }
    }
#endif
    if (ternary_mul_scalar_blocks) {
        ternary_mul_scalar_blocks(term_data, term_scales,
                                  n_terms, layer.out_features, layer.in_features, n_blocks,
                                  input, output);
        return;
    }
    ternary_linear(layer, input, output, override_arch);
}

// ─── MoTE dispatch ─────────────────────────────────────────────────────────
void mote_linear_dispatch(const MoTELayerData& mote, const float* x,
                           float* output, int hidden_size,
                           int intermediate_size) {
    mote_ternary_linear(mote, x, output, hidden_size, intermediate_size);
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
        Logger::debug("  {} vs scalar: max_err = {}  {}", k.name, err, err < 1e-4f ? "OK" : "MISMATCH!");
        if (err > max_err) max_err = err;

        if (output_reference && strcmp(k.name, reference_name) == 0) {
            std::copy(test.begin(), test.end(), output_reference);
        }
    }
    return max_err;
}

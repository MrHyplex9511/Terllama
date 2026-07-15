// kernel_scalar.cpp — Generic C++ fallback (any CPU)
#include "kernel_decl.h"
#include <cmath>
#include <algorithm>
#include <cstring>

void ternary_mul_scalar(const uint32_t* const* term_data, const int* alpha_exps,
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

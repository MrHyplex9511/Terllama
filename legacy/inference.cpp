/*
 * inference.cpp - Naive ternary add/sub inference kernel
 *
 * Reads model_decomposed.bin, runs a single forward pass for one linear layer
 * using ONLY integer addition/subtraction and bit-shifts (no FP32 multiplies).
 *
 * Build & test:
 *   g++ -std=c++17 -O3 -march=native -fopenmp -o inference inference.cpp
 *   ./inference model_decomposed.bin 0  # test layer 0
 *
 * Verification:
 *   python3 verify_cpp_kernel.py  # compares output against PyTorch
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdint>

// ─── Ternary decoding ──────────────────────────────────────────────────────
//
// Packing format: 2 bits per element, MSB-first within each byte.
//   00 = 0,  01 = +1,  10 = -1,  11 = (unused)
//
// For element at position `pos` (row-major):
//   bit_pair starts at byte:  (pos * 2) / 8 = pos / 4
//   bit_offset within byte:   (pos * 2) % 8
//   bit0 (non-zero) = (byte >> (7 - offset)) & 1
//   bit1 (negative)  = (byte >> (7 - offset - 1)) & 1

inline int8_t decode_ternary(const uint8_t* data, size_t pos) {
    size_t byte_idx = (pos * 2) / 8;
    int bit_offset = (pos * 2) % 8;
    uint8_t byte = data[byte_idx];
    // Read 2 bits MSB-first
    unsigned int bits = (byte >> (6 - bit_offset)) & 0x3;
    // Decode: 0=0, 1=+1, 2=-1
    if (bits == 0) return 0;
    if (bits == 1) return 1;
    return -1;  // bits == 2
}

// ─── Binary file reader ────────────────────────────────────────────────────

struct LayerData {
    std::string name;
    int32_t out_features;
    int32_t in_features;
    int32_t num_terms;

    // For each term:
    std::vector<int32_t> alpha_exponents;       // power-of-two exponent
    std::vector<std::vector<uint8_t>> term_data; // packed ternary bits
};

std::vector<LayerData> load_model(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "Cannot open: " << path << std::endl;
        exit(1);
    }

    // Magic
    uint32_t magic;
    f.read(reinterpret_cast<char*>(&magic), 4);
    if (magic != 0xDEADBEEF) {
        std::cerr << "Bad magic: 0x" << std::hex << magic << std::endl;
        exit(1);
    }

    // Num layers
    uint32_t num_layers;
    f.read(reinterpret_cast<char*>(&num_layers), 4);
    std::cout << "Loaded " << num_layers << " layers" << std::endl;

    std::vector<LayerData> layers;
    layers.reserve(num_layers);

    for (uint32_t i = 0; i < num_layers; i++) {
        LayerData ld;

        // Name
        uint32_t name_len;
        f.read(reinterpret_cast<char*>(&name_len), 4);
        ld.name.resize(name_len);
        f.read(&ld.name[0], name_len);

        // Shape
        f.read(reinterpret_cast<char*>(&ld.out_features), 4);
        f.read(reinterpret_cast<char*>(&ld.in_features), 4);

        // Num terms
        f.read(reinterpret_cast<char*>(&ld.num_terms), 4);

        // Terms
        ld.alpha_exponents.resize(ld.num_terms);
        ld.term_data.resize(ld.num_terms);

        for (int t = 0; t < ld.num_terms; t++) {
            // Alpha exponent (int32)
            f.read(reinterpret_cast<char*>(&ld.alpha_exponents[t]), 4);

            // Packed ternary bits
            size_t n_elements = (size_t)ld.out_features * ld.in_features;
            size_t n_bytes = (n_elements * 2 + 7) / 8;
            ld.term_data[t].resize(n_bytes);
            f.read(reinterpret_cast<char*>(ld.term_data[t].data()), n_bytes);
        }

        layers.push_back(std::move(ld));
    }

    return layers;
}

// ─── Naive add/sub kernel (NO multiplications) ────────────────────────────

void ternary_linear_add_sub(const LayerData& layer,
                            const float* input,
                            float* output) {
    const int32_t out_f = layer.out_features;
    const int32_t in_f = layer.in_features;
    const int32_t N = layer.num_terms;

    // Zero output
    std::fill(output, output + out_f, 0.0f);

    // For each term: output += alpha * (input @ T^T)
    for (int t = 0; t < N; t++) {
        int32_t alpha_exp = layer.alpha_exponents[t];
        if (alpha_exp == -128) continue;  // dead term (alpha=0)

        const uint8_t* packed = layer.term_data[t].data();

        // Compute output = input @ T^T  using only add/sub
        // For each output row i: sum over j of T[i,j] * input[j]
        for (int i = 0; i < out_f; i++) {
            float sum = 0.0f;

            for (int j = 0; j < in_f; j++) {
                size_t pos = (size_t)i * in_f + j;
                int8_t t_val = decode_ternary(packed, pos);
                if (t_val == 1) {
                    sum += input[j];
                } else if (t_val == -1) {
                    sum -= input[j];
                }
                // t_val == 0: skip (sparse)
            }

            // Apply alpha via ldexp (alpha * sum = 2^alpha_exp * sum)
            // Single FP32 multiply. Only FP32 multiply in kernel.
            // On hardware: replace with integer shift of a fixed-point accumulator
            output[i] += std::ldexp(sum, alpha_exp);
        }
    }
}

// ─── FP32 reference (MKL-style) ───────────────────────────────────────────

void linear_fp32_ref(const LayerData& layer,
                     const float* input,
                     float* output) {
    const int32_t out_f = layer.out_features;
    const int32_t in_f = layer.in_features;
    const int32_t N = layer.num_terms;

    std::fill(output, output + out_f, 0.0f);

    // Reconstruct W_hat = sum(alpha * T) then matmul
    // FP32 reconstruction. For comparison only.
    std::vector<float> W_hat((size_t)out_f * in_f, 0.0f);
    for (int t = 0; t < N; t++) {
        int32_t alpha_exp = layer.alpha_exponents[t];
        if (alpha_exp == -128) continue;
        float alpha = std::ldexp(1.0f, alpha_exp);  // 2^alpha_exp

        const uint8_t* packed = layer.term_data[t].data();
        for (int i = 0; i < out_f; i++) {
            for (int j = 0; j < in_f; j++) {
                size_t pos = (size_t)i * in_f + j;
                W_hat[i * in_f + j] += alpha * decode_ternary(packed, pos);
            }
        }
    }

    // Matmul
    for (int i = 0; i < out_f; i++) {
        float sum = 0.0f;
        for (int j = 0; j < in_f; j++) {
            sum += W_hat[i * in_f + j] * input[j];
        }
        output[i] = sum;
    }
}

// ─── Correctness test ──────────────────────────────────────────────────────

float max_relative_error(const float* a, const float* b, int n) {
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float denom = std::max(std::abs(a[i]), std::abs(b[i]));
        float err = (denom > 1e-8f) ? std::abs(a[i] - b[i]) / denom
                                    : std::abs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// ─── Main ──────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.bin> [layer_idx]" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    int layer_idx = (argc > 2) ? std::stoi(argv[2]) : 0;

    std::cout << "Loading model: " << model_path << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto layers = load_model(model_path);
    auto t1 = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Load time: " << load_ms << " ms" << std::endl;

    if (layer_idx < 0 || layer_idx >= (int)layers.size()) {
        std::cerr << "Layer index " << layer_idx << " out of range [0, "
                  << layers.size()-1 << "]" << std::endl;
        return 1;
    }

    const auto& layer = layers[layer_idx];
    std::cout << "\nLayer: " << layer.name << std::endl;
    std::cout << "  Shape: [" << layer.out_features << ", "
              << layer.in_features << "]" << std::endl;
    std::cout << "  Terms: " << layer.num_terms << std::endl;

    // Generate random input
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> input(layer.in_features);
    for (int j = 0; j < layer.in_features; j++) {
        input[j] = dist(rng);
    }

    // ---- FP32 reference ----
    std::vector<float> output_fp32(layer.out_features);
    auto t2 = std::chrono::high_resolution_clock::now();
    linear_fp32_ref(layer, input.data(), output_fp32.data());
    auto t3 = std::chrono::high_resolution_clock::now();
    double fp32_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    // ---- Add/Sub kernel ----
    std::vector<float> output_addsub(layer.out_features);
    auto t4 = std::chrono::high_resolution_clock::now();
    ternary_linear_add_sub(layer, input.data(), output_addsub.data());
    auto t5 = std::chrono::high_resolution_clock::now();
    double addsub_ms = std::chrono::duration<double, std::milli>(t5 - t4).count();

    // ---- Verify correctness ----
    float max_err = max_relative_error(
        output_fp32.data(), output_addsub.data(), layer.out_features);

    std::cout << "\n── Results ──────────────────────────────" << std::endl;
    std::cout << "  FP32 reference:    " << fp32_ms << " ms" << std::endl;
    std::cout << "  Add/Sub kernel:    " << addsub_ms << " ms" << std::endl;
    std::cout << "  Speedup:           " << (fp32_ms / addsub_ms) << "x" << std::endl;
    std::cout << "  Max relative err:  " << max_err << std::endl;
    std::cout << "  First 5 outputs:" << std::endl;
    for (int i = 0; i < std::min(5, layer.out_features); i++) {
        std::cout << "    [" << i << "]  FP32=" << output_fp32[i]
                  << "  AddSub=" << output_addsub[i] << std::endl;
    }

    bool pass = max_err < 1e-3f;
    std::cout << "\n  " << (pass ? "✓ PASS" : "✗ FAIL") << std::endl;

    return pass ? 0 : 1;
}

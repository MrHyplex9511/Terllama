/*
 * packrat_reconfig.h — Dynamic auto-reconfiguration (Packrat)
 *
 * Runs a lightweight benchmark at startup to find optimal thread count
 * and batch configuration for the current CPU and model size.  Results
 * are applied to the Sandwich scheduler for phase-aware execution.
 *
 * The benchmark uses synthetic weights to avoid depending on model
 * loading state.  It measures:
 *   - Throughput (tokens/s) for each thread count from 1..max_cores
 *   - Cross-over point where more threads stop improving throughput
 *   - Optimal prefill vs decode thread counts
 */
#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <unistd.h>
#include "core/logger.h"
#include "core/sandwich_scheduler.h"
#include "kernel_decl.h"

// ─── Benchmark result ──────────────────────────────────────────────────
struct PackratConfig {
    int  prefill_threads{4};          // optimal for compute-bound
    int  decode_threads{4};           // optimal for memory-bound
    int  prefill_batch{8};            // micro-batch size for prefill
    int  decode_batch{1};             // always 1 for decode
    bool pin_threads{true};

    // Applied cores
    int  prefill_start_core{0};
    int  prefill_core_stride{1};
    int  decode_start_core{0};
    int  decode_core_stride{2};
};

// ─── Synthetic micro-benchmark ─────────────────────────────────────────
// Runs a ternary matmul with synthetic I2_S weights + random input.
// Measures throughput at different thread counts.
class PackratTuner {
public:
    PackratTuner() {
        detect_cores();
        Logger::info("Packrat: detected %d cores, starting auto-tune...", num_cores_);
    }

    // ─── Run the benchmark and return optimal config ────────────────────
    PackratConfig tune(int hidden_size, int intermediate_size) {
        PackratConfig cfg;

        // Build synthetic I2_S layer data for benchmark
        auto bench_layer = make_synthetic_i2s_layer(hidden_size, intermediate_size * 3);
        auto bench_input = make_random_input(hidden_size);

        // Test prefill (compute-bound): try all thread counts
        Logger::info("Packrat: benchmarking prefill (compute-bound)...");
        std::vector<BenchResult> results;

        for (int t = 1; t <= max_test_threads_; t++) {
            double ms = benchmark_matmul(bench_layer, bench_input, t, 32);
            double tok_s = 32.0 / (ms / 1000.0);
            results.push_back({t, ms, tok_s});
            Logger::info("  [threads=%2d] %7.1f ms  (%8.1f tok/s)", t, ms, tok_s);
        }

        // Pick thread count where throughput stops improving significantly
        int best_prefill = select_optimal(results, 0.05);  // 5% improvement threshold
        cfg.prefill_threads = std::max(1, std::min(best_prefill, num_cores_));

        // Decode (memory-bound): usually benefit from fewer threads
        results.clear();
        int max_decode = std::max(1, num_cores_ / 2);
        Logger::info("Packrat: benchmarking decode (memory-bound)...");
        for (int t = 1; t <= max_decode; t++) {
            double ms = benchmark_matmul(bench_layer, bench_input, t, 1);  // batch=1
            double tok_s = 1.0 / (ms / 1000.0);
            results.push_back({t, ms, tok_s});
            Logger::info("  [threads=%2d] %7.1f ms  (%8.1f tok/s)", t, ms, tok_s);
        }
        int best_decode = select_optimal(results, 0.03);  // 3% threshold (tighter for decode)
        cfg.decode_threads = std::max(1, std::min(best_decode, max_decode));

        // Heuristic: if decode > prefill threads, something's wrong — cap
        if (cfg.decode_threads > cfg.prefill_threads)
            cfg.decode_threads = std::max(1, cfg.prefill_threads / 2);

        // Set core pinning based on thread counts
        cfg.prefill_start_core = 0;
        cfg.prefill_core_stride = (num_cores_ / cfg.prefill_threads);
        cfg.decode_start_core = 0;
        cfg.decode_core_stride = std::max(1, num_cores_ / cfg.decode_threads);

        Logger::info("Packrat: optimal -> prefill=%dt decode=%dt (pin:%s)",
                     cfg.prefill_threads, cfg.decode_threads,
                     cfg.pin_threads ? "yes" : "no");

        return cfg;
    }

    // ─── Apply Packrat config to a Sandwich scheduler ───────────────────
    static void apply_to_scheduler(SandwichScheduler& sched, const PackratConfig& cfg) {
        sched.set_prefill_threads(cfg.prefill_threads);
        sched.set_decode_threads(cfg.decode_threads);
        sched.set_prefill_batch(cfg.prefill_batch);
        sched.set_pin(cfg.pin_threads);
        Logger::info("Packrat: applied config to Sandwich scheduler");
    }

private:
    struct BenchResult {
        int    threads;
        double ms;
        double tok_s;
    };

    void detect_cores() {
        num_cores_ = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (num_cores_ < 1) num_cores_ = 4;
        max_test_threads_ = std::min(num_cores_, 32);  // cap at 32 for test
    }

    // ─── Create synthetic I2_S layer for benchmarking ───────────────────
    LayerData make_synthetic_i2s_layer(int out_features, int in_features) {
        LayerData ld;
        ld.name = "bench";
        ld.out_features = out_features;
        ld.in_features = in_features;
        ld.has_i2s = true;
        ld.i2s_qk = 128;

        int qk = 128;
        int n_blocks = (in_features + qk - 1) / qk;
        int codes_per_block = qk / 4;
        ld.i2s_blocks.resize((size_t)out_features * n_blocks);

        // Fill with deterministic ternary patterns (not all zeros)
        for (size_t i = 0; i < ld.i2s_blocks.size(); i++) {
            ld.i2s_blocks[i].packed.resize(codes_per_block);
            for (int j = 0; j < codes_per_block; j++) {
                uint8_t code = (uint8_t)((i * codes_per_block + j) % 3);
                ld.i2s_blocks[i].packed[j] = (code << 6) | ((code + 1) % 3 << 4) |
                                              ((code + 2) % 3 << 2) | code;
            }
            ld.i2s_blocks[i].scale = 1.0f;
        }
        return ld;
    }

    std::vector<float> make_random_input(int n) {
        std::vector<float> v(n);
        for (int i = 0; i < n; i++)
            v[i] = (float)((i * 12345 + 6789) % 100) / 100.0f - 0.5f;
        return v;
    }

    // ─── Run a single benchmark: time ternary_linear_i2s ────────────────
    double benchmark_matmul(const LayerData& layer,
                            const std::vector<float>& input,
                            int n_threads, int n_batches) {
        std::vector<float> output(layer.out_features);

        // Set thread count
        setenv("OMP_NUM_THREADS", std::to_string(n_threads).c_str(), 1);

        // Warmup
        for (int w = 0; w < 3; w++) {
            ternary_linear_i2s(layer, input.data(), output.data());
        }

        // Measured runs
        auto t0 = std::chrono::high_resolution_clock::now();
        int n_warm = 10;
        for (int b = 0; b < n_batches; b++) {
            for (int w = 0; w < n_warm; w++) {
                ternary_linear_i2s(layer, input.data(), output.data());
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        return total_ms / (double)(n_batches * n_warm);
    }

    // ─── Select optimal thread count from benchmark results ─────────────
    int select_optimal(const std::vector<BenchResult>& results, double min_improvement) {
        (void)min_improvement;  // future use with Elbow method
        if (results.empty()) return 1;

        // Find peak throughput
        double best_tok_s = 0;
        for (const auto& r : results)
            if (r.tok_s > best_tok_s) best_tok_s = r.tok_s;

        // Pick first thread count that achieves >95% of peak
        // (earlier = fewer threads = less power, less contention)
        for (const auto& r : results) {
            if (r.tok_s >= best_tok_s * 0.95)
                return r.threads;
        }
        return results.back().threads;
    }

    int num_cores_{4};
    int max_test_threads_{4};
};

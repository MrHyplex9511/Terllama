/*
 * sandwich_scheduler.h — Phase-aware prefill/decode scheduler
 *
 * Separates inference into two phases with different scheduling:
 *
 *   PREFILL:  process entire prompt in one batch — compute-bound
 *             → high thread count, large batch, compact NUMA
 *   DECODE:   generate one token at a time — memory-bound
 *             → lower thread count, prefetch streams, spread NUMA
 *
 * The scheduler is a passive advisor: it tells the generation loop
 * what phase we're in and what config to use.  It doesn't own threads.
 * Thread pinning uses sched_setaffinity (Linux-only).
 */
#pragma once
#include <vector>
#include <atomic>
#include <thread>
#include <cstring>
#include <cstdlib>
#include <sched.h>
#include <unistd.h>
#include <omp.h>
#include "core/logger.h"

// ─── Phase enum ────────────────────────────────────────────────────────
enum class GenPhase : uint8_t {
    PREFILL = 0,   // processing prompt (compute-bound)
    DECODE  = 1,   // generating tokens (memory-bound)
};

inline const char* phase_name(GenPhase p) {
    return p == GenPhase::PREFILL ? "prefill" : "decode";
}

// ─── Per-phase config ───────────────────────────────────────────────────
struct PhaseConfig {
    int  num_threads{4};          // OpenMP threads for this phase
    int  batch_size{1};           // micro-batch for matmuls (1=no micro-batch)
    bool pin_threads{false};      // pin threads to cores?
    int  start_core{0};           // first CPU core to pin to
    int  core_stride{1};          // stride between pinned cores
};

// ─── Sandwich scheduler ─────────────────────────────────────────────────
class SandwichScheduler {
public:
    SandwichScheduler() {
        detect_num_cores();
        // Default configs
        prefill_cfg_.num_threads = num_cores_;
        prefill_cfg_.batch_size  = 8;   // micro-batch matmuls
        prefill_cfg_.pin_threads = true;
        prefill_cfg_.start_core  = 0;
        prefill_cfg_.core_stride = 1;

        decode_cfg_.num_threads = std::max(1, num_cores_ / 2);
        decode_cfg_.batch_size  = 1;    // no micro-batch, one token
        decode_cfg_.pin_threads = true;
        decode_cfg_.start_core  = 0;
        decode_cfg_.core_stride = 2;    // spread: skip hyperthreads
    }

    // ─── Apply phase configuration ──────────────────────────────────────
    void apply(GenPhase phase) {
        // Check for Packrat env overrides first
        auto cfg = (phase == GenPhase::PREFILL) ? prefill_cfg_ : decode_cfg_;
        const char* pk_threads = std::getenv(
            phase == GenPhase::PREFILL ? "TERLLAMA_PREFILL_THREADS"
                                       : "TERLLAMA_DECODE_THREADS");
        const char* pk_pin = std::getenv("TERLLAMA_PIN_THREADS");
        if (pk_threads) cfg.num_threads = std::max(1, std::stoi(pk_threads));
        if (pk_pin)     cfg.pin_threads = (std::string(pk_pin) == "1");

        // Adjust for decode: memory-bound, fewer threads by default
        if (phase == GenPhase::DECODE && !pk_threads) {
            cfg.num_threads = std::max(1, cfg.num_threads / 2);
        }

        Logger::info("Sandwich: phase=%s threads=%d batch=%d pin=%s",
                     phase_name(phase), cfg.num_threads, cfg.batch_size,
                     cfg.pin_threads ? "yes" : "no");

        // Set OMP_NUM_THREADS for OpenMP
        omp_num_threads_ = cfg.num_threads;
        setenv("OMP_NUM_THREADS", std::to_string(cfg.num_threads).c_str(), 1);

        // Pin threads if requested
        if (cfg.pin_threads) {
            pin_worker_threads(cfg.num_threads, cfg.start_core, cfg.core_stride);
        }
    }

    // ─── Pin calling thread to a specific core ──────────────────────────
    static void pin_self(int core_id) {
        cpu_set_t cset;
        CPU_ZERO(&cset);
        CPU_SET(core_id, &cset);
        if (sched_setaffinity(0, sizeof(cset), &cset) != 0) {
            Logger::warn("Failed to pin thread to core %d", core_id);
        }
    }

    // ─── Pin worker threads across cores ────────────────────────────────
    static void pin_worker_threads(int n_threads, int start, int stride) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            if (tid < n_threads) {
                int core = start + tid * stride;
                pin_self(core);
            }
        }
    }

    // ─── Setters ────────────────────────────────────────────────────────
    void set_prefill_threads(int n)  { prefill_cfg_.num_threads = std::max(1, n); }
    void set_decode_threads(int n)   { decode_cfg_.num_threads  = std::max(1, n); }
    void set_prefill_batch(int b)    { prefill_cfg_.batch_size   = std::max(1, b); }
    void set_pin(bool p) {
        prefill_cfg_.pin_threads = p;
        decode_cfg_.pin_threads  = p;
    }
    int  omp_threads() const { return omp_num_threads_; }

    // ─── Prompt-size heuristics ─────────────────────────────────────────
    static GenPhase decide_phase(int n_prompt_tokens, int tokens_generated_so_far) {
        // Prefill: still processing initial prompt
        // Decode:  one token at a time
        if (tokens_generated_so_far < n_prompt_tokens) return GenPhase::PREFILL;
        return GenPhase::DECODE;
    }

    static int recommended_batch(GenPhase phase, int n_prompt_tokens) {
        if (phase == GenPhase::PREFILL) {
            // Prefill can batch up to ~2048 tokens or all prompt tokens
            return std::min(2048, std::max(1, n_prompt_tokens));
        }
        return 1;  // decode is always single-token
    }

private:
    void detect_num_cores() {
        num_cores_ = (int)sysconf(_SC_NPROCESSORS_ONLN);
        if (num_cores_ < 1) num_cores_ = 4;
        Logger::debug("Sandwich: detected %d logical cores", num_cores_);
    }

    int num_cores_{4};
    PhaseConfig prefill_cfg_;
    PhaseConfig decode_cfg_;
    std::atomic<int> omp_num_threads_{4};
};

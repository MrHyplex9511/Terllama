/*
 * kernel_mote.cpp — MoTE (Mixture of Ternary Experts) forward pass
 *
 * Processes top-K experts sequentially per token.
 * Reuses existing ternary_linear_dispatch for per-expert matmuls.
 *
 * Router: FP32 weights → softmax → top-K gating → weighted combine
 */
#include "kernel_decl.h"
#include "model.h"
#include "inference.h"
#include <cmath>
#include <algorithm>
#include <cstring>

static inline float silu_mote(float x) {
    return x / (1.0f + std::exp(-x));
}

// ═══════════════════════════════════════════════════════════════════════════
// mote_ternary_linear — full MoTE FFN forward
// ═══════════════════════════════════════════════════════════════════════════
void mote_ternary_linear(const MoTELayerData& mote, const float* x,
                          float* output, int hidden_size,
                          int intermediate_size) {
    int K = mote.num_experts;
    if (K == 0 || !mote.is_mote) {
        // Fallback: just run shared expert as regular FFN
        std::vector<float> gate_buf(intermediate_size), up_buf(intermediate_size);
        ternary_linear_dispatch(mote.gate_proj, x, gate_buf.data());
        ternary_linear_dispatch(mote.up_proj, x, up_buf.data());
        for (int i = 0; i < intermediate_size; i++)
            gate_buf[i] = silu_mote(gate_buf[i]) * up_buf[i];
        ternary_linear_dispatch(mote.down_proj, gate_buf.data(), output);
        return;
    }

    // ─── 1. Router: compute logits ────────────────────────────────────
    const float* rw = mote.router_weight.data();
    std::vector<float> router_logits(K, 0.0f);
    #pragma omp parallel for
    for (int k = 0; k < K; k++) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++)
            sum += x[i] * rw[i * K + k];
        router_logits[k] = sum * mote.router_scale;
    }

    // Softmax
    float max_logit = *std::max_element(router_logits.begin(), router_logits.end());
    std::vector<float> router_probs(K, 0.0f);
    float sum_probs = 0.0f;
    for (int k = 0; k < K; k++) {
        router_probs[k] = std::exp(router_logits[k] - max_logit);
        sum_probs += router_probs[k];
    }
    float inv_sum = 1.0f / (sum_probs + 1e-10f);
    for (int k = 0; k < K; k++) router_probs[k] *= inv_sum;

    // ─── 2. Top-K selection ───────────────────────────────────────────
    int topk = std::min(mote.top_k, K);
    struct Scored { float prob; int idx; };
    std::vector<Scored> scored(K);
    for (int k = 0; k < K; k++) scored[k] = {router_probs[k], k};
    std::partial_sort(scored.begin(), scored.begin() + topk, scored.end(),
                      [](const Scored& a, const Scored& b) { return a.prob > b.prob; });

    // Reuse buffers
    int IS = intermediate_size;
    std::vector<float> gate_buf(IS), up_buf(IS);
    std::vector<float> combined(IS, 0.0f);

    // ─── 3. Shared expert ─────────────────────────────────────────────
    ternary_linear_dispatch(mote.gate_proj, x, gate_buf.data());
    ternary_linear_dispatch(mote.up_proj, x, up_buf.data());
    for (int i = 0; i < IS; i++)
        combined[i] = silu_mote(gate_buf[i]) * up_buf[i];

    // ─── 4. Routed top-K experts ──────────────────────────────────────
    // Two passes so the elementwise combine uses ONE parallel region for all
    // experts instead of spawning a team per expert matmul. The expert
    // GEMM dispatches stay outside any region (each keeps its own internal
    // parallelism — wrapping them in the outer region would force nested
    // single-threaded execution). Per-element accumulation order is
    // unchanged, so results are bit-identical to the old per-expert loop.
    std::vector<float> expert_contrib((size_t)topk * IS);
    for (int e = 0; e < topk; e++) {
        int ek = scored[e].idx;
        float weight = scored[e].prob;

        ternary_linear_dispatch(mote.expert_gate[ek], x, gate_buf.data());
        ternary_linear_dispatch(mote.expert_up[ek], x, up_buf.data());

        float* dest = &expert_contrib[(size_t)e * IS];
        for (int i = 0; i < IS; i++)
            dest[i] = weight * silu_mote(gate_buf[i]) * up_buf[i];
    }

    #pragma omp parallel for
    for (int i = 0; i < IS; i++) {
        float acc = combined[i];
        for (int e = 0; e < topk; e++) acc += expert_contrib[(size_t)e * IS + i];
        combined[i] = acc;
    }

    // ─── 5. Down projection ───────────────────────────────────────────
    ternary_linear_dispatch(mote.down_proj, combined.data(), output);
}

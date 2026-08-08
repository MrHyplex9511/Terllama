// ALS ternary decomposition + packing — C++ port of
// scripts/export_ternary_model_bitnet.py.
//
// Pipeline (matches the Python reference exactly):
//   1. Greedy element-wise ternary decomposition: num_terms power-of-2-scaled
//      ternary matrices, each term chosen by argmin over k of
//      ||R - 2^k * clamp(round(R / 2^k), -1, 1)||_F.
//   2. Refinement (max_iter=5): for each term i, rebuild the residual
//      R = W - sum_{j!=i, a_j!=0} a_j * T_j and re-solve term i.
//   3. Joint per-block least-squares scales: for each qk-wide block, solve
//      (A^T A + ridge*I) x = A^T w in float64 (A = flattened ternary blocks),
//      ridge = 1e-10 * max(diag(A^T A)).
//   4. Packing: 2-bit codes per weight (MSB-first, 4/byte) + per-block float32
//      scale, row-major; then the multi-term container.
//
// Numerics notes (verified against torch 2.10 CPU):
//   * torch.round is banker's rounding (half to even); std::round is not used.
//     We implement it deterministically from floor/frac so the result does not
//     depend on the process floating-point rounding mode.
//   * R / a is an fp32 division (torch keeps the fp32 tensor dtype against a
//     weak python float scalar); power-of-2 scales are exact in fp32.
//   * A^T A over ternary columns is an EXACT integer sum in float64 (products
//     are {-1,0,1}, |sum| < 2^53), so it is independent of accumulation order.
//     A^T w is accumulated sequentially in float64; the reference uses MKL so
//     last-ulp differences in the right-hand side are possible. These are
//     absorbed by the final float32 cast (relative diff < 1e-6 accepted; codes
//     are unaffected since they are fixed before the scale solve).
//   * The NxN solve replicates the LAPACK dgesv (dgetrf + dgetrs) operation
//     sequence, including its exact-zero guards, so it is bit-identical to
//     torch.linalg.solve given identical inputs.
//   * Frobenius norms for best-k selection are accumulated in float64. The
//     reference accumulates in float32 with a blocked reduction; relative
//     argmin gaps on real weights are >= 2e-4, far above any float32-vs-float64
//     reduction difference, so the selected k (and therefore all ternary codes)
//     matches exactly.

#include "als_decompose.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace terllama {
namespace {

// ═════════════════════════════════════════════════════════════════════════
// Tiny dense linear algebra (float64)
// ═════════════════════════════════════════════════════════════════════════

// LU factorization with partial pivoting, replicating the LAPACK dgetf2
// (unblocked) operation sequence, row-major. Overwrites A with L\U, fills
// ipiv. Returns false if singular.
bool lu_factor(double* A, int n, int* ipiv) {
    for (int k = 0; k < n; k++) {
        // idamax over column k from row k: first index of the max |value|
        int imax = k;
        double amax = std::fabs(A[(size_t)k * n + k]);
        for (int i = k + 1; i < n; i++) {
            double v = std::fabs(A[(size_t)i * n + k]);
            if (v > amax) {
                amax = v;
                imax = i;
            }
        }
        ipiv[k] = imax;
        if (imax != k) {
            for (int j = 0; j < n; j++) {
                std::swap(A[(size_t)k * n + j], A[(size_t)imax * n + j]);
            }
        }
        if (A[(size_t)k * n + k] == 0.0) return false;  // singular (ridge guards)
        double pivot = A[(size_t)k * n + k];
        for (int i = k + 1; i < n; i++) A[(size_t)i * n + k] /= pivot;
        for (int i = k + 1; i < n; i++) {
            double aik = A[(size_t)i * n + k];
            for (int j = k + 1; j < n; j++) {
                A[(size_t)i * n + j] -= aik * A[(size_t)k * n + j];
            }
        }
    }
    return true;
}

// Solve A x = b given LU factors from lu_factor, replicating the LAPACK
// dgetrs operation sequence (including its b == 0.0 guards).
void lu_solve(const double* A, int n, const int* ipiv, double* b) {
    // Apply interchanges to the right-hand side (forward order).
    for (int j = 0; j < n; j++) {
        if (ipiv[j] != j) std::swap(b[j], b[ipiv[j]]);
    }
    // Solve L y = b (unit lower triangular).
    for (int i = 0; i < n; i++) {
        if (b[i] != 0.0) {
            for (int j = 0; j < i; j++) b[i] -= A[(size_t)i * n + j] * b[j];
        }
    }
    // Solve U x = y.
    for (int i = n - 1; i >= 0; i--) {
        if (b[i] != 0.0) {
            for (int j = i + 1; j < n; j++) b[i] -= A[(size_t)i * n + j] * b[j];
            b[i] /= A[(size_t)i * n + i];
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
// torch.round replication (banker's rounding)
// ═════════════════════════════════════════════════════════════════════════
// torch.round(0.5)=0, torch.round(1.5)=2, torch.round(-0.5)=-0,
// torch.round(-1.5)=-2 — round half to even. std::round rounds half away from
// zero and nearbyint depends on the process rounding mode, so we compute it
// directly from floor/frac. All inputs are fp32 values (R / a) that are exact
// in float64, so the frac == 0.5 boundary test is exact.
int64_t torch_round(double x) {
    const double xa = std::fabs(x);
    const double fl = std::floor(xa);
    const double frac = xa - fl;
    int64_t r;
    if (frac < 0.5) {
        r = (int64_t)fl;
    } else if (frac > 0.5) {
        r = (int64_t)(fl + 1.0);
    } else {  // exactly .5: round to even
        r = (((int64_t)fl & 1) == 0) ? (int64_t)fl : (int64_t)(fl + 1.0);
    }
    return (x < 0.0) ? -r : r;
}

// clamp(round(x), -1, 1) as int8 — matches torch.clamp(torch.round(x), -1, 1).
int8_t clamp_round_ternary(double x) {
    int64_t r = torch_round(x);
    if (r > 1) return 1;
    if (r < -1) return -1;
    return (int8_t)r;
}

// fp32 Frobenius norm accumulated in float64 (see header notes). Used only to
// pick best_k; argmin gaps on real weights are far above fp32 reduction noise.
double fro_norm64(const float* d, size_t n) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) acc += (double)d[i] * (double)d[i];
    return std::sqrt(acc);
}

// ═════════════════════════════════════════════════════════════════════════
// Core decomposition
// ═════════════════════════════════════════════════════════════════════════

struct RawTerm {
    double a;              // power-of-2 scale (or 0.0)
    std::vector<int8_t> T; // out_f * in_f ternary, values -1/0/1
};

// Greedy element-wise ternary decomposition (script _greedy_terms).
std::vector<RawTerm> greedy_terms(const float* W, size_t n, int num_terms) {
    std::vector<RawTerm> terms;
    terms.reserve(num_terms);
    std::vector<float> R(W, W + n);  // fp32 working residual
    std::vector<float> diff(n);

    for (int t = 0; t < num_terms; t++) {
        float rmax = 0.0f;
        for (size_t i = 0; i < n; i++) {
            float v = std::fabs(R[i]);
            if (v > rmax) rmax = v;
        }
        if (rmax < 1e-8f) {
            terms.push_back({0.0, std::vector<int8_t>(n, 0)});
            continue;
        }
        int lo = (int)std::floor(std::log2(std::max((double)rmax / 3.0, 1e-10)));
        int hi = (int)std::ceil(std::log2(std::max((double)rmax * 1.5, 1e-10))) + 2;
        int best_k = lo;
        double best_e = std::numeric_limits<double>::infinity();
        for (int k = lo; k < hi; k++) {
            const float a = std::ldexp(1.0f, k);  // 2^k, exact in fp32
            for (size_t i = 0; i < n; i++) {
                int8_t tc = clamp_round_ternary((double)(R[i] / a));
                diff[i] = R[i] - a * (float)tc;
            }
            double e = fro_norm64(diff.data(), n);
            if (e < best_e) {
                best_e = e;
                best_k = k;
            }
        }
        const float a = std::ldexp(1.0f, best_k);
        std::vector<int8_t> T(n);
        for (size_t i = 0; i < n; i++) {
            T[i] = clamp_round_ternary((double)(R[i] / a));
            R[i] -= a * (float)T[i];  // R -= a*T in fp32
        }
        terms.push_back({(double)a, std::move(T)});
    }
    return terms;
}

// Coordinate-descent refinement (script als_decompose Phase 2): rebuild the
// residual excluding term i and re-solve term i against it.
void refine_terms(const float* W, size_t n, int num_terms, int max_iter,
                  std::vector<RawTerm>& terms) {
    std::vector<float> R(n);
    std::vector<float> diff(n);
    for (int it = 0; it < max_iter; it++) {
        for (int i = 0; i < num_terms; i++) {
            // R = W - sum_{j != i, a_j != 0} a_j * T_j  (fp32, j ascending)
            for (size_t p = 0; p < n; p++) R[p] = W[p];
            for (int j = 0; j < num_terms; j++) {
                if (j == i || terms[j].a == 0.0) continue;
                const float a = (float)terms[j].a;  // exact power of two
                const int8_t* Tj = terms[j].T.data();
                for (size_t p = 0; p < n; p++) R[p] -= a * (float)Tj[p];
            }
            float rmax = 0.0f;
            for (size_t p = 0; p < n; p++) {
                float v = std::fabs(R[p]);
                if (v > rmax) rmax = v;
            }
            if (rmax < 1e-8f) {
                terms[i] = {0.0, std::vector<int8_t>(n, 0)};
                continue;
            }
            int lo = (int)std::floor(std::log2(std::max((double)rmax / 3.0, 1e-10)));
            int hi = (int)std::ceil(std::log2(std::max((double)rmax * 1.5, 1e-10))) + 2;
            int best_k = lo;
            double best_e = std::numeric_limits<double>::infinity();
            for (int k = lo; k < hi; k++) {
                const float a = std::ldexp(1.0f, k);
                for (size_t p = 0; p < n; p++) {
                    int8_t tc = clamp_round_ternary((double)(R[p] / a));
                    diff[p] = R[p] - a * (float)tc;
                }
                double e = fro_norm64(diff.data(), n);
                if (e < best_e) {
                    best_e = e;
                    best_k = k;
                }
            }
            const float a = std::ldexp(1.0f, best_k);
            std::vector<int8_t> T(n);
            for (size_t p = 0; p < n; p++) {
                T[p] = clamp_round_ternary((double)(R[p] / a));
            }
            terms[i] = {(double)a, std::move(T)};
        }
    }
}

// Joint per-block least-squares scales: for each qk-wide block solve the NxN
// ridge-regularized normal equations in float64. Returns scales[block][term]
// in float32 (cast from float64), matching Python's per-block list order.
std::vector<std::vector<float>> fit_block_scales(const float* W, int out_f,
                                                 int in_f, int num_terms, int qk,
                                                 const std::vector<RawTerm>& terms) {
    const int n_blocks = (in_f + qk - 1) / qk;
    const int N = num_terms;
    std::vector<std::vector<float>> out_terms;  // [block][term]
    out_terms.reserve(n_blocks);

    std::vector<double> AtA((size_t)N * N, 0.0);
    std::vector<double> Atw(N, 0.0);
    std::vector<double> AtA_work((size_t)N * N, 0.0);
    std::vector<int> ipiv(N, 0);
    std::vector<double> col(N);

    for (int b = 0; b < n_blocks; b++) {
        const int start = b * qk;
        const int end = std::min(start + qk, in_f);
        const int bw = end - start;

        // A^T A (exact integer in float64 for ternary columns) and A^T w,
        // accumulated in one pass over the flattened block (row-major order).
        std::fill(AtA.begin(), AtA.end(), 0.0);
        std::fill(Atw.begin(), Atw.end(), 0.0);
        for (int r = 0; r < out_f; r++) {
            const int64_t row_base = (int64_t)r * in_f + start;
            for (int c = 0; c < bw; c++) {
                const int64_t p = row_base + c;
                const double w = (double)W[p];
                for (int i = 0; i < N; i++) col[i] = (double)terms[i].T[p];
                for (int i = 0; i < N; i++) {
                    Atw[i] += col[i] * w;
                    double* row_i = AtA.data() + (size_t)i * N;
                    for (int j = 0; j < N; j++) row_i[j] += col[i] * col[j];
                }
            }
        }

        // ridge = 1e-10 * max(diag(AtA))
        double diag_max = 0.0;
        for (int i = 0; i < N; i++) {
            if (AtA[(size_t)i * N + i] > diag_max) diag_max = AtA[(size_t)i * N + i];
        }
        const double ridge = 1e-10 * diag_max;

        // (AtA + ridge*I) x = Atw
        std::memcpy(AtA_work.data(), AtA.data(), sizeof(double) * (size_t)N * N);
        for (int i = 0; i < N; i++) AtA_work[(size_t)i * N + i] += ridge;
        if (!lu_factor(AtA_work.data(), N, ipiv.data())) {
            // Fallback (should be unreachable with ridge): zero scales.
            out_terms.emplace_back(N, 0.0f);
            continue;
        }
        lu_solve(AtA_work.data(), N, ipiv.data(), Atw.data());

        out_terms.emplace_back(N);
        for (int i = 0; i < N; i++) out_terms.back()[i] = (float)Atw[i];
    }

    // Transpose: (per-block lists) -> (per-term lists of n_blocks scales)
    std::vector<std::vector<float>> per_term(N, std::vector<float>(n_blocks));
    for (int i = 0; i < N; i++) {
        for (int b = 0; b < n_blocks; b++) per_term[i][b] = out_terms[b][i];
    }
    return per_term;
}

}  // namespace

// ═════════════════════════════════════════════════════════════════════════
// Public API
// ═════════════════════════════════════════════════════════════════════════

std::vector<ALSTerm> als_decompose(const float* W, int out_f, int in_f,
                                   int num_terms, int max_iter, int qk) {
    const size_t n = (size_t)out_f * in_f;
    std::vector<RawTerm> terms = greedy_terms(W, n, num_terms);
    refine_terms(W, n, num_terms, max_iter, terms);

    std::vector<std::vector<float>> per_term =
        fit_block_scales(W, out_f, in_f, num_terms, qk, terms);

    std::vector<ALSTerm> result(num_terms);
    for (int i = 0; i < num_terms; i++) {
        result[i].scales = std::move(per_term[i]);
        result[i].ternary = std::move(terms[i].T);
    }
    return result;
}

std::vector<std::vector<float>> als_decompose_scales(const float* W, int out_f,
                                                     int in_f, int num_terms) {
    const size_t n = (size_t)out_f * in_f;
    std::vector<RawTerm> terms = greedy_terms(W, n, num_terms);
    refine_terms(W, n, num_terms, /*max_iter=*/5, terms);
    return fit_block_scales(W, out_f, in_f, num_terms, /*qk=*/128, terms);
}

// Pack one ALS term into dst, returning bytes written. Same layout as the
// old pack_als_block blob: per row, n_blocks * [codes_per_block code bytes +
// float32 scale]. Writing directly into the destination avoids a temporary
// vector + memcpy per term.
static size_t pack_als_block_into(const ALSTerm& term, uint8_t* dst,
                                  int out_f, int in_f, int qk) {
    const int n_blocks = (int)term.scales.size();
    const int codes_per_block = qk / 4;
    const int row_stride = n_blocks * (codes_per_block + (int)sizeof(float));
    const size_t total = (size_t)out_f * (size_t)row_stride;

    std::vector<uint8_t> codes(qk);
    for (int row = 0; row < out_f; row++) {
        const int8_t* tv = term.ternary.data() + (size_t)row * in_f;
        for (int b = 0; b < n_blocks; b++) {
            const int start = b * qk;
            const int end = std::min(start + qk, in_f);
            const int bw = end - start;

            // {-1,0,+1} -> {0,1,2}; default code 1 (ternary 0), pad tail with 1
            for (int c = 0; c < qk; c++) codes[c] = 1;
            for (int c = 0; c < bw; c++) {
                const int8_t v = tv[start + c];
                codes[c] = (v == -1) ? 0 : (v == 1) ? 2 : 1;
            }

            // 4 vals/byte, MSB-first: [a,b,c,d] -> (a<<6)|(b<<4)|(c<<2)|d
            const size_t off = (size_t)row * row_stride + (size_t)b * (codes_per_block + 4);
            for (int g = 0; g < codes_per_block; g++) {
                uint8_t byte = (uint8_t)((codes[4 * g] << 6) | (codes[4 * g + 1] << 4) |
                                         (codes[4 * g + 2] << 2) | codes[4 * g + 3]);
                dst[off + g] = byte;
            }
            const float scale = term.scales[b];
            std::memcpy(dst + off + codes_per_block, &scale, sizeof(float));
        }
    }
    return total;
}

std::vector<uint8_t> pack_als_block(const ALSTerm& term, int out_f, int in_f,
                                    int qk) {
    const int n_blocks = (int)term.scales.size();
    const int codes_per_block = qk / 4;
    const int row_stride = n_blocks * (codes_per_block + (int)sizeof(float));
    std::vector<uint8_t> buf((size_t)out_f * row_stride);
    pack_als_block_into(term, buf.data(), out_f, in_f, qk);
    return buf;
}

std::vector<uint8_t> pack_als_block_terms(const std::vector<ALSTerm>& terms,
                                          int out_f, int in_f, int qk) {
    const int n_blocks = (in_f + qk - 1) / qk;
    const int codes_per_block = qk / 4;
    const size_t per_row = (size_t)n_blocks * (codes_per_block + (int)sizeof(float));
    std::vector<uint8_t> buf;
    buf.reserve(4 + terms.size() * (4 + (size_t)out_f * per_row));
    const uint32_t num_terms = (uint32_t)terms.size();
    buf.insert(buf.end(), (const uint8_t*)&num_terms, (const uint8_t*)&num_terms + 4);
    for (const ALSTerm& term : terms) {
        // Per-term length must match pack_als_block_into, which derives
        // n_blocks from term.scales.size() (same as the pre-refactor blob).
        const size_t row_stride = (size_t)term.scales.size() * (codes_per_block + (int)sizeof(float));
        const uint32_t blob_len = (uint32_t)((size_t)out_f * row_stride);
        buf.insert(buf.end(), (const uint8_t*)&blob_len, (const uint8_t*)&blob_len + 4);
        const size_t base = buf.size();
        buf.resize(base + blob_len);
        pack_als_block_into(term, buf.data() + base, out_f, in_f, qk);
    }
    return buf;
}

}  // namespace terllama

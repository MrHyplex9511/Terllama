/*
 * partitioner.h — Memory-proportional layer allocation for Terllama cluster.
 *
 * Allocates transformer layers across N worker nodes so that each node hosts
 * a layer count proportional to its available RAM (Exo-style largest-remainder
 * allocation), then derives contiguous half-open [start, end) ShardSpecs in
 * pipeline order.
 *
 * Header-only (inline) for testability.
 */
#pragma once

#include "protocol.h"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace tldist {

// ─── Memory-proportional layer counts (largest-remainder) ───────────────────
// Port of Exo's allocate_layers_proportionally:
//   raw[i]    = memory_fractions[i] * total_layers
//   result[i] = floor(raw[i])
//   distribute the remaining (total_layers - sum(result)) layers one each,
//   in order of descending fractional remainder
//   then force min 1 layer per node by taking from the largest allocation.
//
// @param memory_fractions  fraction of total memory per node (need not sum to 1,
//                          but is normalized by compute_shards before calling)
// @return per-node layer counts (sum == total_layers)
inline std::vector<int> allocate_layers_proportionally(
    int total_layers, const std::vector<double>& memory_fractions) {
    const size_t n = memory_fractions.size();
    if (n == 0)
        throw std::runtime_error(
            "allocate_layers_proportionally: zero nodes");
    if ((size_t)total_layers < n)
        throw std::runtime_error(
            "allocate_layers_proportionally: total_layers (" +
            std::to_string(total_layers) + ") < nodes (" +
            std::to_string(n) + ")");

    // raw[i] = frac[i] * total_layers, floor into result.
    std::vector<double> raw(n);
    std::vector<int> result(n, 0);
    int sum = 0;
    for (size_t i = 0; i < n; i++) {
        double frac = memory_fractions[i];
        if (frac < 0.0) frac = 0.0;
        raw[i] = frac * (double)total_layers;
        result[i] = (int)std::floor(raw[i]);
        sum += result[i];
    }

    // Distribute the remainder one each, by descending fractional part.
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
        [&raw, &result](size_t a, size_t b) {
            double ra = raw[a] - (double)result[a];
            double rb = raw[b] - (double)result[b];
            if (ra != rb) return ra > rb;
            return a < b;
        });

    int remaining = total_layers - sum;
    for (int k = 0; k < remaining; k++) result[order[(size_t)k]]++;

    // Enforce min 1 layer per node by taking from the largest allocation.
    for (size_t i = 0; i < n; i++) {
        if (result[i] > 0) continue;
        int max_j = -1;
        for (size_t j = 0; j < n; j++) {
            if (j == i || result[j] <= 1) continue;
            if (max_j < 0 || result[j] > result[max_j]) max_j = (int)j;
        }
        if (max_j < 0)
            throw std::runtime_error(
                "allocate_layers_proportionally: cannot guarantee 1 layer/node");
        result[max_j]--;
        result[i]++;
    }
    return result;
}

// ─── Contiguous pipeline shard specs ────────────────────────────────────────
// Splits n_layers into world_size contiguous half-open [start, end) ranges in
// rank order. Layer counts are proportional to each node's available RAM.
// Each node's required RAM share is validated against what it reported.
//
// @param n_layers             total transformer layers in the model
// @param world_size           number of workers
// @param ram_available_bytes  per-node available RAM (bytes), size == world_size
// @param model_size_bytes     total model weight size (bytes), used both for
//                             the proportionality fractions (via RAM) and for
//                             the per-node fit check
// @return shards[i] = ShardSpec for worker i (device_rank == i)
inline std::vector<ShardSpec> compute_shards(
    int n_layers, int world_size,
    const std::vector<int64_t>& ram_available_bytes,
    int64_t model_size_bytes) {
    if (world_size <= 0)
        throw std::runtime_error("compute_shards: world_size must be > 0");
    if (n_layers < world_size)
        throw std::runtime_error("compute_shards: n_layers (" +
                                 std::to_string(n_layers) + ") < world_size (" +
                                 std::to_string(world_size) + ")");
    if ((size_t)world_size != ram_available_bytes.size())
        throw std::runtime_error(
            "compute_shards: ram_available_bytes size mismatch");

    // Fractions = normalized available RAM. If no RAM info at all, fall back
    // to equal split (every node gets 1/world_size).
    std::vector<double> fractions((size_t)world_size, 0.0);
    double total_ram = 0.0;
    for (int64_t b : ram_available_bytes) total_ram += (double)b;
    if (total_ram > 0.0) {
        for (int i = 0; i < world_size; i++)
            fractions[(size_t)i] = (double)ram_available_bytes[(size_t)i] / total_ram;
    } else {
        for (int i = 0; i < world_size; i++)
            fractions[(size_t)i] = 1.0 / (double)world_size;
    }

    std::vector<int> counts =
        allocate_layers_proportionally(n_layers, fractions);

    std::vector<ShardSpec> shards((size_t)world_size);
    int start = 0;
    for (int i = 0; i < world_size; i++) {
        int end = start + counts[(size_t)i];
        ShardSpec s;
        s.device_rank = i;
        s.world_size = world_size;
        s.start_layer = start;
        s.end_layer = end;
        s.n_layers = n_layers;
        s.is_first = (start == 0);
        s.is_last = (end == n_layers);
        shards[(size_t)i] = s;

        // Fit check: this node's RAM must cover its proportional share of the
        // model weights.
        double need =
            (double)model_size_bytes * (double)counts[(size_t)i] / (double)n_layers;
        if ((double)ram_available_bytes[(size_t)i] < need) {
            throw std::runtime_error(
                "compute_shards: node " + std::to_string(i) +
                " has insufficient RAM (available " +
                std::to_string(ram_available_bytes[(size_t)i]) +
                " bytes, needs ~" + std::to_string((int64_t)need) +
                " bytes for " + std::to_string(counts[(size_t)i]) +
                " of " + std::to_string(n_layers) + " layers)");
        }
        start = end;
    }
    return shards;
}

}  // namespace tldist

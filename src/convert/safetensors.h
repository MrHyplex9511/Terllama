/*
 * safetensors.h — Native safetensors format parser (fp32 output).
 *
 * Format: 8-byte little-endian u64 JSON length, then the JSON header, then
 * raw tensor data (8-byte aligned). JSON maps each tensor name to
 * {dtype, shape, data_offsets: [start, end)} where offsets are relative to
 * the start of the data region (offset 8 + json_len).
 *
 * Supported dtypes: F32 (copied), F16 / BF16 (converted to fp32).
 * Other dtypes (I8, I16, I32, F64, ...) are skipped with a warning.
 * Little-endian host assumed (x86_64/aarch64), matching safetensors spec.
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace terllama {

struct STTensor {
    std::string name;
    std::vector<int64_t> shape;
    std::string dtype; // original dtype string, e.g. "BF16"
    std::vector<float> data; // converted to fp32
};

// Parse a complete safetensors buffer. Returns false on structural errors
// (truncated header, bad offsets, oversized values). Tensors with
// unsupported dtypes are skipped with a warning — parse still succeeds.
bool parse_safetensors(const std::vector<uint8_t>& buf,
                       std::vector<STTensor>& out);

// Read an entire safetensors file from disk and parse it.
bool load_safetensors_file(const std::string& path,
                           std::vector<STTensor>& out);

} // namespace terllama

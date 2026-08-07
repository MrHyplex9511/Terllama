/*
 * safetensors.cpp — Native safetensors parser (fp32 output).
 *
 * Layout: [u64 LE json_len][json header][raw data @ 8+json_len, 8-aligned]
 * Offsets in the JSON are relative to the start of the data region.
 * Supports F32 / F16 / BF16; other dtypes skipped with a warning.
 */
#include "convert/safetensors.h"

#include "core/logger.h"
#include <json.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>

using json = nlohmann::json;

namespace terllama {

namespace {

// IEEE-754 binary16 -> fp32.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1Fu;
    const uint32_t man  = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0x1Fu) { // inf / nan
        bits = sign | 0x7F800000u | (man << 13);
    } else if (exp == 0) {
        if (man == 0) {
            bits = sign; // +/- zero
        } else {
            // subnormal: normalize into normal range
            int e = 127 - 15;
            uint32_t m = man;
            while (!(m & 0x400u)) { m <<= 1; --e; }
            m &= 0x3FFu;
            bits = sign | ((uint32_t)e << 23) | (m << 13);
        }
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

// bfloat16 -> fp32 (8-bit exponent, 7-bit mantissa).
float bf16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 7) & 0xFFu;
    const uint32_t man  = h & 0x7Fu;
    uint32_t bits;
    if (exp == 0xFFu) { // inf / nan
        bits = sign | 0x7F800000u | (man << 16);
    } else if (exp == 0) {
        if (man == 0) {
            bits = sign; // +/- zero
        } else {
            int e = 127 - 126;
            uint32_t m = man;
            while (!(m & 0x80u)) { m <<= 1; --e; }
            m &= 0x7Fu;
            bits = sign | ((uint32_t)e << 23) | (m << 16);
        }
    } else {
        bits = sign | (exp << 23) | (man << 16);
    }
    float out;
    std::memcpy(&out, &bits, sizeof out);
    return out;
}

} // namespace

bool parse_safetensors(const std::vector<uint8_t>& buf,
                       std::vector<STTensor>& out) {
    out.clear();

    // 8-byte little-endian JSON length.
    if (buf.size() < 8) {
        Logger::error("safetensors: buffer smaller than header ({} bytes)",
                      buf.size());
        return false;
    }
    uint64_t json_len = 0;
    for (int i = 0; i < 8; ++i) {
        json_len |= static_cast<uint64_t>(buf[i]) << (8 * i);
    }
    if (json_len > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        json_len > buf.size() - 8) {
        Logger::error("safetensors: header length {} exceeds buffer", json_len);
        return false;
    }

    const uint8_t* json_start = buf.data() + 8;
    const std::string json_str(reinterpret_cast<const char*>(json_start),
                               static_cast<size_t>(json_len));

    // Data region must be 8-byte aligned (safetensors pads the JSON).
    const size_t data_off = 8 + static_cast<size_t>(json_len);
    if (data_off % 8 != 0) {
        Logger::warn("safetensors: data offset {} not 8-aligned (padded JSON "
                     "expected)", data_off);
        return false;
    }

    json header;
    try {
        header = json::parse(json_str);
    } catch (const std::exception& e) {
        Logger::error("safetensors: header JSON parse failed: {}", e.what());
        return false;
    }
    if (!header.is_object()) {
        Logger::error("safetensors: header is not a JSON object{}", "");
        return false;
    }

    const uint8_t* data = buf.data() + data_off;
    const size_t data_size = buf.size() - data_off;

    int skipped = 0;
    for (auto it = header.begin(); it != header.end(); ++it) {
        // Reserved keys (e.g. "__metadata__") carry non-tensor data.
        if (it.key().size() >= 2 && it.key().compare(0, 2, "__") == 0) {
            continue;
        }
        if (!it.value().is_object()) continue;
        const json& t = it.value();

        const std::string dtype = t.value("dtype", "");
        const std::string name = it.key();

        // Shape.
        std::vector<int64_t> shape;
        if (t.contains("shape") && t["shape"].is_array()) {
            for (const auto& d : t["shape"]) {
                if (!d.is_number_integer()) {
                    Logger::error("safetensors: non-integer shape in '{}'",
                                  name);
                    return false;
                }
                const auto v = d.get<int64_t>();
                if (v < 0) {
                    Logger::error("safetensors: negative shape dim in '{}'",
                                  name);
                    return false;
                }
                shape.push_back(v);
            }
        } else {
            Logger::error("safetensors: tensor '{}' has no shape array", name);
            return false;
        }

        // Data offsets.
        uint64_t start = 0, end = 0;
        if (t.contains("data_offsets") && t["data_offsets"].is_array() &&
            t["data_offsets"].size() == 2 &&
            t["data_offsets"][0].is_number_integer() &&
            t["data_offsets"][1].is_number_integer()) {
            start = t["data_offsets"][0].get<uint64_t>();
            end = t["data_offsets"][1].get<uint64_t>();
        } else {
            Logger::error("safetensors: tensor '{}' has bad data_offsets",
                          name);
            return false;
        }
        if (start > end || end > data_size) {
            Logger::error("safetensors: bad data range [{},{}) for '{}'",
                          start, end, name);
            return false;
        }

        STTensor tt;
        tt.name = name;
        tt.dtype = dtype;
        tt.shape = shape;

        // Element count (guarded against overflow).
        uint64_t nelem = 1;
        for (int64_t d : shape) {
            if (d != 0 && nelem > std::numeric_limits<uint64_t>::max() /
                                     static_cast<uint64_t>(d)) {
                Logger::error("safetensors: shape overflow for '{}'", name);
                return false;
            }
            nelem *= static_cast<uint64_t>(d);
        }

        // Bytes per element for the raw dtype.
        uint64_t esize = 0;
        if (dtype == "F32") esize = 4;
        else if (dtype == "F16") esize = 2;
        else if (dtype == "BF16") esize = 2;
        else {
            Logger::warn("safetensors: skipping '{}' (unsupported dtype {})",
                         name, dtype);
            ++skipped;
            continue;
        }

        const uint64_t expected = nelem * esize;
        if (expected != end - start) {
            Logger::error("safetensors: size mismatch for '{}' (need {} bytes, "
                          "have {})", name, expected, end - start);
            return false;
        }
        if (nelem > static_cast<uint64_t>(std::vector<float>().max_size())) {
            Logger::error("safetensors: '{}' too large for fp32", name);
            return false;
        }

        tt.data.resize(static_cast<size_t>(nelem));
        const uint8_t* src = data + start;
        if (dtype == "F32") {
            std::memcpy(tt.data.data(), src, tt.data.size() * sizeof(float));
        } else if (dtype == "F16") {
            for (size_t i = 0; i < tt.data.size(); ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                tt.data[i] = f16_to_f32(h);
            }
        } else { // BF16
            for (size_t i = 0; i < tt.data.size(); ++i) {
                uint16_t h;
                std::memcpy(&h, src + i * 2, 2);
                tt.data[i] = bf16_to_f32(h);
            }
        }

        out.push_back(std::move(tt));
    }

    if (skipped > 0) {
        Logger::warn("safetensors: skipped {} tensor(s) with unsupported "
                     "dtypes", skipped);
    }
    return true;
}

bool load_safetensors_file(const std::string& path,
                           std::vector<STTensor>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        Logger::error("safetensors: cannot open '{}'", path);
        return false;
    }
    const std::streampos size = f.tellg();
    if (size < 0) {
        Logger::error("safetensors: cannot size '{}'", path);
        return false;
    }
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    if (size > 0) {
        f.read(reinterpret_cast<char*>(buf.data()),
               static_cast<std::streamsize>(buf.size()));
        if (!f) {
            Logger::error("safetensors: short read on '{}'", path);
            return false;
        }
    }
    return parse_safetensors(buf, out);
}

} // namespace terllama

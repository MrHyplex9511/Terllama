#include "core/tokenizer.h"
#include <sstream>
#include <algorithm>
#include <cctype>

bool Tokenizer::load_from_gguf(const std::vector<std::string>& tokens,
                                const std::vector<float>& scores_in,
                                const std::vector<int32_t>& types_in,
                                const std::string& model,
                                int bos, int eos) {
    vocab = tokens;
    scores = scores_in;
    types = types_in;
    model_type = model;
    bos_id = bos;
    eos_id = eos;
    valid = true;
    return true;
}

// Check if a token string is a byte-fallback like "<0x0A>"
static bool is_byte_fallback(const std::string& s, uint8_t& byte_val) {
    if (s.size() != 6) return false;
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x') return false;
    if (s[5] != '>') return false;
    auto hex_val = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    };
    int hi = hex_val(s[3]);
    int lo = hex_val(s[4]);
    if (hi < 0 || lo < 0) return false;
    byte_val = (uint8_t)((hi << 4) | lo);
    return true;
}

std::string Tokenizer::decode(const std::vector<int>& token_ids) const {
    if (!valid || vocab.empty()) return "?";

    std::ostringstream oss;

    for (int id : token_ids) {
        // Out of range
        if (id < 0 || id >= (int)vocab.size()) continue;

        // Skip control tokens (type 3), BOS/EOS
        if (id < (int)types.size() && types[id] == 3) continue;
        if (id == bos_id || id == eos_id) continue;

        std::string token = vocab[id];

        if (model_type == "llama") {
            // SentencePiece decoding

            // Handle byte-fallback: "<0xNN>" → single byte
            uint8_t byte_val;
            if (is_byte_fallback(token, byte_val)) {
                oss.put((char)byte_val);
                continue;
            }

            // Replace ▁ (UTF-8: 0xE2 0x96 0x81) with space (0x20)
            std::string processed;
            processed.reserve(token.size());
            size_t i = 0;
            while (i < token.size()) {
                if (i + 2 < token.size() &&
                    (uint8_t)token[i]     == 0xE2 &&
                    (uint8_t)token[i + 1] == 0x96 &&
                    (uint8_t)token[i + 2] == 0x81) {
                    processed += ' ';
                    i += 3;
                } else {
                    processed += token[i];
                    i++;
                }
            }
            oss << processed;

        } else {
            // GPT-2 BPE: concatenate raw strings
            oss << token;
        }
    }

    std::string result = oss.str();

    // Strip leading space (SentencePiece adds one)
    if (model_type == "llama" && !result.empty() && result[0] == ' ') {
        result.erase(0, 1);
    }

    return result;
}

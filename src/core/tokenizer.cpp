#include "core/tokenizer.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <fstream>

#include <json.hpp>

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

bool Tokenizer::load_from_tokenizer_json(const std::string& path) {
    std::ifstream f(path);
    if (!f) return false;
    nlohmann::json j;
    try { f >> j; } catch (...) { return false; }
    if (!j.contains("model")) return false;

    const auto& model = j["model"];
    std::string type = model.value("type", std::string());
    if (type != "BPE" && type != "Unigram" && type != "WordPiece" &&
        type != "SentencePiece") {
        // unknown tokenizer type — bail, caller will fall back to Python
        return false;
    }

    // Byte-level BPE → gpt2 model_type; others → llama (SentencePiece-like)
    bool byte_level = false;
    if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object() &&
        j["pre_tokenizer"].value("type", std::string()) == "Sequence") {
        // check nested ByteLevel
        for (const auto& pt : j["pre_tokenizer"]["pretokenizers"]) {
            if (pt.value("type", std::string()) == "ByteLevel") byte_level = true;
        }
    } else if (j.contains("pre_tokenizer") && j["pre_tokenizer"].is_object() &&
               j["pre_tokenizer"].value("type", std::string()) == "ByteLevel") {
        byte_level = true;
    }

    model_type = (type == "BPE" && byte_level) ? "gpt2" : "llama";
    // Build vocab: id → string (vocab is string → id)
    std::vector<std::pair<int, std::string>> ordered;
    if (model.contains("vocab") && model["vocab"].is_object()) {
        for (auto it = model["vocab"].begin(); it != model["vocab"].end(); ++it) {
            ordered.emplace_back(it.value().get<int>(), it.key());
        }
    }
    // Added tokens
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& at : j["added_tokens"]) {
            if (!at.is_object()) continue;
            int id = at.value("id", -1);
            std::string content = at.value("content", std::string());
            if (id >= 0 && !content.empty()) ordered.emplace_back(id, content);
        }
    }
    if (ordered.empty()) return false;

    std::sort(ordered.begin(), ordered.end());
    vocab.clear();
    vocab.reserve(ordered.size());
    for (const auto& [id, s] : ordered) {
        if (id < 0) continue;
        // pad gaps (e.g. id 0..vocab.size()-1)
        while ((int)vocab.size() < id) vocab.push_back("▁");
        if ((int)vocab.size() == id) vocab.push_back(s);
    }

    scores.assign(vocab.size(), 0.0f);
    types.assign(vocab.size(), 0);
    // Mark special tokens (control type 3) so decode skips them
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& at : j["added_tokens"]) {
            if (!at.is_object()) continue;
            int id = at.value("id", -1);
            bool special = at.value("special", false);
            if (id >= 0 && id < (int)types.size() && special) types[id] = 3;
        }
    }

    // BOS/EOS from added_tokens / special tokens
    bos_id = -1; eos_id = -1;
    int first_special = -1;
    if (j.contains("added_tokens") && j["added_tokens"].is_array()) {
        for (const auto& at : j["added_tokens"]) {
            if (!at.is_object()) continue;
            int id = at.value("id", -1);
            std::string content = at.value("content", std::string());
            bool special = at.value("special", false);
            if (content == "<|endoftext|>" && id >= 0) {
                if (bos_id < 0) bos_id = id;
                eos_id = id;
            }
            if (special && first_special < 0) first_special = id;
        }
    }
    if (bos_id < 0) bos_id = first_special;
    if (eos_id < 0) eos_id = first_special;

    // Native decode can only handle SentencePiece-style ("llama") tokens.
    // Byte-level BPE ("gpt2") tokens are byte-encoded unicode (Ġ etc.) that
    // only GigaToken / Python can decode correctly, so leave valid=false —
    // callers must use the GigaToken wrapper (see handlers.cpp / commands.cpp).
    valid = (model_type == "llama");
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
    if (model_type == "gpt2") return "?";  // byte-level BPE — use GigaToken

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
            // GPT-2 BPE (non-byte-level): concatenate raw strings
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

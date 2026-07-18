#pragma once
#include <string>
#include <vector>
#include <cstdint>

struct Tokenizer {
    std::vector<std::string> vocab;
    std::vector<float> scores;
    std::vector<int32_t> types;
    std::string model_type;  // "llama" or "gpt2"
    int bos_id{-1}, eos_id{-1};
    bool valid{false};

    bool load_from_gguf(const std::vector<std::string>& tokens,
                        const std::vector<float>& scores_in,
                        const std::vector<int32_t>& types_in,
                        const std::string& model,
                        int bos, int eos);

    std::string decode(const std::vector<int>& token_ids) const;
};

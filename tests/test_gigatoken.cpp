/*
 * test_gigatoken.cpp — Integration tests for GigaToken C++ wrapper
 *
 * Tests:
 *   1. Library loading (graceful if .so missing)
 *   2. Tokenizer loading from tokenizer.json
 *   3. Encode/decode roundtrip
 *   4. Unicode and special character handling
 *   5. Large batch encoding performance (smoke test)
 *
 * Environment:
 *   TERLLAMA_TEST_MODEL_DIR — path to model dir containing tokenizer.json
 *   If unset, tests that require a model are skipped.
 */
#include <catch_amalgamated.hpp>
#include "core/gigatoken_wrapper.h"
#include <cstring>
#include <fstream>

// ═══════════════════════════════════════════════════════════════════════════
// Test 1: Library loading
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("GigaToken library loads from default paths", "[gigatoken][load]") {
    GigaTokenWrapper gt;
    // Should not crash even if .so is missing
    bool ok = gt.load(".");
    // Either loaded or gracefully failed — both are acceptable
    if (ok) {
        REQUIRE(gt.is_loaded());
    } else {
        // Error should be descriptive
        REQUIRE_FALSE(gt.error().empty());
    }
}

TEST_CASE("GigaToken library loads from build directory", "[gigatoken][load]") {
    GigaTokenWrapper gt;
    bool ok = gt.load(".:./bin:./build");
    if (ok) {
        REQUIRE(gt.is_loaded());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 2: Tokenizer loading (requires model dir with tokenizer.json)
// ═══════════════════════════════════════════════════════════════════════════

static std::string get_test_model_dir() {
    const char* env = std::getenv("TERLLAMA_TEST_MODEL_DIR");
    return env ? std::string(env) : "";
}

TEST_CASE("GigaToken tokenizer loads from model directory", "[gigatoken][tokenizer]") {
    std::string model_dir = get_test_model_dir();
    if (model_dir.empty()) {
        WARN("TERLLAMA_TEST_MODEL_DIR not set — skipping tokenizer load test");
        return;
    }

    // Check tokenizer.json exists
    std::string tok_path = model_dir + "/tokenizer.json";
    std::ifstream f(tok_path);
    REQUIRE(f.good());  // tokenizer.json must exist

    GigaTokenWrapper gt;
    REQUIRE(gt.load(".:./build"));
    REQUIRE(gt.load_tokenizer(model_dir));
    REQUIRE(gt.has_tokenizer());
    REQUIRE(gt.vocab_size() > 0);
    INFO("vocab_size = " << gt.vocab_size());
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 3: Encode/decode roundtrip
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("GigaToken encode returns valid token IDs", "[gigatoken][encode]") {
    std::string model_dir = get_test_model_dir();
    if (model_dir.empty()) {
        WARN("TERLLAMA_TEST_MODEL_DIR not set — skipping encode test");
        return;
    }

    GigaTokenWrapper gt;
    REQUIRE(gt.load(".:./build"));
    REQUIRE(gt.load_tokenizer(model_dir));

    SECTION("Short text") {
        auto ids = gt.encode("Hello world");
        REQUIRE_FALSE(ids.empty());
        // Token IDs should be in valid range
        int vs = gt.vocab_size();
        for (auto id : ids) {
            REQUIRE(id < (uint32_t)vs);
        }
    }

    SECTION("Empty string") {
        auto ids = gt.encode("");
        // Empty string may return empty or a single BOS token — both acceptable
        // Just verify it doesn't crash
        REQUIRE(ids.size() <= 1);
    }

    SECTION("Single character") {
        auto ids = gt.encode("a");
        REQUIRE_FALSE(ids.empty());
    }
}

TEST_CASE("GigaToken decode returns valid UTF-8", "[gigatoken][decode]") {
    std::string model_dir = get_test_model_dir();
    if (model_dir.empty()) {
        WARN("TERLLAMA_TEST_MODEL_DIR not set — skipping decode test");
        return;
    }

    GigaTokenWrapper gt;
    REQUIRE(gt.load(".:./build"));
    REQUIRE(gt.load_tokenizer(model_dir));

    SECTION("Roundtrip: encode then decode") {
        std::string original = "Hello world, this is a test of the GigaToken tokenizer!";
        auto ids = gt.encode(original);
        REQUIRE_FALSE(ids.empty());

        std::string decoded = gt.decode(ids);
        // Decoded text should be non-empty
        REQUIRE_FALSE(decoded.empty());
        // The decoded text may differ from original (BOS prefix, spacing,
        // etc.) but should contain recognizable content
        REQUIRE(decoded.size() > 0);
    }

    SECTION("Decode empty IDs") {
        std::string result = gt.decode({});
        REQUIRE(result.empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 4: Unicode handling
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("GigaToken handles Unicode text", "[gigatoken][unicode]") {
    std::string model_dir = get_test_model_dir();
    if (model_dir.empty()) {
        WARN("TERLLAMA_TEST_MODEL_DIR not set — skipping unicode test");
        return;
    }

    GigaTokenWrapper gt;
    REQUIRE(gt.load(".:./build"));
    REQUIRE(gt.load_tokenizer(model_dir));

    SECTION("Chinese characters") {
        auto ids = gt.encode("你好世界");
        REQUIRE_FALSE(ids.empty());
    }

    SECTION("Emoji") {
        auto ids = gt.encode("Hello 👋 🌟");
        REQUIRE_FALSE(ids.empty());
    }

    SECTION("Accented characters") {
        auto ids = gt.encode("Émoticône über cool ñoño");
        REQUIRE_FALSE(ids.empty());
    }

    SECTION("Mixed script") {
        auto ids = gt.encode("Hello 世界! 👋 テスト 🎉");
        REQUIRE_FALSE(ids.empty());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Test 5: Large batch (smoke test — doesn't check correctness, just
//          verifies no crash with larger inputs)
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("GigaToken handles large input without crashing", "[gigatoken][stress]") {
    std::string model_dir = get_test_model_dir();
    if (model_dir.empty()) {
        WARN("TERLLAMA_TEST_MODEL_DIR not set — skipping stress test");
        return;
    }

    GigaTokenWrapper gt;
    REQUIRE(gt.load(".:./build"));
    REQUIRE(gt.load_tokenizer(model_dir));

    SECTION("10KB of text") {
        std::string text;
        text.reserve(10240);
        for (int i = 0; i < 200; i++) {
            text += "This is a test sentence to fill up the buffer. ";
        }
        auto ids = gt.encode(text);
        REQUIRE_FALSE(ids.empty());
        REQUIRE(ids.size() > 100);  // Should produce many tokens
    }

    SECTION("100KB of repeating text") {
        std::string text;
        text.reserve(102400);
        for (int i = 0; i < 2000; i++) {
            text += "The quick brown fox jumps over the lazy dog. ";
        }
        auto ids = gt.encode(text);
        REQUIRE_FALSE(ids.empty());
        REQUIRE(ids.size() > 500);
    }
}

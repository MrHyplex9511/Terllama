// ═══════════════════════════════════════════════════════════════════════════
// GigaToken C++ Wrapper
// ═══════════════════════════════════════════════════════════════════════════
//
// Loads libgigatoken_rs.so at runtime via dlopen/dlsym and exposes a clean
// C++ interface. Falls back gracefully when the .so is not present.
#pragma once

#include <string>
#include <vector>
#include <memory>

class GigaTokenWrapper {
public:
    /// Construct wrapper. Does NOT load the .so yet — call load().
    GigaTokenWrapper();

    /// Destructor — unloads library and frees tokenizer.
    ~GigaTokenWrapper();

    // Not copyable
    GigaTokenWrapper(const GigaTokenWrapper&) = delete;
    GigaTokenWrapper& operator=(const GigaTokenWrapper&) = delete;

    // Movable
    GigaTokenWrapper(GigaTokenWrapper&& other) noexcept;
    GigaTokenWrapper& operator=(GigaTokenWrapper&& other) noexcept;

    /// Load libgigatoken_rs.so and initialize.
    /// @param search_paths  colon-separated dirs to search (e.g. ".:./lib")
    /// @return true on success, false if library couldn't be loaded.
    bool load(const std::string& search_paths = ".");

    /// Check if the library was loaded successfully.
    bool is_loaded() const { return lib_ != nullptr; }

    /// Check if a tokenizer is loaded.
    bool has_tokenizer() const { return tok_ != nullptr; }

    /// Load a HuggingFace tokenizer from a directory containing tokenizer.json.
    /// @param path  directory path
    /// @return true on success
    bool load_tokenizer(const std::string& path);

    /// Encode UTF-8 text to token IDs.
    /// @return empty vector on error
    std::vector<uint32_t> encode(const std::string& text);

    /// Decode token IDs back to UTF-8 text.
    /// @return empty string on error
    std::string decode(const std::vector<uint32_t>& ids);

    /// Get vocabulary size.
    int32_t vocab_size() const;

    /// Get last error message.
    const std::string& error() const { return err_; }

private:
    void* lib_{nullptr};              // dlopen handle
    void* tok_{nullptr};              // GigaTokenizer* opaque handle

    std::string err_;

    // Function pointer type aliases
    using gt_init_fn           = int32_t (*)(void);
    using gt_cleanup_fn        = void    (*)(void);
    using gt_load_hf_fn        = void*   (*)(const char*);
    using gt_free_fn           = void    (*)(void*);
    using gt_encode_fn         = int32_t (*)(void*, const char*, int32_t, uint32_t*, int32_t*, int32_t);
    using gt_decode_fn         = int32_t (*)(void*, const uint32_t*, int32_t, char*, int32_t*, int32_t);
    using gt_vocab_size_fn     = int32_t (*)(const void*);

    // Resolved function pointers
    gt_init_fn      fn_init_{nullptr};
    gt_cleanup_fn   fn_cleanup_{nullptr};
    gt_load_hf_fn   fn_load_hf_{nullptr};
    gt_free_fn      fn_free_{nullptr};
    gt_encode_fn    fn_encode_{nullptr};
    gt_decode_fn    fn_decode_{nullptr};
    gt_vocab_size_fn fn_vocab_size_{nullptr};
};

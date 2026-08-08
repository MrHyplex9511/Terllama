// ═══════════════════════════════════════════════════════════════════════════
// GigaToken C++ Wrapper — implementation
// ═══════════════════════════════════════════════════════════════════════════
#include "gigatoken_wrapper.h"

#include <dlfcn.h>
#include <cstring>
#include <limits>
#include <sstream>

GigaTokenWrapper::GigaTokenWrapper() = default;

GigaTokenWrapper::~GigaTokenWrapper() {
    if (tok_ && fn_free_) fn_free_(tok_);
    if (lib_) {
        if (fn_cleanup_) fn_cleanup_();
        dlclose(lib_);
    }
}

GigaTokenWrapper::GigaTokenWrapper(GigaTokenWrapper&& other) noexcept
    : lib_(other.lib_), tok_(other.tok_), err_(std::move(other.err_)),
      fn_init_(other.fn_init_), fn_cleanup_(other.fn_cleanup_),
      fn_load_hf_(other.fn_load_hf_), fn_free_(other.fn_free_),
      fn_encode_(other.fn_encode_), fn_decode_(other.fn_decode_),
      fn_vocab_size_(other.fn_vocab_size_)
{
    other.lib_ = nullptr;
    other.tok_ = nullptr;
    other.fn_init_ = nullptr;
    other.fn_cleanup_ = nullptr;
    other.fn_load_hf_ = nullptr;
    other.fn_free_ = nullptr;
    other.fn_encode_ = nullptr;
    other.fn_decode_ = nullptr;
    other.fn_vocab_size_ = nullptr;
}

GigaTokenWrapper& GigaTokenWrapper::operator=(GigaTokenWrapper&& other) noexcept {
    if (this != &other) {
        // Clean up current
        if (tok_ && fn_free_) fn_free_(tok_);
        if (lib_) {
            if (fn_cleanup_) fn_cleanup_();
            dlclose(lib_);
        }
        // Move
        lib_ = other.lib_;          other.lib_ = nullptr;
        tok_ = other.tok_;          other.tok_ = nullptr;
        err_ = std::move(other.err_);
        fn_init_ = other.fn_init_;         other.fn_init_ = nullptr;
        fn_cleanup_ = other.fn_cleanup_;   other.fn_cleanup_ = nullptr;
        fn_load_hf_ = other.fn_load_hf_;   other.fn_load_hf_ = nullptr;
        fn_free_ = other.fn_free_;         other.fn_free_ = nullptr;
        fn_encode_ = other.fn_encode_;     other.fn_encode_ = nullptr;
        fn_decode_ = other.fn_decode_;     other.fn_decode_ = nullptr;
        fn_vocab_size_ = other.fn_vocab_size_; other.fn_vocab_size_ = nullptr;
    }
    return *this;
}

bool GigaTokenWrapper::load(const std::string& search_paths) {
    // Try each directory in search_paths
    std::istringstream ss(search_paths);
    std::string dir;
    while (std::getline(ss, dir, ':')) {
        std::string path = dir + "/libgigatoken_rs.so";
        lib_ = dlopen(path.c_str(), RTLD_NOW);
        if (lib_) break;
    }
    if (!lib_) {
        err_ = "dlopen(libgigatoken_rs.so): ";
        err_ += dlerror();
        return false;
    }

    // Resolve all function pointers
    auto resolve = [&](const char* name) -> void* {
        void* sym = dlsym(lib_, name);
        if (!sym) {
            err_ = "dlsym(" + std::string(name) + "): ";
            err_ += dlerror();
        }
        return sym;
    };

    fn_init_       = reinterpret_cast<gt_init_fn>(resolve("gt_init"));
    fn_cleanup_    = reinterpret_cast<gt_cleanup_fn>(resolve("gt_cleanup"));
    fn_load_hf_    = reinterpret_cast<gt_load_hf_fn>(resolve("gt_tokenizer_load_hf"));
    fn_free_       = reinterpret_cast<gt_free_fn>(resolve("gt_tokenizer_free"));
    fn_encode_     = reinterpret_cast<gt_encode_fn>(resolve("gt_encode"));
    fn_decode_     = reinterpret_cast<gt_decode_fn>(resolve("gt_decode"));
    fn_vocab_size_ = reinterpret_cast<gt_vocab_size_fn>(resolve("gt_vocab_size"));

    if (!fn_init_ || !fn_cleanup_ || !fn_load_hf_ || !fn_free_ ||
        !fn_encode_ || !fn_decode_ || !fn_vocab_size_) {
        dlclose(lib_);
        lib_ = nullptr;
        return false;
    }

    fn_init_();
    return true;
}

bool GigaTokenWrapper::load_tokenizer(const std::string& path) {
    if (!lib_ || !fn_load_hf_) {
        err_ = "library not loaded";
        return false;
    }
    // Free existing tokenizer if any
    if (tok_ && fn_free_) fn_free_(tok_);
    tok_ = fn_load_hf_(path.c_str());
    if (!tok_) {
        err_ = "gt_tokenizer_load_hf failed for: " + path;
        return false;
    }
    return true;
}

std::vector<uint32_t> GigaTokenWrapper::encode(const std::string& text) {
    if (!tok_ || !fn_encode_) return {};

    // Start with a buffer scaled to the input instead of a fixed 262144-token
    // (~1MiB) allocation on every call. The C API writes up to `cap` tokens
    // and reports the actual count via `out_len`. If out_len == cap the result
    // may have been truncated, so we double and retry.
    int64_t initial = (int64_t)text.size() / 4 + 64;   // ≈4 chars per token
    int32_t cap = 64;
    if (initial > cap)
        cap = static_cast<int32_t>(
            std::min<int64_t>(initial, std::numeric_limits<int32_t>::max()));
    std::vector<uint32_t> ids;
    int32_t out_len = 0;
    for (;;) {
        ids.resize(cap);
        out_len = 0;
        int32_t ret = fn_encode_(tok_, text.data(), static_cast<int32_t>(text.size()),
                                 ids.data(), &out_len, cap);
        if (ret != 0) {
            err_ = "gt_encode failed with code " + std::to_string(ret);
            return {};
        }
        if (out_len < cap) break;  // complete result
        // Truncated — the text is larger than our guess; grow and retry
        cap *= 2;
    }
    ids.resize(out_len);
    return ids;
}

std::string GigaTokenWrapper::decode(const std::vector<uint32_t>& ids) {
    if (!tok_ || !fn_decode_ || ids.empty()) return {};

    const int32_t cap = 65536;

    // Reuse a per-thread buffer instead of allocating a fresh 64KiB string on
    // every call. resize() only reallocates when capacity shrank below cap;
    // in steady-state (16-token streaming batches) the 64KiB block is
    // allocated once per thread and reused for all subsequent calls.
    thread_local std::string buf;
    buf.resize(cap);
    int32_t out_len = 0;
    int32_t ret = fn_decode_(tok_, ids.data(), static_cast<int32_t>(ids.size()),
                             buf.data(), &out_len, cap);
    if (ret != 0) {
        err_ = "gt_decode failed with code " + std::to_string(ret);
        return {};
    }
    buf.resize(out_len);
    return buf;
}

int32_t GigaTokenWrapper::vocab_size() const {
    if (!tok_ || !fn_vocab_size_) return 0;
    return fn_vocab_size_(tok_);
}

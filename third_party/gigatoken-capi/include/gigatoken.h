// ═══════════════════════════════════════════════════════════════════════════
// GigaToken C API — header for C/C++ consumers
// ═══════════════════════════════════════════════════════════════════════════
//
// Load at runtime:  void *lib = dlopen("libgigatoken_rs.so", RTLD_NOW);
// Get symbols:      auto fn = (gt_tokenizer_load_hf_fn)dlsym(lib, "gt_tokenizer_load_hf");
//
// See gigatoken_wrapper.h for the higher-level C++ API.
#ifndef GIGATOKEN_H
#define GIGATOKEN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Error codes ─────────────────────────────────────────────────────────

#define GT_OK         0
#define GT_ERR_LOAD  -1
#define GT_ERR_PARSE -2

// ─── Opaque handle ───────────────────────────────────────────────────────

typedef struct GigaTokenizer GigaTokenizer;

// ─── Lifecycle ───────────────────────────────────────────────────────────

int32_t gt_init(void);
void    gt_cleanup(void);

// ─── Tokenizer creation / destruction ────────────────────────────────────

GigaTokenizer* gt_tokenizer_load_hf(const char* path);
void           gt_tokenizer_free(GigaTokenizer* tok);

// ─── Encode: text → token IDs ────────────────────────────────────────────

int32_t gt_encode(
    GigaTokenizer* tok,
    const char*    text,
    int32_t        text_len,
    uint32_t*      ids,
    int32_t*       out_len,
    int32_t        capacity
);

// ─── Decode: token IDs → text (null-terminated) ──────────────────────────

int32_t gt_decode(
    GigaTokenizer* tok,
    const uint32_t* ids,
    int32_t        id_count,
    char*          out,
    int32_t*       out_len,
    int32_t        capacity
);

// ─── Introspection ───────────────────────────────────────────────────────

int32_t gt_vocab_size(const GigaTokenizer* tok);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GIGATOKEN_H

// ═══════════════════════════════════════════════════════════════════════════
// GigaToken C API — exported symbols for C/C++ consumers (via dlopen/dlsym)
// ═══════════════════════════════════════════════════════════════════════════
//
// These symbols are exported by the cdylib (.so / .dylib) build.
// C/C++ code loads the .so with dlopen() and retrieves function pointers
// with dlsym(). See capi/gigatoken.h for the C header.

#![allow(non_camel_case_types, dead_code)]

use crate::bpe::tiktoken::Tokenizer as BPETokenizer;
use crate::bpe::sentencepiece::{SentencePieceBPE, EncodeState};
use crate::load_tokenizer::hf;
use crate::token::TokenId;
use std::sync::Mutex;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::path::PathBuf;

// ─── Error codes ─────────────────────────────────────────────────────────

pub const GT_OK: i32 = 0;
pub const GT_ERR_LOAD: i32 = -1;
pub const GT_ERR_PARSE: i32 = -2;

// ─── Opaque handle (internal enum) ───────────────────────────────────────

pub enum GigaTokenizer {
    BPE(Mutex<BPETokenizer>),
    SentencePiece {
        tok: Mutex<SentencePieceBPE>,
        state: Mutex<EncodeState>,
    },
}

// ─── Lifecycle ───────────────────────────────────────────────────────────

#[unsafe(no_mangle)]
pub extern "C" fn gt_init() -> i32 {
    GT_OK
}

#[unsafe(no_mangle)]
pub extern "C" fn gt_cleanup() {}

// ─── Tokenizer creation ──────────────────────────────────────────────────

/// Load tokenizer from a directory containing `tokenizer.json`.
/// Returns opaque handle, or NULL on failure.
/// Caller must free with `gt_tokenizer_free()`.
#[unsafe(no_mangle)]
pub extern "C" fn gt_tokenizer_load_hf(path: *const c_char) -> *mut GigaTokenizer {
    if path.is_null() {
        return std::ptr::null_mut();
    }
    let cstr = unsafe { CStr::from_ptr(path) };
    let dir_str = match cstr.to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    // Join directory path with "tokenizer.json" to get the file path
    let mut json_path = PathBuf::from(dir_str);
    if json_path.is_dir() {
        json_path.push("tokenizer.json");
    }
    // If it's already a path to tokenizer.json, use it as-is
    let json_path_str = match json_path.to_str() {
        Some(s) => s.to_string(),
        None => return std::ptr::null_mut(),
    };

    // Try SentencePiece (byte_fallback, Llama-style) first, then BPE (GPT-2 style)
    match hf::load_hf_sentencepiece(&json_path_str) {
        Ok(sp) => {
            let tok = Box::new(GigaTokenizer::SentencePiece {
                tok: Mutex::new(sp),
                state: Mutex::new(EncodeState::new()),
            });
            Box::into_raw(tok)
        }
        Err(e1) => match hf::load_hf_bpe(&json_path_str) {
            Ok(bpe) => {
                let tok = Box::new(GigaTokenizer::BPE(Mutex::new(bpe)));
                Box::into_raw(tok)
            }
            Err(e2) => {
                eprintln!(
                    "[gt_tokenizer_load_hf] SentencePiece error: {e1}\n\
                     [gt_tokenizer_load_hf] BPE error: {e2}"
                );
                std::ptr::null_mut()
            }
        },
    }
}

/// Free a tokenizer handle created by `gt_tokenizer_load_hf()`.
#[unsafe(no_mangle)]
pub extern "C" fn gt_tokenizer_free(tok: *mut GigaTokenizer) {
    if !tok.is_null() {
        unsafe { drop(Box::from_raw(tok)); }
    }
}

// ─── Encode ──────────────────────────────────────────────────────────────

/// Encode UTF-8 text to token IDs.
///
/// Parameters:
///   tok       — tokenizer handle
///   text      — UTF-8 input bytes
///   text_len  — byte length of text
///   ids       — output buffer for token IDs
///   out_len   — in: capacity (max IDs), out: actual IDs written
///   capacity  — max number of u32 entries in ids buffer
///
/// Returns: GT_OK on success, negative on error.
#[unsafe(no_mangle)]
pub extern "C" fn gt_encode(
    tok: *mut GigaTokenizer,
    text: *const c_char,
    text_len: i32,
    ids: *mut u32,
    out_len: *mut i32,
    capacity: i32,
) -> i32 {
    if tok.is_null() || text.is_null() || ids.is_null() || out_len.is_null() {
        return GT_ERR_LOAD;
    }
    let tok = unsafe { &mut *tok };
    let text_slice =
        unsafe { std::slice::from_raw_parts(text as *const u8, text_len as usize) };
    let cap = capacity as usize;

    match tok {
        GigaTokenizer::BPE(mtx) => {
            let mut bpe = mtx.lock().unwrap();
            let mut out = Vec::new();
            bpe.encode_with_added_tokens_flat(text_slice, &mut out);
            let n = out.len().min(cap);
            unsafe {
                std::ptr::copy_nonoverlapping(out.as_ptr(), ids, n);
                *out_len = n as i32;
            }
        }
        GigaTokenizer::SentencePiece { tok: mtx, state } => {
            let text_str = match std::str::from_utf8(text_slice) {
                Ok(s) => s,
                Err(_) => return GT_ERR_PARSE,
            };
            let mut sp = mtx.lock().unwrap();
            let mut st = state.lock().unwrap();
            let mut out: Vec<TokenId> = Vec::new();
            sp.encode_raw_with(&mut st, text_str, &mut out);
            let n = out.len().min(cap);
            unsafe {
                // TokenId is repr(transparent) over u32
                std::ptr::copy_nonoverlapping(out.as_ptr() as *const u32, ids, n);
                *out_len = n as i32;
            }
        }
    }
    GT_OK
}

// ─── Decode ──────────────────────────────────────────────────────────────

/// Decode token IDs back to UTF-8 text (null-terminated).
///
/// Parameters:
///   tok       — tokenizer handle
///   ids       — input token IDs
///   id_count  — number of IDs
///   out       — output buffer for UTF-8 text
///   out_len   — in: capacity (bytes), out: bytes written (excluding null)
///   capacity  — max bytes in out buffer
///
/// Returns: GT_OK on success, negative on error.
#[unsafe(no_mangle)]
pub extern "C" fn gt_decode(
    tok: *mut GigaTokenizer,
    ids: *const u32,
    id_count: i32,
    out: *mut c_char,
    out_len: *mut i32,
    capacity: i32,
) -> i32 {
    if tok.is_null() || ids.is_null() || out.is_null() || out_len.is_null() {
        return GT_ERR_LOAD;
    }
    let tok = unsafe { &*tok };
    // TokenId is repr(transparent) over u32
    let id_slice: &[TokenId] =
        unsafe { std::slice::from_raw_parts(ids as *const TokenId, id_count as usize) };
    let cap = capacity as usize;

    let bytes: Vec<u8> = match tok {
        GigaTokenizer::BPE(mtx) => {
            let bpe = mtx.lock().unwrap();
            bpe.decode(id_slice).collect()
        }
        GigaTokenizer::SentencePiece { tok: mtx, .. } => {
            let sp = mtx.lock().unwrap();
            sp.decode(id_slice)
        }
    };

    let n = bytes.len().min(cap.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out as *mut u8, n);
        *out.offset(n as isize) = 0; // null-terminate
        *out_len = n as i32;
    }
    GT_OK
}

// ─── Introspection ───────────────────────────────────────────────────────

/// Return the vocabulary size (number of tokens).
#[unsafe(no_mangle)]
pub extern "C" fn gt_vocab_size(tok: *const GigaTokenizer) -> i32 {
    if tok.is_null() {
        return 0;
    }
    let tok = unsafe { &*tok };
    match tok {
        GigaTokenizer::BPE(mtx) => mtx.lock().unwrap().vocab_size() as i32,
        GigaTokenizer::SentencePiece { tok: mtx, .. } => mtx.lock().unwrap().vocab_size() as i32,
    }
}

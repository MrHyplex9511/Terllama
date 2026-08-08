//! Python-free C API build of the GigaToken tokenizer.
//!
//! Reuses the vendored gigatoken sources (third_party/gigatoken) via
//! `#[path]` includes so we get the pure-Rust BPE/SentencePiece tokenizer
//! without pulling in pyo3 / numpy / libpython. The exported symbols are
//! the C ABI in `c_api` (gt_init, gt_cleanup, gt_tokenizer_load_hf,
//! gt_tokenizer_free, gt_encode, gt_decode, gt_vocab_size) — the exact
//! set the engine's GigaTokenWrapper dlsym's.
//!
//! NOTE: nested `mod` declarations inside the included files resolve
//! relative to the included file's own directory, so the original crate's
//! module layout is preserved.

// GigaToken sources need the portable_simd feature (used by
// bpe/sentencepiece.rs), which is nightly-only — same as the vendored crate.
#![feature(portable_simd)]
// The vendored crate also allows these lints.
#![allow(unused)]

// The C API module lives in this crate (committed copy, since the vendored
// repo's copy is untracked and not in the pinned upstream commit).
pub mod c_api;

// ─── Included vendored modules (pure Rust, no pyo3) ──────────────────────

#[path = "../../gigatoken/src/token.rs"]
pub(crate) mod token;

#[path = "../../gigatoken/src/bpe/mod.rs"]
pub(crate) mod bpe;

#[path = "../../gigatoken/src/pretokenize/mod.rs"]
pub(crate) mod pretokenize;

#[path = "../../gigatoken/src/input/mod.rs"]
pub(crate) mod input;

#[path = "../../gigatoken/src/load_tokenizer/mod.rs"]
pub(crate) mod load_tokenizer;

// c_api re-uses the same types the vendored crate exposes at its crate root.
pub use crate::bpe::Tokenizer;
pub use crate::bpe::sentencepiece::EncodeState;
pub use crate::load_tokenizer::hf;

/*
 * export.h — Native ALS model converter (Track C).
 *
 * End-to-end port of scripts/export_ternary_model_bitnet.py:
 *   download config.json + tokenizer files + safetensors from HuggingFace
 *   (libcurl), parse with the native safetensors parser, enumerate Linear
 *   layers in PyTorch named_modules order, ALS-decompose the quantized
 *   projections (Track B), and write model_decomposed.bin + model_extra.bin
 *   byte-compatible with the Python reference.
 *
 * No Python, no subprocesses. Single-shot CLI usage only.
 */
#pragma once

#include <string>

namespace terllama {

// Convert an HF repo to the native ALS format. Downloads everything into
// outdir (created if missing), then writes model_decomposed.bin and
// model_extra.bin plus the tokenizer files.
//   repo_id:   HF repo ("owner/name"), e.g. "HuggingFaceTB/SmolLM2-135M"
//   outdir:    destination directory (created recursively)
//   num_terms: ALS terms per quantized layer
//   format:    "als" (only real format; "gguf" accepted and routed to ALS,
//              matching the Python script's informational --format flag)
// Prints [PROGRESS] NN% lines to stdout. Returns 0 on success, 1 on failure.
int convert_model(const std::string& repo_id, const std::string& outdir,
                  int num_terms, const std::string& format);

// CLI subcommand "convert":
//   terllama convert --model <hf_repo> [--outdir <dir>] [--terms N] [--format als|gguf] [--rotate N]
int export_main(int argc, char** argv);

} // namespace terllama

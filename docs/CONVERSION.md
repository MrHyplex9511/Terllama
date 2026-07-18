# Model Conversion Guide

Convert HuggingFace models to Terllama's ternary inference format (I2_S or ALS).

## Prerequisites

- **Python 3.8+** — tokenizer helpers and export script
- **pip packages:**

```bash
pip install torch transformers
```

- **Hardware:** Any CPU or CUDA system (export runs on GPU if available; CPU is fine for small models)
- **Disk space:** ~2× the HF model size for export artifacts

## Quick Start

```bash
# 1. Install deps
pip install torch transformers

# 2. Export a model (I2_S format — default)
python scripts/export_ternary_model_bitnet.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --format i2s

# 3. Verify output
ls ~/.terllama/models/HuggingFaceTB-SmolLM2-135M/
#   model_extra.bin          (config + embeddings + norms)
#   model_decomposed_i2s.bin (packed ternary weights)
```

## Format Options

### `--format i2s` (default)

BitNet mean-scale ternary quantization.

- **Method:** Block-wise mean quantization (block_size=128). Each block is scaled by `1 / mean(|W|)` and rounded to `{-1, 0, +1}`.
- **Packing:** 4 ternary values per byte, 2-bit codes. Per-block `float32` scale appended.
- **Output file:** `model_decomposed_i2s.bin`
- **Best for:** Fast conversion, balanced accuracy/size.

```bash
python scripts/export_ternary_model_bitnet.py --model <name> --format i2s
```

### `--format als`

Alternating Least Squares multi-term ternary decomposition.

- **Method:** Decomposes each weight matrix into a sum of rank-1 ternary terms via ALS. Each term: `alpha × outer(u, v)` where `u, v ∈ {-1, 0, +1}`.
- **Packing:** Per-term alpha (`float32`) + 2-bit ternary codes (4 values/byte).
- **Output file:** `model_decomposed.bin`
- **Best for:** Higher accuracy (more terms = lower error). Use `--terms N` to control quality/size tradeoff.

```bash
# 12 terms (default): ~1.02× PPL vs FP32
python scripts/export_ternary_model_bitnet.py --model <name> --format als --terms 12
```

## Architecture Support

Terllama's export script auto-detects model architecture. Tested on:

| Architecture | Examples | Status |
|---|---|---|
| **Mistral** | `mistralai/Mistral-7B-v0.1` | ✅ |
| **Qwen** | `Qwen/Qwen-7B` | ✅ |
| **Llama** | `meta-llama/Llama-2-7b-hf`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ✅ |
| **SmolLM** | `HuggingFaceTB/SmolLM2-135M` | ✅ Primary target |
| **Gemma** | `google/gemma-2b` | ⚠️ Experimental |

Architectures must use:
- `nn.Linear` projections for Q, K, V, O, gate, up, down
- `nn.Embedding` for input tokens
- RMSNorm (pre-layer norms)
- RoPE positional embeddings

The script quantizes **7 projections per transformer block**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. All other layers stay in FP32.

## GGUF Model Support

Terllama can directly load GGUF format models (Q2_0 g128) via its built-in GGUF parser:

- **GGUF v3** supported (header + metadata + Q2_0 quantized tensors)
- Loads `.gguf` files directly: `terllama serve --model path/to/model.gguf`
- Auto-detects `.gguf` files in model directories
- Converts Q2_0 ternary weights to I2_S format in-memory
- Supported architectures: Mistral, Qwen, Llama, SmolLM, Gemma

GGUF models do not require the Python export script — just download and run.

For GGUF models without a companion `model_extra.bin`, the tokenizer vocab is
extracted from the GGUF metadata automatically (SentencePiece or BPE).

## Output Files

| File | Contents | Always present? |
|---|---|---|
| `model_extra.bin` | 9 config fields (int32/float32), embedding table `[vocab_size × hidden_size]`, final norm, per-layer input + post-attention norms | ✅ Yes |
| `model_decomposed_i2s.bin` | I2_S packed ternary weights + per-block scales | With `--format i2s` |
| `model_decomposed.bin` | ALS multi-term ternary weights | With `--format als` |

### `model_extra.bin` Layout

```
[9 config fields]         int32: vocab_size, hidden_size, intermediate_size,
                                  num_hidden_layers, num_attention_heads,
                                  num_key_value_heads, max_position_embeddings
                          float32: rms_norm_eps, rope_theta
[embedding table]         float32 [vocab_size × hidden_size]
[final norm]              float32 [hidden_size]
[layer norms]             float32 [num_hidden_layers × 2 × hidden_size]
                          (input_layernorm + post_attention_layernorm per layer)
```

## Custom Model Export

```bash
# Specify output directory
python scripts/export_ternary_model_bitnet.py \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --format als \
    --terms 12 \
    --outdir /tmp/my-model

# Rotate RoPE theta (advanced)
python scripts/export_ternary_model_bitnet.py \
    --model HuggingFaceTB/SmolLM2-135M \
    --rotate -1  # set to max_positional
```

## Troubleshooting

### "Out of memory" during export

Large models (7B+) may OOM on CPU. Solutions:
- Run on a GPU machine (`torch` auto-detects CUDA)
- Reduce batch — export script processes one layer at a time, but embedding requires full vocab
- Use `--format i2s` (lower memory than ALS)

### "Tokenization failed" or "Decoding failed" at inference

The server calls `tokenize_helper.py` and `decode_helper.py` via `system()`. Ensure:
- `scripts/` is co-located with the terllama binary, **or**
- The working directory contains `scripts/`

### "Model not loaded" (503 from server)

- Verify model files exist in the model directory
- Check `TERLLAMA_MODEL_DIR` env var
- Run `./terllama list` to verify the model is registered
- Run the server with `TERLLAMA_ARCH=scalar` to test without SIMD

### Export script can't find transformers module

```bash
pip install --upgrade transformers torch
```

### "Unrecognized architecture" warning

The export script inspects `model.named_modules()`. If your architecture uses non-standard projection names, it skips quantization for those layers. The model still runs — those layers use raw FP32.

## Post-Export Validation

After converting a model, verify it works correctly:

1. **Quick sanity check:** `terllama bench` — runs a fixed prompt and reports
   tokens/second and memory usage.
2. **Chat test:** `terllama run` — send a few messages and check output quality.
3. **PPL validation (advanced):** Compare perplexity against the FP32 baseline
   to ensure the ternary decomposition quality is within expected range
   (typically <5% degradation).

Expected benchmark range for SmolLM2-135M: ~50-80 tok/sec on modern CPU.

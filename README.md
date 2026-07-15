# Terllama

**Alpha** — under active development. Kernels work, output is plausible, but expect rough edges.

Ternary LLM inference engine. CPU-first, multi-ISA (scalar, AVX2, NEON).  
Runs SmolLM2-135M (and similar ternary-quantized models) using I2_S packed weights, INT8 activations, and tile-parallel tiling.

## Build

```bash
make          # terllama + terllama-bench
make terllama # inference binary only
make bench    # benchmark only
```

Detects available ISA extensions (AVX2+FMA, NEON) and compiles matching kernels. Missing ISAs are skipped at build time. On x86_64 without AVX2 the scalar fallback is used.

## Usage

```bash
# Export a HuggingFace model to Terllama format
python3 scripts/export_ternary_model_bitnet.py --hf <model>

# Run inference
./terllama model_decomposed_i2s.bin

# Benchmark all kernels
./terllama-bench model_decomposed_i2s.bin
```

Discord: https://discord.com/invite/TBB6KNkP7M

## Results (SmolLM2-135M)

### Model Size

| Format | Size | Notes |
|--------|------|-------|
| FP32 original | ~540 MB | 135M params × 4 bytes |
| I2_S .gguf (BitNet) | 1.2 GB | BitNet-b1.58-2B-4T reference |
| Decomposed I2S binary | **139 MB** | Terllama format, ~4× smaller than FP32 |

### PPL on WikiText-2

| Method | PPL | Ratio vs FP32 |
|--------|-----|---------------|
| FP32 baseline | 15.89 | 1.0× |
| Terllama (8-term FFN / 10-term QKV / 12-term O / 15-term LM) | 16.84 | 1.06× |
| Terllama (10+ terms all layers) | 16.23 | 1.02× |

### Per-Layer Decomposition Accuracy (8 ALS terms)

| Layer | Shape | Rel Error | Cos Sim | Best Method |
|-------|-------|-----------|---------|-------------|
| Attention Q | 576×576 | 7.93% | 0.997 | ALS |
| Attention K | 576×576 | 7.03% | 0.998 | ALS |
| Attention V | 576×576 | 5.08% | 0.999 | ALS |
| Attention O | 576×576 | 11.36% | 0.994 | ALS |
| FFN Gate | 1536×576 | 4.92% | 0.999 | ALS |
| FFN Up | 1536×576 | 3.78% | 0.999 | ALS |
| FFN Down | 576×1536 | 7.41% | 0.997 | ALS |
| LM Head | 49152×576 | 13.43% | 0.996 | ALS |

Accuracy improves with more terms: at 10 terms the FFN layers drop below 2% error; at 12 terms the attention projections reach <5%.

### Compared: TinyLlama-1.1B

| Method | PPL | Ratio |
|--------|-----|-------|
| FP32 baseline | 8.24 | 1.0× |
| Terllama (12 terms all layers) | 8.26 | **1.003×** |

Larger models tolerate decomposition better — 1.1B is virtually lossless at 12 terms.

## Architecture

| Layer | File | Role |
|-------|------|------|
| Dispatcher | `src/dispatcher.cpp` | Runtime CPU detection, selects optimal kernel |
| Kernels | `src/kernel_avx2.cpp`, `src/kernel_neon.cpp` | ISA-specific ternary matmul |
| Model | `src/model.h` | Binary format, layer layout |
| Loader | `src/loader.h` | I2_S + ALS format loader |
| Inference | `src/inference.h` | Autoregressive generation loop |
| Main | `src/main.cpp` | CLI entry point |
| Benchmark | `src/benchmark.cpp` | Per-kernel correctness + speed |

## Optimizations

- **I2_S packing** — 4 ternary values per byte, 2-bit codes
- **INT8 activations** — quantize FP32 to INT8 before matmul
- **Mean scaling** — block-wise mean-based ternary quantization
- **Selective layer quant** — 7 projection layers per transformer block
- **Tile-parallel tiling** — 128-col tiles, weights unpacked once per tile

## Files

```
src/           C++ inference engine
scripts/       Model export + packing helpers
legacy/        Standalone pre-terllama prototypes
```

## License

MIT

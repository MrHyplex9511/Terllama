# Terllama

Ternary LLM inference engine. CPU-first, multi-ISA.

Runs SmolLM2-135M (and similar ternary-quantized models) using I2_S packed weights, INT8 activations, and tile-parallel tiling.

## Build

```bash
make          # terllama + terllama-bench
make terllama # inference binary only
make bench    # benchmark only
```

Detects available ISA extensions (SSE4.2, AVX, AVX2+FMA, AVX-512, NEON) and compiles matching kernels. Missing ISAs are skipped at build time.

## Usage

```bash
# Export a HuggingFace model to Terllama format
python3 scripts/export_ternary_model_bitnet.py --hf <model>

# Run inference
./terllama model_decomposed_i2s.bin

# Benchmark all kernels
./terllama-bench model_decomposed_i2s.bin
```

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

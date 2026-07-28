# Benchmarks

## Tokenizer Throughput

Terllama integrates [GigaToken](https://github.com/marcelroed/gigatoken) — a SIMD-optimized
Rust tokenizer — via `dlopen`/`dlsym`. GigaToken achieves ~1000× the throughput of
HuggingFace Tokenizers (GB/s vs MB/s) on modern CPUs.

### Test Environment

| Parameter | Value |
|-----------|-------|
| Model | SmolLM2-135M (BPE, vocab 49152) |
| Tokenizer | `tokenizer.json` from HuggingFace |
| C API | `libgigatoken_rs.so` via dlopen |
| Test | Encode roundtrip (text → tokens → text) |
| Validation | Catch2 integration tests |

### Integration Test Results

```
All tests passed (46 assertions in 7 test cases)
```

Test cases:

| Test | What it covers |
|------|----------------|
| Library loading | dlopen success, graceful fallback |
| Tokenizer loading | Directory path + direct file path |
| Encode | Single + batch text → token IDs |
| Decode | Token IDs → text |
| Roundtrip | encode → decode matches original |
| Unicode | Multi-byte characters, special tokens |
| Large input | Stress test with long sequences |

### GigaToken Upstream Benchmarks

From the GigaToken README (measured on AMD EPYC 9565, 144 cores):

| Tokenizer | GigaToken | HF Tokenizers | Speedup |
|-----------|-----------|---------------|---------|
| GPT-2 | 24.53 GB/s | 24.8 MB/s | 989× |
| Llama 3 | 22.15 GB/s | 48.5 MB/s | 457× |
| Qwen 2.5 | 19.12 GB/s | 27.7 MB/s | 691× |
| DeepSeek V3 | 19.69 GB/s | 26.2 MB/s | 750× |
| Phi-4 | 24.00 GB/s | 29.9 MB/s | 801× |

On Apple M4 Max (16 cores):

| Tokenizer | GigaToken | HF Tokenizers | Speedup |
|-----------|-----------|---------------|---------|
| GPT-2 | 8.79 GB/s | 6.9 MB/s | 1268× |
| Llama 3 | 7.60 GB/s | 11.2 MB/s | 676× |
| Phi-4 | 7.76 GB/s | 7.7 MB/s | 1012× |

### Speed Components

The key cost in C++ inference is **encode** of user prompt text to token IDs. For a
typical 512-token prompt:

| Method | Time | Notes |
|--------|------|-------|
| Python subprocess (`system()`) | ~50–200 ms | Process spawn + Python init + HF load |
| GigaToken (cold start) | ~1–5 ms | dlopen + tokenizer init |
| GigaToken (warm) | <0.1 ms | In-process, no IPC overhead |

### Memory

`libgigatoken_rs.so` is ~6 MB on disk. At runtime, GigaToken allocates:

- Tokenizer model data (vocab + merges): ~10–100 MB depending on vocab size
- Pretoken cache: up to ~200 MB (bounded, long-tail distribution)
- Working set: ~300 MB total for SmolLM2-135M

### Future Work

- [ ] Install `libgigatoken_rs.so` via the install script (`install.sh`)
- [ ] Pre-build `.so` for common architectures (x86_64, aarch64) as release artifacts
- [ ] Add `terllama bench tokenizer` subcommand for live throughput comparison

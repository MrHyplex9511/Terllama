# Roadmap

Ternary LLM inference engine. CPU-first, multi-ISA.

## 1. Core Engine — Quality & Performance

### 1.1 Fix ALS export (top blocker)
**Problem:** `--format als` export produces garbled output. ALS term reordering must match loader expectation. The `pack_weights.py` module exists but the export script packs ALS terms per-layer then writes them, while loader reads term-by-term. Investigate ordering mismatch.

**Check:** `scripts/pack_weights.py` for correct bitplane layout vs `loader.h::decode_ternary`. The bitplane format uses 2-bit codes (00=0, 10=+1, 11=-1) packed MSB-first per byte. Verify ALS output matches.

**Goal:** `./terllama pull HuggingFaceTB/SmolLM2-135M --format als` produces valid output for any term count 8-15.

### 1.2 Validate I2_S path end-to-end
The I2_S pipeline works in principle (README shows good PPL numbers). Reproduce on a fresh checkout:
```bash
make clean && make
./terllama pull HuggingFaceTB/SmolLM2-135M --format i2s
echo "Hello" | ./terllama --temp 0
```
Document exact steps and expected output.

### 1.3 T-SAR integration (DATE 2026, arXiv 2511.13676)
[T-SAR](https://arxiv.org/abs/2511.13676) achieves **5.6–24.5× GEMM speedup** over current ternary SOTA by generating LUTs in-register on AVX2, eliminating memory bottlenecks.

**Priority:** After ALS fix + I2_S validation. This is the single biggest performance lever.

**Plan:**
1. Study T-SAR LUT generation for AVX2 (vpgatherdq + vperm2i128)
2. Implement `kernel_avx2_tsar.cpp` as new kernel variant
3. Benchmark against current I2_S tiled matmul
4. Fall back gracefully on non-AVX2 (NEON variant later)
5. Report: tok/s speedup at batch=1, batch=4, batch=16 on SmolLM2-135M

**Dependencies:** AVX2 host for development.

### 1.4 Ternary Bonsai model support
[Ternary Bonsai](https://huggingface.co/prism-ml) (Prism ML, 2026) — 3 models at 1.7B, 4B, 8B params, pre-trained ternary in GGUF Q2_0 format (g128 block size). This is the largest open ternary model family available.

**Priority:** After T-SAR. Enables running real-world-sized ternary models.

**Plan:**
1. Add GGUF Q2_0 loader (or converter from Q2_0 to I2_S format)
2. Map Ternary Bonsai layer names to Terllama's expected schema
3. Support `./terllama pull prism-ml/Ternary-Bonsai-1.7B-gguf`
4. Report PPL on WikiText-2 vs FP16 baseline

### 1.5 TriLM / Spectra-1.1 compatibility
[Spectra-1.1](https://arxiv.org/abs/2506.23025) (ACL 2025) — ternary models trained up to 3.6B params on 1.2T tokens. Uses 2-bit and 1.6-bit packing schemes with TriRun GPU kernel (up to 5× speedup).

**Relevance:** Spectra models are the best-studied ternary LMs with scaling laws. If Terllama can run them, we get immediate comparison against published baselines.

**Plan:**
1. Convert Spectra TriLM packed format → I2_S or native format
2. Validate PPL matches published numbers
3. Benchmark on CPU to establish baseline for T-SAR comparison

### 1.6 Mixed-precision architecture (per TernaryLM findings)
[TernaryLM](https://arxiv.org/abs/2602.07374) shows non-uniform sparsity across transformer depth: middle layers (L5–L9) achieve 60–62% sparsity vs 45–55% for boundary layers. Design principle: apply ternary precision to high-sparsity middle layers, higher precision to boundary layers.

**Plan:**
1. Profile per-layer quantization error in Terllama's current ALS output
2. Implement mixed-precision dispatch: boundary layers at higher term count (12–15), middle layers at lower count (6–8)
3. Measure PPL vs tokens/s tradeoff curve

## 2. Model Ecosystem

### 2.1 HuggingFace integration
**Current:** `./terllama pull <repo> --format <i2s|als>` calls a Python script that downloads + converts. Works for HF models.

**Gaps:**
- No `--format auto` detection (must specify format explicitly)
- No model card display
- No support for GGUF-native ternary models (Ternary Bonsai)
- Conversion status not cached

**Plan:**
1. Add format auto-detection from repo contents (check for .gguf, existing decomposed files)
2. Cache conversion artifacts so re-pull is no-op
3. List available formats for installed models in `show` command
4. Report download progress in MB/s

### 2.2 Native ternary model fine-tuning
**Long-term:** Export trained ternary models from Terllama format back to HF. Allow:
```bash
./terllama export SmolLM2-135M --format hf --out ./my-model
```

This would enable the full loop: pull FP32 → decompose → validate → fine-tune → republish.

### 2.3 Model zoo / registry
A curated list of tested models with known-good configs (term counts, PPL, tokens/s). Stored as `models.json` or YAML in repo.

**Format:**
```yaml
smollm2-135m:
  hf: HuggingFaceTB/SmolLM2-135M
  formats: [i2s, als]
  terms: 12
  ppl_wikitext2: 16.23
  tok_s_avx2: 45
  params: 135M
```

## 3. Infrastructure

### 3.1 Test suite
**Current:** Zero tests. `terllama-bench` validates kernel correctness (compares against scalar baseline) but no regression suite.

**Plan:**
1. Reference outputs: generate known-good logits/tokens for SmolLM2-135M prompt "Hello"
2. `make test`: runs inference on fixed prompts, compares against reference
3. CI runs per commit (see 3.2)

### 3.2 CI pipeline
**Current:** No `.github/` directory.

**Plan:**
1. GitHub Actions: `ubuntu-latest`, `gcc-12`/`clang-16`, `cmake`, `make`
2. Build matrix: `{scalar, avx2}` × `{gcc, clang}`
3. Test step: `make test` against reference outputs
4. Benchmark step (manual trigger): `make bench` → PR comment with result table

### 3.3 Benchmark harness
**Current:** `terllama-bench` runs per-kernel benchmarks. No end-to-end throughput tracking.

**Plan:**
1. `terllama-bench --end-to-end --model SmolLM2-135M --prompt-len 32 --gen-len 128`
2. Output: prompt processing tok/s, generation tok/s, peak memory
3. Track across commits to catch regressions

## Milestones

| # | What | Depends On | Est. Effort |
|---|------|-----------|-------------|
| M1 | ALS export fixed + validated | — | 2–3 days |
| M2 | Tests + CI running | M1 | 1–2 days |
| M3 | I2_S path validated end-to-end | M1 | 0.5 day |
| M4 | T-SAR AVX2 kernel | M2–M3 | 1–2 weeks |
| M5 | Ternary Bonsai support | M3 | 3–5 days |
| M6 | Spectra-1.1 support | M3 | 3–5 days |
| M7 | Mixed-precision dispatch | M4 | 1 week |
| M8 | HF integration + model registry | M5–M6 | 3–5 days |
| M9 | Native model export | M8 | 1 week |

## References

- **T-SAR:** Ma et al. "T-SAR: Accelerating Ternary Processing via In-Register Lookup Table Computation." DATE 2026. arXiv:2511.13676
- **Ternary Bonsai:** prism-ml. Pre-trained ternary models in GGUF Q2_0. 2026. https://huggingface.co/prism-ml
- **Spectra-1.1 / TriLM:** Vaidhya et al. "Scaling Laws and Efficient Inference for Ternary Language Models." ACL 2025. arXiv:2506.23025
- **TernaryLM:** Nisharg et al. "TernaryLM: Memory-Efficient Language Modeling via Native 1.5-Bit Quantization with Adaptive Layer-wise Scaling." arXiv:2602.07374
- **BitNet b1.58:** Wang et al. "The Era of 1-bit LLMs." arXiv:2402.17764

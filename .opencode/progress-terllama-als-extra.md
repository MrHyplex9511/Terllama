# Progress: Terllama ALS + model_extra.bin pipeline

## Completed
- Added `write_extra()` to export script — writes model_extra.bin with config, embedding, RMS norms
- Integrated `write_extra()` call into `main()` after ALS/I2_S export
- Verified end-to-end: `terllama pull HuggingFaceTB/SmolLM2-135M --format als` produces both `model_extra.bin` (109 MB) + `model_decomposed.bin` (311 MB)
- Fixed `helper_dir` path resolution in 3 locations (main.cpp legacy, main.cpp chat, server.cpp) — was incorrectly resolving to `<model_dir>/scripts/` when TERLLAMA_MODEL_DIR was set
- C++ loader successfully loads both files: `terllama show` shows all model config fields
- Legacy and chat inference modes work end-to-end (~1.7 tok/s for SmolLM2-135M ALS)

## Key Decisions
- `write_extra()` reloads HF model from cache (not from decomposed data) for simplicity
- `helper_dir` resolved via binary path (`/proc/self/exe` + dirname) in server.cpp for robustness
- EOS token ID = 0 (standard for <|endoftext|>)

## Known Issues
- ALS reconstruction error is very high (~85-99% per layer), producing garbled output ("ectable" repetition)
- `terllama list` shows "No models installed" — models.json not updated by direct Python script run
- Default max_tokens=256 in chat mode takes ~150s at current inference speed

## Workers
- Level 1: Orchestration, planning, fix identification
- No delegation needed

## Notes
- ALS quality is the main blocker for useful output. Consider more ALS terms (16+), better SVD initialization, or per-row scaling factors.

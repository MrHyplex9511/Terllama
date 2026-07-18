# Progress: Terllama Desktop App v1.0.0

## Completed
- Full Tauri v2 + Svelte 5 + Tailwind CSS desktop app (49 files, 24K lines)
- Rust backend: process mgmt, streaming download, chat proxy, config, system tray
- Svelte frontend: library grid, chat with streaming, settings, onboarding
- Registry updated to new schema with quants (ternary/q4_k_m/q8_0) for 7 models
- CI: desktop-release.yml (5-platform build)
- Binary builds at 247MB debug

## Decisions
- Tauri v2 chosen over Electron: ~10MB vs ~100MB, native WebView, Rust→C++ FFI
- Svelte 5 over React: lighter runtime, simpler state with runes
- Sidecar pattern: Rust spawns terllama binary, communicates via HTTP
- GGUF mode note: binary may not support standard GGUF — design allows fallback

## Workers
- Level 3: Full desktop app implementation (Tauri backend + Svelte frontend + registry + CI)

## Notes
- Tag v1.0.0 moved to latest commit (88854ea) to include desktop app
- Registry now versioned with quants structure — old format replaced
- Desktop build requires system deps: libwebkit2gtk-4.1-dev, librsvg2-dev on Linux

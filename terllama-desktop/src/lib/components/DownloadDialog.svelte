<script lang="ts">
  import type { RegistryModel, DownloadProgress, DownloadFormat } from '../../types';
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';

  let {
    model,
    show = false,
    onClose,
    onLoad,
  }: {
    model: RegistryModel | null;
    show?: boolean;
    onClose?: () => void;
    onLoad?: (modelId: string) => void;
  } = $props();

  let selectedFormat = $state<DownloadFormat>('ternary');
  let showConvertConfirm = $state(false);
  let isDownloading = $state(false);
  let progress = $state<DownloadProgress | null>(null);
  let completed = $state(false);
  let errorMessage = $state<string | null>(null);
  let cleanup: (() => void) | null = null;

  let dialogEl: HTMLDivElement | undefined = $state();

  $effect(() => {
    if (show) {
      const unlisten = listen<DownloadProgress>('download-progress', (event) => {
        progress = event.payload;
        if (event.payload.downloaded >= event.payload.total) {
          completed = true;
          isDownloading = false;
        }
      });
      cleanup = () => { unlisten.then(fn => fn()); };
      return () => {
        cleanup?.();
        cleanup = null;
      };
    }
  });

  function formatSize(mb: number): string {
    if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
    return mb + ' MB';
  }

  // Rough RAM / time estimate for FP32 → ternary conversion.
  // Based on the FP source size: conversion needs ~2.5x the model in RAM
  // (torch + weights + workspace) and runs at ~1 GB / 5 min on a typical CPU.
  function convertWarnings() {
    if (!model) return { ram: '', time: '' };
    const fpMb = model.formats?.fp?.size_mb || model.size_mb;
    const ramGb = Math.ceil((fpMb * 2.5) / 1024);
    const timeMin = Math.max(2, Math.ceil((fpMb / 1024) * 5));
    return {
      ram: `${ramGb}+ GB free RAM`,
      time: timeMin >= 60 ? `~${(timeMin / 60).toFixed(1)} hours` : `~${timeMin} minutes`,
    };
  }

  async function handleDownload() {
    if (!model) return;
    if (selectedFormat === 'ternary' && model.formats.ternary.needs_conversion && !showConvertConfirm) {
      showConvertConfirm = true;
      return;
    }
    showConvertConfirm = false;
    isDownloading = true;
    completed = false;
    progress = null;
    errorMessage = null;
    try {
      await invoke('download_model', { modelId: model.id, format: selectedFormat });
    } catch (e) {
      errorMessage = typeof e === 'string' ? e : e?.message || 'Download failed — check console for details';
      console.error('Download failed:', e);
      isDownloading = false;
    }
  }

  function handleClose() {
    show = false;
    completed = false;
    progress = null;
    isDownloading = false;
    errorMessage = null;
    showConvertConfirm = false;
    onClose?.();
  }

  function handleLoad() {
    if (!model) return;
    onLoad?.(model.id);
    handleClose();
  }

  function progressPercent(): number {
    if (!progress || progress.total === 0) return 0;
    return Math.min(100, Math.round((progress.downloaded / progress.total) * 100));
  }
</script>

{#if show && model}
  <!-- Overlay -->
  <div class="overlay" onclick={handleClose} role="presentation">
    <!-- Dialog -->
    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div class="dialog" onclick={(e) => e.stopPropagation()} onkeydown={(e) => e.stopPropagation()} role="dialog" tabindex="0" bind:this={dialogEl}>
      <div class="header">
        <h2>{model.name || model.id}</h2>
        <button class="close-btn" onclick={handleClose} aria-label="Close dialog">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <p class="desc">{model.description}</p>

      <!-- Format selector -->
      <div class="quant-section">
        <span class="section-label">Select Format</span>
        <div class="quant-grid">
          <button
            class="quant-card"
            class:selected={selectedFormat === 'fp'}
            disabled={!model.formats.fp.available}
            onclick={() => { selectedFormat = 'fp'; showConvertConfirm = false; }}
          >
            <div class="quant-name">Normal FP</div>
            <div class="quant-info">
              <span>Size: {formatSize(model.formats.fp.size_mb)}</span>
              <span>Original weights</span>
            </div>
            {#if !model.formats.fp.available}
              <div class="quant-badge coming-soon">Unavailable</div>
            {/if}
          </button>

          <button
            class="quant-card"
            class:selected={selectedFormat === 'q4'}
            disabled={!model.formats.q4.available}
            onclick={() => { selectedFormat = 'q4'; showConvertConfirm = false; }}
          >
            <div class="quant-name">Q4 (GGUF Q4_K_M)</div>
            <div class="quant-info">
              <span>Size: {formatSize(model.formats.q4.size_mb)}</span>
              <span>4-bit quantized</span>
            </div>
            {#if !model.formats.q4.available}
              <div class="quant-badge coming-soon">Unavailable</div>
            {/if}
          </button>

          <button
            class="quant-card"
            class:selected={selectedFormat === 'ternary'}
            disabled={!model.formats.ternary.available}
            onclick={() => { selectedFormat = 'ternary'; showConvertConfirm = false; }}
          >
            <div class="quant-name">Ternary (1.58-bit)</div>
            <div class="quant-info">
              <span>Size: {formatSize(model.formats.ternary.size_mb)}</span>
              <span>{model.formats.ternary.needs_conversion ? 'Convert from FP' : 'Pre-made weights'}</span>
            </div>
            {#if model.formats.ternary.needs_conversion}
              <div class="quant-badge ternary">Convert</div>
            {:else}
              <div class="quant-badge ternary">Recommended</div>
            {/if}
          </button>
        </div>

        {#if selectedFormat && model.formats[selectedFormat].note}
          <p class="format-note">{model.formats[selectedFormat].note}</p>
        {/if}
      </div>

      <!-- Convert warning banner (FP32 → ternary) -->
      {#if selectedFormat === 'ternary' && model.formats.ternary.needs_conversion && !isDownloading && !completed}
        <div class="convert-warning">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" /><line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
          <div>
            <strong>This model has no pre-made ternary weights.</strong>
            <span>Terllama will download the FP weights and convert them locally on your CPU.</span>
            <span class="warn-stats">Estimated: {convertWarnings().ram} RAM &middot; {convertWarnings().time} conversion time</span>
          </div>
        </div>
        {#if showConvertConfirm}
          <div class="convert-confirm">
            <span>This will take a while and uses significant RAM. Continue?</span>
            <div class="confirm-actions">
              <button class="btn primary confirm-yes" onclick={handleDownload}>Yes, download &amp; convert</button>
              <button class="btn secondary" onclick={() => (showConvertConfirm = false)}>Go back</button>
            </div>
          </div>
        {/if}
      {/if}

      <!-- Error message -->
      {#if errorMessage}
        <div class="error-section">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <span>{errorMessage}</span>
        </div>
      {/if}

      <!-- Progress -->
      {#if isDownloading}
        <div class="progress-section">
          <div class="progress-bar">
            <div class="progress-fill" style="width: {progressPercent()}%"></div>
          </div>
          <div class="progress-info">
            <span>{progressPercent()}%</span>
            {#if progress}
              <span>{formatSize(progress.total)} @ {(progress.speed / 1e6).toFixed(1)} MB/s</span>
            {/if}
          </div>
        </div>
      {/if}

      <!-- Actions -->
      <div class="actions">
        {#if completed}
          <button class="btn primary" onclick={handleLoad}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Load Model
          </button>
        {:else}
          <button class="btn primary" onclick={handleDownload} disabled={isDownloading}>
            {#if selectedFormat === 'ternary' && model.formats.ternary.needs_conversion && !showConvertConfirm}
              {isDownloading ? 'Converting...' : 'Download & Convert'}
            {:else}
              {isDownloading ? 'Downloading...' : 'Download'}
            {/if}
          </button>
        {/if}
        <button class="btn secondary" onclick={handleClose} disabled={isDownloading && !completed}>
          Cancel
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
    backdrop-filter: blur(4px);
    animation: overlayFadeIn 0.2s ease-out;
  }

  @keyframes overlayFadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
  }

  .dialog {
    background: hsla(var(--surface-secondary), 0.85);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid hsla(var(--border), 0.6);
    border-radius: var(--radius);
    padding: 28px;
    width: 520px;
    max-width: 90vw;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: var(--shadow);
    animation: dialogSlideUp 0.25s ease-out;
  }

  @keyframes dialogSlideUp {
    from { opacity: 0; transform: translateY(16px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 8px;
  }

  .header h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 700;
    color: hsl(var(--content));
  }

  .close-btn {
    background: none;
    border: none;
    color: hsl(var(--content-muted));
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
  }

  .close-btn:hover {
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content));
  }

  .desc {
    color: hsl(var(--content-muted));
    font-size: 13px;
    margin: 0 0 20px;
    line-height: 1.5;
  }

  .quant-section {
    margin-bottom: 20px;
  }

  .section-label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: hsl(var(--content-muted));
    margin-bottom: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .quant-grid {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .quant-card {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding: 14px;
    border: 1px solid hsla(var(--border), 0.5);
    border-radius: var(--radius-sm);
    background: hsla(var(--surface-tertiary), 0.4);
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
    position: relative;
  }

  .quant-card:hover:not(:disabled) {
    border-color: hsl(var(--brand));
    background: hsla(var(--brand), 0.05);
  }

  .quant-card.selected {
    border-color: hsl(var(--brand));
    background: hsla(var(--brand), 0.1);
    box-shadow: 0 0 12px hsla(var(--brand), 0.1);
  }

  .quant-card:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .quant-name {
    font-size: 14px;
    font-weight: 600;
    color: hsl(var(--content));
  }

  .quant-info {
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: hsl(var(--content-muted));
  }

  .quant-badge {
    position: absolute;
    top: 8px;
    right: 8px;
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
  }

  .quant-badge.ternary {
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: white;
  }

  .quant-badge.coming-soon {
    background: hsla(var(--warning), 0.2);
    color: hsl(var(--warning));
  }

  .format-note {
    margin: 10px 2px 0;
    font-size: 12px;
    color: hsl(var(--content-muted));
    line-height: 1.5;
  }

  .convert-warning {
    display: flex;
    gap: 10px;
    padding: 12px 14px;
    margin-bottom: 14px;
    background: hsla(38, 90%, 50%, 0.12);
    border: 1px solid hsla(38, 90%, 50%, 0.35);
    border-radius: var(--radius-sm);
    color: hsl(38, 90%, 75%);
    font-size: 12px;
    line-height: 1.5;
  }

  .convert-warning svg {
    flex-shrink: 0;
    margin-top: 1px;
  }

  .convert-warning strong {
    display: block;
    font-size: 13px;
    margin-bottom: 2px;
  }

  .convert-warning span {
    display: block;
  }

  .convert-warning .warn-stats {
    margin-top: 6px;
    font-weight: 600;
    color: hsl(38, 90%, 85%);
  }

  .convert-confirm {
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 12px 14px;
    margin-bottom: 14px;
    background: hsla(0, 70%, 50%, 0.1);
    border: 1px solid hsla(0, 70%, 50%, 0.3);
    border-radius: var(--radius-sm);
    color: hsl(0, 70%, 75%);
    font-size: 13px;
    line-height: 1.5;
  }

  .confirm-actions {
    display: flex;
    gap: 8px;
  }

  .confirm-actions .btn {
    flex: 0 1 auto;
    padding: 8px 14px;
    font-size: 13px;
  }

  .confirm-actions .confirm-yes {
    background: linear-gradient(135deg, hsl(0, 70%, 55%), hsl(0, 70%, 45%));
  }

  .error-section {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 12px 14px;
    margin-bottom: 16px;
    background: hsla(0, 70%, 50%, 0.1);
    border: 1px solid hsla(0, 70%, 50%, 0.3);
    border-radius: var(--radius-sm);
    color: hsl(0, 70%, 70%);
    font-size: 13px;
    line-height: 1.5;
    animation: dialogSlideUp 0.2s ease-out;
  }

  .error-section svg {
    flex-shrink: 0;
    margin-top: 1px;
  }

  .progress-section {
    margin-bottom: 20px;
  }

  .progress-bar {
    height: 6px;
    background: hsla(var(--surface-tertiary), 0.6);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .progress-fill {
    height: 100%;
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: hsl(var(--content-muted));
  }

  .actions {
    display: flex;
    gap: 10px;
  }

  .btn {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .btn.primary {
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: white;
  }

  .btn.primary:hover:not(:disabled) {
    opacity: 0.9;
    box-shadow: 0 0 16px hsla(var(--brand), 0.3);
  }

  .btn.secondary {
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content));
    border: 1px solid hsla(var(--border), 0.5);
  }

  .btn.secondary:hover:not(:disabled) {
    background: hsla(var(--border), 0.6);
  }
</style>

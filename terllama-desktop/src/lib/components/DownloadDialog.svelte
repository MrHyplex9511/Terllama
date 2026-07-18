<script lang="ts">
  import type { RegistryModel, DownloadProgress } from '../../types';
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

  let selectedQuant = $state('ternary');
  let isDownloading = $state(false);
  let progress = $state<DownloadProgress | null>(null);
  let completed = $state(false);
  let cleanup: (() => void) | null = null;

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

  function getSelectedInfo() {
    if (!model) return null;
    return model.quants[selectedQuant as keyof typeof model.quants];
  }

  async function handleDownload() {
    if (!model) return;
    isDownloading = true;
    completed = false;
    progress = null;
    try {
      await invoke('download_model', { modelId: model.id, quant: selectedQuant });
    } catch (e) {
      console.error('Download failed:', e);
      isDownloading = false;
    }
  }

  function handleClose() {
    show = false;
    completed = false;
    progress = null;
    isDownloading = false;
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
    <div class="dialog" onclick={(e) => e.stopPropagation()} role="dialog">
      <div class="header">
        <h2>{model.name || model.id}</h2>
        <button class="close-btn" onclick={handleClose}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <p class="desc">{model.description}</p>

      <!-- Quant selector -->
      <div class="quant-section">
        <label class="section-label">Select Quantization</label>
        <div class="quant-grid">
          <button
            class="quant-card"
            class:selected={selectedQuant === 'ternary'}
            onclick={() => (selectedQuant = 'ternary')}
          >
            <div class="quant-name">Ternary (1.58-bit)</div>
            <div class="quant-info">
              <span>Size: {formatSize(model.quants.ternary.size_mb)}</span>
              <span>Fast CPU inference</span>
            </div>
            <div class="quant-badge ternary">Recommended</div>
          </button>

          <button
            class="quant-card"
            class:selected={selectedQuant === 'q4_k_m'}
            disabled={!model.quants.q4_k_m.available}
            onclick={() => (selectedQuant = 'q4_k_m')}
          >
            <div class="quant-name">GGUF Q4_K_M</div>
            <div class="quant-info">
              <span>Size: {formatSize(model.quants.q4_k_m.size_mb)}</span>
              <span>Balanced quality</span>
            </div>
            {#if !model.quants.q4_k_m.available}
              <div class="quant-badge coming-soon">Coming Soon</div>
            {/if}
          </button>

          <button
            class="quant-card"
            class:selected={selectedQuant === 'q8_0'}
            disabled={!model.quants.q8_0.available}
            onclick={() => (selectedQuant = 'q8_0')}
          >
            <div class="quant-name">GGUF Q8_0</div>
            <div class="quant-info">
              <span>Size: {formatSize(model.quants.q8_0.size_mb)}</span>
              <span>High quality</span>
            </div>
            {#if !model.quants.q8_0.available}
              <div class="quant-badge coming-soon">Coming Soon</div>
            {/if}
          </button>
        </div>
      </div>

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
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            Load Model
          </button>
        {:else}
          <button class="btn primary" onclick={handleDownload} disabled={isDownloading}>
            {isDownloading ? 'Downloading...' : 'Download'}
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
  }

  .dialog {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    width: 520px;
    max-width: 90vw;
    max-height: 90vh;
    overflow-y: auto;
    box-shadow: var(--shadow);
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
  }

  .close-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
  }

  .close-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .desc {
    color: var(--text-secondary);
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
    color: var(--text-secondary);
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
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-tertiary);
    cursor: pointer;
    transition: all 0.15s;
    text-align: left;
    position: relative;
  }

  .quant-card:hover:not(:disabled) {
    border-color: var(--accent);
  }

  .quant-card.selected {
    border-color: var(--accent);
    background: rgba(124, 58, 237, 0.1);
  }

  .quant-card:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .quant-name {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .quant-info {
    display: flex;
    gap: 16px;
    font-size: 12px;
    color: var(--text-secondary);
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
    background: var(--accent);
    color: white;
  }

  .quant-badge.coming-soon {
    background: var(--warning);
    color: #000;
  }

  .progress-section {
    margin-bottom: 20px;
  }

  .progress-bar {
    height: 6px;
    background: var(--bg-tertiary);
    border-radius: 3px;
    overflow: hidden;
    margin-bottom: 8px;
  }

  .progress-fill {
    height: 100%;
    background: var(--accent-gradient);
    border-radius: 3px;
    transition: width 0.3s;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-secondary);
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
    background: var(--accent-gradient);
    color: white;
  }

  .btn.primary:hover:not(:disabled) {
    opacity: 0.9;
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
  }

  .btn.secondary:hover:not(:disabled) {
    background: var(--border);
  }
</style>

<script lang="ts">
  import type { RegistryModel, DownloadedModel } from '../../types';

  let {
    model,
    isDownloaded = false,
    onDownload,
    onLoad,
  }: {
    model: RegistryModel;
    isDownloaded?: boolean;
    onDownload?: (modelId: string, quant: string) => void;
    onLoad?: (modelId: string) => void;
  } = $props();

  function formatSize(mb: number): string {
    if (mb >= 1024) {
      return (mb / 1024).toFixed(1) + ' GB';
    }
    return mb + ' MB';
  }

  function handleDownload(quant: string) {
    onDownload?.(model.id, quant);
  }

  function handleLoad() {
    onLoad?.(model.id);
  }
</script>

<div class="card">
  <div class="card-header">
    <h3 class="model-name">{model.name || model.id}</h3>
    <span class="model-id">{model.id}</span>
  </div>

  <p class="description">{model.description}</p>

  <div class="meta">
    <span class="badge context-badge">{model.context} ctx</span>
    <span class="badge size-badge">{formatSize(model.size_mb)}</span>
    <span class="badge format-badge">{model.format}</span>
  </div>

  <div class="quants">
    <button
      class="quant-btn ternary"
      class:active={model.quants.ternary.available}
      disabled={!model.quants.ternary.available}
      onclick={() => handleDownload('ternary')}
    >
      <span class="quant-label">Ternary</span>
      <span class="quant-size">{formatSize(model.quants.ternary.size_mb)}</span>
    </button>

    <button
      class="quant-btn"
      class:active={model.quants.q4_k_m.available}
      disabled={!model.quants.q4_k_m.available}
      onclick={() => handleDownload('q4_k_m')}
    >
      <span class="quant-label">Q4_K_M</span>
      <span class="quant-size">{formatSize(model.quants.q4_k_m.size_mb)}</span>
    </button>

    <button
      class="quant-btn"
      class:active={model.quants.q8_0.available}
      disabled={!model.quants.q8_0.available}
      onclick={() => handleDownload('q8_0')}
    >
      <span class="quant-label">Q8_0</span>
      <span class="quant-size">{formatSize(model.quants.q8_0.size_mb)}</span>
    </button>
  </div>

  {#if isDownloaded}
    <button class="load-btn" onclick={handleLoad}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      Load Model
    </button>
  {:else}
    <button class="download-btn" onclick={() => handleDownload('ternary')}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
      Download
    </button>
  {/if}
</div>

<style>
  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px;
    transition: all 0.2s;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .card:hover {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px rgba(124, 58, 237, 0.2);
  }

  .card-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
  }

  .model-name {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .model-id {
    font-size: 11px;
    color: var(--text-secondary);
    font-family: monospace;
  }

  .description {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
    line-height: 1.5;
  }

  .meta {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .badge {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 500;
  }

  .context-badge {
    background: rgba(124, 58, 237, 0.15);
    color: var(--accent-hover);
  }

  .size-badge {
    background: rgba(34, 197, 94, 0.15);
    color: var(--success);
  }

  .format-badge {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  .quants {
    display: flex;
    gap: 8px;
  }

  .quant-btn {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 8px 4px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-tertiary);
    cursor: pointer;
    transition: all 0.15s;
  }

  .quant-btn:hover:not(:disabled) {
    border-color: var(--accent);
  }

  .quant-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .quant-btn.ternary:not(:disabled) {
    border-color: var(--accent);
    background: rgba(124, 58, 237, 0.1);
  }

  .quant-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .quant-size {
    font-size: 10px;
    color: var(--text-secondary);
  }

  .download-btn,
  .load-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    margin-top: auto;
  }

  .download-btn {
    background: var(--accent-gradient);
    color: white;
  }

  .download-btn:hover {
    opacity: 0.9;
  }

  .load-btn {
    background: rgba(34, 197, 94, 0.15);
    color: var(--success);
  }

  .load-btn:hover {
    background: rgba(34, 197, 94, 0.25);
  }
</style>

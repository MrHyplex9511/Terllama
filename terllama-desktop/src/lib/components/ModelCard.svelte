<script lang="ts">
  import type { RegistryModel } from '../../types';
  import ShinyText from './ui/ShinyText.svelte';

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
    if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
    return mb + ' MB';
  }

  function handleDownload() {
    onDownload?.(model.id, 'ternary');
  }

  function handleLoad() {
    onLoad?.(model.id);
  }
</script>

<div class="card-base model-card">
  <div class="card-header">
    <h3 class="model-name">
      <ShinyText text={model.name || model.id} baseColor="hsl(var(--content))" shimmerColor="hsla(var(--brand), 0.3)" speed={4} />
    </h3>
    <span class="model-id">{model.id}</span>
  </div>

  <p class="description">{model.description}</p>

  <div class="tags">
    <span class="tag context-tag">{model.context} ctx</span>
    <span class="tag size-tag">{formatSize(model.size_mb)}</span>
    <span class="tag format-tag">{model.format}</span>
  </div>

  {#if isDownloaded}
    <button class="action-btn load-btn" onclick={handleLoad}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12" />
      </svg>
      Load Model
    </button>
  {:else}
    <button class="action-btn download-btn" onclick={handleDownload}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
        <polyline points="7 10 12 15 17 10" />
        <line x1="12" y1="15" x2="12" y2="3" />
      </svg>
      Download
    </button>
  {/if}
</div>

<style>
  .model-card {
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    transition: all 0.2s;
  }

  .model-card:hover {
    border-color: hsl(var(--brand));
    box-shadow: var(--shadow-glow);
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
    color: hsl(var(--content));
  }

  .model-id {
    font-size: 11px;
    color: hsl(var(--content-muted));
    font-family: monospace;
  }

  .description {
    margin: 0;
    font-size: 13px;
    color: hsl(var(--content-muted));
    line-height: 1.5;
  }

  .tags {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }

  .tag {
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 4px;
    font-weight: 500;
  }

  .context-tag {
    background: hsla(var(--brand), 0.15);
    color: hsl(var(--brand-hover));
  }

  .size-tag {
    background: hsla(var(--success), 0.15);
    color: hsl(var(--success));
  }

  .format-tag {
    background: hsla(var(--surface-tertiary), 0.8);
    color: hsl(var(--content-muted));
  }

  .action-btn {
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
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: white;
  }

  .download-btn:hover {
    opacity: 0.9;
    box-shadow: 0 0 16px hsla(var(--brand), 0.3);
  }

  .load-btn {
    background: hsla(var(--success), 0.15);
    color: hsl(var(--success));
  }

  .load-btn:hover {
    background: hsla(var(--success), 0.25);
  }
</style>

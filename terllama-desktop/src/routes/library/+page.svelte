<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import type { RegistryModel, DownloadedModel } from '../../types';
  import DownloadDialog from '../../lib/components/DownloadDialog.svelte';
  import { getModelsState } from '../../lib/stores/models.svelte';
  import BlurText from '../../lib/components/ui/BlurText.svelte';
  import FadeContent from '../../lib/components/ui/FadeContent.svelte';
  import ShuffleText from '../../lib/components/ui/ShuffleText.svelte';

  const models = getModelsState();

  let selectedModel = $state<RegistryModel | null>(null);
  let showDialog = $state(false);
  let error = $state<string | null>(null);

  // Search: query string + explicit search button.
  let searchQuery = $state('');
  let appliedQuery = $state('');

  async function loadData() {
    models.setLoading(true);
    error = null;
    try {
      const reg = await invoke<{ models: RegistryModel[] }>('fetch_registry');
      models.setRegistry(reg.models);
      const downloaded = await invoke<DownloadedModel[]>('list_downloaded_models');
      models.setDownloadedModels(downloaded);
    } catch (e) {
      error = String(e);
    } finally {
      models.setLoading(false);
    }
  }

  $effect(() => {
    loadData();
  });

  // Apply the search (button click or Enter in the input).
  function applySearch() {
    appliedQuery = searchQuery.trim().toLowerCase();
  }

  function clearSearch() {
    searchQuery = '';
    appliedQuery = '';
  }

  const filteredModels = $derived(
    appliedQuery
      ? models.registry.filter((m) =>
          [m.name, m.id, m.hf_repo, m.description, m.format]
            .join(' ')
            .toLowerCase()
            .includes(appliedQuery)
        )
      : models.registry
  );

  function isDownloaded(modelId: string): boolean {
    return models.downloadedModels.some((d) => d.id === modelId);
  }

  function isBusy(modelId: string): boolean {
    // Cannot load while a conversion runs; also show a lock on the model being downloaded.
    return models.isConverting || (models.isDownloading && models.downloadProgress?.model_id === modelId);
  }

  function handleDownload(modelId: string, quant: string) {
    const model = models.registry.find((m) => m.id === modelId);
    if (model) {
      selectedModel = model;
      showDialog = true;
    }
  }

  async function handleLoad(modelId: string) {
    if (isBusy(modelId)) return;
    try {
      const { getSettingsState } = await import('../../lib/stores/settings.svelte');
      const settings = getSettingsState();
      await invoke('start_server', { modelId, port: settings.settings.port });
      models.setActiveModel(modelId);
    } catch (e) {
      error = String(e);
    }
  }

  function handleDialogClose() {
    showDialog = false;
    selectedModel = null;
  }

  function handleDialogLoad(modelId: string) {
    handleLoad(modelId);
    showDialog = false;
    selectedModel = null;
  }

  function formatSize(mb: number): string {
    if (mb >= 1024) return (mb / 1024).toFixed(1) + ' GB';
    return mb + ' MB';
  }
</script>

<div class="library-page">
  <div class="page-header">
    <BlurText
      text="Model Library"
      animateBy="words"
      direction="top"
      delay={60}
      duration={0.6}
      class="page-title"
    />
    <div class="header-actions">
      <span class="model-count">{filteredModels.length} models</span>
      <button class="refresh-btn" onclick={loadData} disabled={models.loading}>
        <svg
          width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
          stroke-linecap="round" stroke-linejoin="round"
          class:spinning={models.loading}
        >
          <polyline points="23 4 23 10 17 10" />
          <polyline points="1 20 1 14 7 14" />
          <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
        </svg>
        Refresh
      </button>
    </div>
  </div>

  <!-- Search bar -->
  <div class="search-bar">
    <svg class="search-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </svg>
    <input
      type="text"
      placeholder="Search models by name, id, or repo…"
      bind:value={searchQuery}
      onkeydown={(e) => { if (e.key === 'Enter') applySearch(); }}
    />
    <button class="search-btn" onclick={applySearch} title="Search models">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
      Search
    </button>
    {#if appliedQuery}
      <button class="search-clear" onclick={clearSearch} title="Clear search">×</button>
    {/if}
  </div>

  {#if error}
    <div class="error-banner">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
      </svg>
      {error}
    </div>
  {/if}

  {#if models.loading}
    <div class="loading">
      <div class="spinner"></div>
      <ShuffleText text="Loading models..." speed={0.04} duration={0.8} />
    </div>
  {:else if filteredModels.length === 0}
    <div class="empty-state">
      {#if appliedQuery}
        <span>No models match “{appliedQuery}”.</span>
        <button class="empty-clear" onclick={clearSearch}>Clear search</button>
      {:else}
        <span>No models in the registry.</span>
      {/if}
    </div>
  {:else}
    <FadeContent duration={0.4} stagger={0.03}>
      <div class="model-list">
        {#each filteredModels as model}
          {@const downloaded = isDownloaded(model.id)}
          {@const busy = isBusy(model.id)}
          <div class="model-row">
            <div class="row-info">
              <div class="row-title">
                <span class="row-name">{model.name || model.id}</span>
                {#if downloaded}
                  <span class="installed-badge">Installed</span>
                {/if}
              </div>
              <div class="row-sub">
                <span class="row-id">{model.id}</span>
                <span class="row-tags">
                  <span class="tag">{model.context} ctx</span>
                  <span class="tag">{formatSize(model.size_mb)}</span>
                  <span class="tag">{model.format}</span>
                </span>
              </div>
              <p class="row-desc">{model.description}</p>
            </div>
            <div class="row-actions">
              {#if busy}
                <button class="action-btn busy-btn" disabled title="Model is being downloaded/converted — loading is locked">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="23 4 23 10 17 10" />
                    <polyline points="1 20 1 14 7 14" />
                    <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
                  </svg>
                  {models.isConverting ? 'Converting…' : 'Downloading…'}
                </button>
              {:else if downloaded}
                <button class="action-btn load-btn" onclick={() => handleLoad(model.id)}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  Load
                </button>
              {:else}
                <button class="action-btn download-btn" onclick={() => handleDownload(model.id, 'ternary')}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                    <polyline points="7 10 12 15 17 10" />
                    <line x1="12" y1="15" x2="12" y2="3" />
                  </svg>
                  Download
                </button>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    </FadeContent>
  {/if}
</div>

<DownloadDialog
  model={selectedModel}
  show={showDialog}
  onClose={handleDialogClose}
  onLoad={handleDialogLoad}
/>

<style>
  .library-page {
    padding: 24px;
    max-width: 1100px;
    margin: 0 auto;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
  }

  :global(.page-title) {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: hsl(var(--content));
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .model-count {
    font-size: 13px;
    color: hsl(var(--content-muted));
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius-sm);
    background: hsla(var(--surface-secondary), 0.6);
    color: hsl(var(--content));
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .refresh-btn:hover:not(:disabled) {
    border-color: hsl(var(--brand));
  }

  .refresh-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .spinning {
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  /* ── Search bar ── */
  .search-bar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 8px 6px 14px;
    margin-bottom: 20px;
    background: hsla(var(--surface-secondary), 0.7);
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius);
    transition: border-color 0.15s;
  }

  .search-bar:focus-within {
    border-color: hsl(var(--brand));
  }

  .search-icon {
    color: hsl(var(--content-muted));
    flex-shrink: 0;
  }

  .search-bar input {
    flex: 1;
    min-width: 0;
    background: transparent;
    border: none;
    outline: none;
    color: hsl(var(--content));
    font-size: 14px;
    font-family: inherit;
    padding: 8px 0;
  }

  .search-bar input::placeholder {
    color: hsl(var(--content-muted));
  }

  .search-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border: none;
    border-radius: var(--radius-sm);
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: #fff;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    flex-shrink: 0;
  }

  .search-btn:hover {
    opacity: 0.9;
    box-shadow: 0 0 12px hsla(var(--brand), 0.3);
  }

  .search-clear {
    background: none;
    border: none;
    color: hsl(var(--content-muted));
    font-size: 18px;
    line-height: 1;
    cursor: pointer;
    padding: 4px 6px;
    border-radius: 4px;
    flex-shrink: 0;
  }

  .search-clear:hover {
    color: hsl(var(--content));
    background: hsla(var(--surface-tertiary), 0.6);
  }

  .error-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: hsla(var(--danger), 0.1);
    border: 1px solid hsla(var(--danger), 0.3);
    border-radius: var(--radius-sm);
    color: hsl(var(--danger));
    font-size: 13px;
    margin-bottom: 20px;
  }

  .loading {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 80px;
    gap: 16px;
    color: hsl(var(--content-muted));
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid hsla(var(--surface-tertiary), 0.8);
    border-top-color: hsl(var(--brand));
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
    padding: 60px;
    color: hsl(var(--content-muted));
    font-size: 14px;
  }

  .empty-clear {
    padding: 8px 14px;
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius-sm);
    background: hsla(var(--surface-secondary), 0.6);
    color: hsl(var(--content));
    font-size: 13px;
    cursor: pointer;
  }

  .empty-clear:hover {
    border-color: hsl(var(--brand));
  }

  /* ── Vertical model list ── */
  .model-list {
    display: flex;
    flex-direction: column;
    gap: 10px;
  }

  .model-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 16px 20px;
    background: hsla(var(--surface-secondary), 0.55);
    border: 1px solid hsla(var(--border), 0.5);
    border-radius: var(--radius);
    transition: all 0.2s;
  }

  .model-row:hover {
    border-color: hsl(var(--brand));
    box-shadow: var(--shadow-glow);
  }

  .row-info {
    min-width: 0;
    flex: 1;
  }

  .row-title {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .row-name {
    font-size: 15px;
    font-weight: 600;
    color: hsl(var(--content));
  }

  .installed-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    background: hsla(var(--success), 0.15);
    color: hsl(var(--success));
    flex-shrink: 0;
  }

  .row-sub {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 4px;
    flex-wrap: wrap;
  }

  .row-id {
    font-size: 11px;
    color: hsl(var(--content-muted));
    font-family: monospace;
  }

  .row-tags {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }

  .tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 4px;
    background: hsla(var(--surface-tertiary), 0.8);
    color: hsl(var(--content-muted));
    font-weight: 500;
  }

  .row-desc {
    margin: 6px 0 0;
    font-size: 13px;
    color: hsl(var(--content-muted));
    line-height: 1.5;
    overflow: hidden;
    text-overflow: ellipsis;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
  }

  .row-actions {
    flex-shrink: 0;
  }

  .action-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 10px 20px;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
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

  .busy-btn {
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content-muted));
    cursor: not-allowed;
    opacity: 0.85;
  }
</style>

<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import type { RegistryModel, DownloadedModel } from '../../types';
  import ModelCard from '../../lib/components/ModelCard.svelte';
  import DownloadDialog from '../../lib/components/DownloadDialog.svelte';
  import { getModelsState } from '../../lib/stores/models';

  const models = getModelsState();

  let selectedModel = $state<RegistryModel | null>(null);
  let showDialog = $state(false);
  let error = $state<string | null>(null);

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

  function isDownloaded(modelId: string): boolean {
    return models.downloadedModels.some((d) => d.id === modelId);
  }

  function handleDownload(modelId: string, quant: string) {
    const model = models.registry.find((m) => m.id === modelId);
    if (model) {
      selectedModel = model;
      showDialog = true;
    }
  }

  async function handleLoad(modelId: string) {
    try {
      const { getSettingsState } = await import('../../lib/stores/settings');
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
</script>

<div class="library-page">
  <div class="page-header">
    <h1>Model Library</h1>
    <div class="header-actions">
      <span class="model-count">{models.registry.length} models</span>
      <button class="refresh-btn" onclick={loadData} disabled={models.loading}>
        <svg
          width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
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

  {#if error}
    <div class="error-banner">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="10" /><line x1="15" y1="9" x2="9" y2="15" /><line x1="9" y1="9" x2="15" y2="15" />
      </svg>
      {error}
    </div>
  {/if}

  {#if models.loading}
    <div class="loading">
      <div class="spinner"></div>
      <span>Loading models...</span>
    </div>
  {:else}
    <div class="model-grid">
      {#each models.registry as model}
        <ModelCard
          {model}
          isDownloaded={isDownloaded(model.id)}
          onDownload={handleDownload}
          onLoad={handleLoad}
        />
      {/each}
    </div>
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
    max-width: 1200px;
    margin: 0 auto;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
  }

  h1 {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
  }

  .header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .model-count {
    font-size: 13px;
    color: var(--text-secondary);
  }

  .refresh-btn {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 14px;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    background: var(--bg-secondary);
    color: var(--text-primary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .refresh-btn:hover:not(:disabled) {
    border-color: var(--accent);
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

  .error-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.3);
    border-radius: var(--radius-sm);
    color: var(--danger);
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
    color: var(--text-secondary);
  }

  .spinner {
    width: 32px;
    height: 32px;
    border: 3px solid var(--bg-tertiary);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  .model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }
</style>

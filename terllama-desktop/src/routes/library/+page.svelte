<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import type { RegistryModel, DownloadedModel } from '../../types';
  import ModelCard from '../../lib/components/ModelCard.svelte';
  import DownloadDialog from '../../lib/components/DownloadDialog.svelte';
  import { getModelsState } from '../../lib/stores/models.svelte';
  import BlurText from '../../lib/components/ui/BlurText.svelte';
  import FadeContent from '../../lib/components/ui/FadeContent.svelte';
  import ShinyText from '../../lib/components/ui/ShinyText.svelte';
  import ShuffleText from '../../lib/components/ui/ShuffleText.svelte';

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
      <span class="model-count">{models.registry.length} models</span>
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
  {:else}
    <FadeContent duration={0.4} stagger={0.04}>
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
    max-width: 1200px;
    margin: 0 auto;
  }

  .page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
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

  .model-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: 16px;
  }
</style>

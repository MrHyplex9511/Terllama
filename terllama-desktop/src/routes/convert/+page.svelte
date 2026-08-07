<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import BlurText from '../../lib/components/ui/BlurText.svelte';
  import FadeContent from '../../lib/components/ui/FadeContent.svelte';

  let modelName = $state('HuggingFaceTB/SmolLM2-135M');
  let format = $state('als');
  let terms = $state(12);

  let engineStatus = $state<string>('');
  let engineOk = $state<boolean | null>(null);
  let isConverting = $state(false);
  let output = $state<string[]>([]);
  let outputEl: HTMLDivElement | undefined = $state();

  $effect(() => {
    checkEngine();
  });

  async function checkEngine() {
    engineStatus = 'Checking...';
    try {
      const info = await invoke<string>('check_engine');
      engineStatus = info;
      engineOk = true;
    } catch (e: any) {
      engineStatus = e;
      engineOk = false;
    }
  }

  async function startConvert() {
    if (isConverting) return;
    output = [];
    isConverting = true;

    const unlisten = await listen<any>('convert-progress', (event) => {
      const { line, done, error } = event.payload;
      output = [...output, line];
      if (done) {
        isConverting = false;
        if (error) {
          output = [...output, `\nError: ${error}`];
        }
      }
      if (outputEl) {
        requestAnimationFrame(() => {
          outputEl.scrollTop = outputEl.scrollHeight;
        });
      }
    });

    try {
      await invoke('convert_model', {
        model: modelName.trim(),
        format,
        terms,
      });
    } catch (e: any) {
      output = [...output, `\n❌ ${e}`];
      isConverting = false;
    } finally {
      unlisten();
    }
  }

  async function cancelConvert() {
    try {
      await invoke('cancel_conversion');
      output = [...output, '\n⏳ Cancelling...'];
    } catch (e: any) {
      output = [...output, `\nError: ${e}`];
    }
  }

  function clearOutput() {
    output = [];
  }
</script>

<div class="convert-page">
  <BlurText text="Convert Model" animateBy="words" direction="top" delay={60} duration={0.6} class="convert-title" />
  <p class="subtitle">Download and ternarize HuggingFace models for local inference</p>

  <FadeContent duration={0.4} delay={0.05}>
  <!-- Engine Status -->
  <div class="card-base status-card">
    <div class="status-row">
      <span class="label">Engine Status</span>
      <span
        class="value"
        class:ok={engineOk === true}
        class:err={engineOk === false}
      >
        {engineStatus || 'Checking...'}
      </span>
    </div>
    {#if engineOk === false && engineStatus}
      <div class="deps-warning">
        <p><strong>Conversion engine not ready.</strong></p>
        <button class="btn-sm" onclick={checkEngine}>Retry</button>
      </div>
    {:else if engineOk === true}
      <div class="deps-ok">✓ Native engine ready (no Python required)</div>
    {/if}
  </div>

  <!-- Config Card -->
  <div class="card-base config-card">
    <div class="field">
      <label for="modelName">HuggingFace Model</label>
      <input
        id="modelName"
        type="text"
        bind:value={modelName}
        placeholder="e.g. HuggingFaceTB/SmolLM2-135M"
        disabled={isConverting}
      />
      <p class="hint">Full HuggingFace model ID (e.g. mistralai/Mistral-7B-v0.3)</p>
    </div>

    <div class="field-row">
      <div class="field">
        <label for="format">Format</label>
        <select id="format" bind:value={format} disabled={isConverting}>
          <option value="als">ALS (higher quality, slower)</option>
        </select>
        <p class="hint">Multi-term rank-1 ternary decomposition</p>
      </div>

      {#if format === 'als'}
        <div class="field">
          <label for="terms">ALS Terms ({terms})</label>
          <input id="terms" type="range" bind:value={terms} min="4" max="24" step="2" disabled={isConverting} />
          <p class="hint">More terms = higher quality but larger file</p>
        </div>
      {/if}
    </div>

    <div class="actions">
      {#if !isConverting}
        <button class="btn-primary" onclick={startConvert} disabled={!engineOk || !modelName.trim()}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <polyline points="1 20 1 14 7 14" />
            <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
          </svg>
          Convert
        </button>
      {:else}
        <button class="btn-danger" onclick={cancelConvert}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
          Cancel
        </button>
      {/if}

      {#if output.length > 0 && !isConverting}
        <button class="btn-secondary" onclick={clearOutput}>Clear Output</button>
      {/if}
    </div>
  </div>

  <!-- Output Terminal -->
  {#if output.length > 0}
    <div class="card-base output-card">
      <div class="output-header">
        <span class="output-title">Output</span>
        <span class="output-status" class:running={isConverting}>
          {isConverting ? 'Running...' : 'Done'}
        </span>
      </div>
      <div class="output-terminal" bind:this={outputEl}>
        {#each output as line}
          <div class="line">{line}</div>
        {/each}
      </div>
    </div>
  {/if}
  </FadeContent>
</div>

<style>
  .convert-page {
    padding: 32px;
    max-width: 720px;
    margin: 0 auto;
  }

  :global(.convert-title) {
    font-size: 24px;
    font-weight: 700;
    margin: 0 0 4px;
    color: hsl(var(--content));
  }

  .subtitle {
    color: hsl(var(--content-muted));
    font-size: 14px;
    margin: 0 0 24px;
  }

  .status-card {
    padding: 16px;
    margin-bottom: 16px;
  }

  .status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .label {
    font-size: 13px;
    font-weight: 600;
    color: hsl(var(--content-muted));
  }

  .value {
    font-size: 13px;
    font-family: 'JetBrains Mono', monospace;
    color: hsl(var(--content));
  }

  .value.ok { color: hsl(var(--success)); }
  .value.err { color: hsl(var(--danger)); }

  .deps-warning {
    margin-top: 12px;
    padding: 12px;
    background: hsla(var(--warning), 0.1);
    border: 1px solid hsla(var(--warning), 0.3);
    border-radius: var(--radius-sm);
    font-size: 13px;
  }

  .deps-warning p { margin: 0 0 8px; }

  .deps-ok {
    margin-top: 8px;
    font-size: 13px;
    color: hsl(var(--success));
  }

  .config-card {
    padding: 20px;
    margin-bottom: 16px;
  }

  .field {
    margin-bottom: 16px;
  }

  .field-row {
    display: flex;
    gap: 16px;
  }
  .field-row .field {
    flex: 1;
  }

  label {
    display: block;
    font-size: 13px;
    font-weight: 600;
    color: hsl(var(--content-muted));
    margin-bottom: 6px;
  }

  input[type="text"], select {
    width: 100%;
    padding: 10px 14px;
    background: hsl(var(--surface));
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius-sm);
    color: hsl(var(--content));
    font-size: 14px;
    font-family: inherit;
    outline: none;
    transition: border-color 0.15s;
  }

  input[type="text"]:focus, select:focus {
    border-color: hsl(var(--brand));
  }

  input[type="range"] {
    width: 100%;
    height: 6px;
    -webkit-appearance: none;
    appearance: none;
    background: hsl(var(--border));
    border-radius: 3px;
    outline: none;
    cursor: pointer;
  }

  input[type="range"]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    cursor: pointer;
    border: 2px solid hsl(var(--surface-secondary));
  }

  .hint {
    font-size: 12px;
    color: hsl(var(--content-muted));
    margin: 4px 0 0;
  }

  .actions {
    display: flex;
    gap: 8px;
    margin-top: 8px;
  }

  .btn-primary, .btn-danger, .btn-secondary {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 10px 20px;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn-primary {
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: white;
  }
  .btn-primary:hover { opacity: 0.9; box-shadow: 0 0 12px hsla(var(--brand), 0.3); }
  .btn-primary:disabled { opacity: 0.4; cursor: not-allowed; }

  .btn-danger {
    background: hsl(var(--danger));
    color: white;
  }
  .btn-danger:hover { opacity: 0.9; }

  .btn-secondary {
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content));
  }
  .btn-secondary:hover { opacity: 0.8; }

  .btn-sm {
    padding: 4px 12px;
    border: 1px solid hsl(var(--border));
    border-radius: 6px;
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content));
    font-size: 12px;
    cursor: pointer;
  }
  .btn-sm:hover { opacity: 0.8; }

  .output-card {
    padding: 0;
    overflow: hidden;
  }

  .output-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 1px solid hsl(var(--border));
    background: hsla(var(--surface-tertiary), 0.4);
  }

  .output-title {
    font-size: 13px;
    font-weight: 600;
    color: hsl(var(--content));
  }

  .output-status {
    font-size: 12px;
    color: hsl(var(--content-muted));
  }
  .output-status.running {
    color: hsl(var(--brand));
    animation: pulse 1.5s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .output-terminal {
    max-height: 400px;
    overflow-y: auto;
    padding: 16px;
    background: #0a0a14;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.6;
  }

  .line {
    white-space: pre-wrap;
    word-break: break-all;
    color: #c8c8d8;
  }
</style>

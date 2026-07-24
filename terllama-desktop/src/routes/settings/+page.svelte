<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { open } from '@tauri-apps/plugin-dialog';
  import { getSettingsState } from '../../lib/stores/settings.svelte';
  import SettingsToggle from '../../lib/components/SettingsToggle.svelte';
  import BlurText from '../../lib/components/ui/BlurText.svelte';
  import FadeContent from '../../lib/components/ui/FadeContent.svelte';
  import GridScan from '../../lib/components/bg/GridScan.svelte';

  const s = getSettingsState();

  let saveStatus = $state<string | null>(null);
  let updateStatus = $state<{ text: string; available: boolean } | null>(null);

  async function checkNow() {
    updateStatus = { text: 'Checking...', available: false };
    try {
      const info: any = await invoke('check_update');
      if (info.update_available) {
        updateStatus = {
          text: `v${info.latest_version} available! Download from GitHub`,
          available: true,
        };
      } else {
        updateStatus = { text: 'Up to date (v1.0.0)', available: false };
      }
    } catch {
      updateStatus = { text: 'Could not check (offline?)', available: false };
    }
  }

  async function browseModelDir() {
    const dir = await open({
      directory: true,
      title: 'Select Model Directory',
    });
    if (dir) {
      s.updateSetting('modelDir', dir);
      await save();
    }
  }

  async function save() {
    await s.saveSettings();
    saveStatus = 'Saved!';
    setTimeout(() => (saveStatus = null), 2000);
  }
</script>

<div class="settings-page">
  <div class="settings-header">
    <BlurText text="Settings" animateBy="words" direction="top" delay={60} duration={0.6} class="settings-title" />
    <GridScan color="hsla(var(--brand), 0.03)" lineColor="hsla(var(--brand), 0.06)" speed={6} />
  </div>

  <!-- Inference Mode -->
  <FadeContent duration={0.4} delay={0.05}>
    <div class="card-base settings-card">
      <h2>Inference Mode</h2>
      <p class="card-desc">Choose how models are loaded and executed.</p>

      <div class="mode-cards">
        <div class="mode-card selected">
          <div class="mode-header">
            <span class="mode-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="4" y="4" width="16" height="16" rx="2" />
                <rect x="9" y="9" width="6" height="6" />
              </svg>
            </span>
            <span class="mode-title">Ternary (CPU)</span>
            <span class="mode-badge">Active</span>
          </div>
          <p class="mode-desc">1.58-bit ternary precision. Fast CPU inference using Terllama's native engine. Supports Q2_0 GGUF and native i2s/als formats.</p>
          <ul class="mode-features">
            <li>✓ Low RAM usage</li>
            <li>✓ Fast CPU inference</li>
            <li>✓ Native Terllama format</li>
          </ul>
        </div>

        <div class="mode-card disabled">
          <div class="mode-header">
            <span class="mode-icon">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </span>
            <span class="mode-title">Standard GGUF (CPU/GPU)</span>
            <span class="mode-badge planned">Planned</span>
          </div>
          <p class="mode-desc">Full-precision GGUF with Q4_K_M and Q8_0 quantization. Requires llama.cpp backend. Coming in a future release.</p>
          <ul class="mode-features">
            <li>✗ Not available in v1.0.0</li>
            <li>✗ Requires llama.cpp runner</li>
          </ul>
        </div>
      </div>
    </div>
  </FadeContent>

  <!-- System Config -->
  <FadeContent duration={0.4} delay={0.1}>
    <div class="card-base settings-card">
      <h2>System Configuration</h2>

      <div class="setting-row">
        <div class="setting-label">
          <span>Model Directory</span>
          <span class="setting-desc">Where downloaded models are stored</span>
        </div>
        <div class="setting-control">
          <input
            type="text"
            value={s.settings.modelDir}
            oninput={(e) => s.updateSetting('modelDir', e.currentTarget.value)}
            class="text-input"
          />
          <button class="browse-btn" onclick={browseModelDir}>Browse</button>
        </div>
      </div>

      <div class="setting-row">
        <div class="setting-label">
          <span>API Port</span>
          <span class="setting-desc">Server port for the inference API</span>
        </div>
        <input
          type="number"
          value={s.settings.port}
          oninput={(e) => s.updateSetting('port', parseInt(e.currentTarget.value) || 8375)}
          class="number-input"
          min="1024"
          max="65535"
        />
      </div>

      <div class="setting-row">
        <div class="setting-label">
          <span>Keep-Alive</span>
          <span class="setting-desc">Seconds to keep model loaded after last request</span>
        </div>
        <input
          type="number"
          value={s.settings.keepAlive}
          oninput={(e) => s.updateSetting('keepAlive', parseInt(e.currentTarget.value) || 300)}
          class="number-input"
          min="0"
          max="3600"
        />
      </div>

      <SettingsToggle
        checked={s.settings.autoStart}
        onChange={(v) => { s.updateSetting('autoStart', v); save(); }}
        label="Auto-start server on launch"
        description="Start the inference server when Terllama opens"
      />

      <div class="setting-row">
        <div class="setting-label">
          <span>Theme</span>
        </div>
        <div class="theme-selector">
          {#each ['dark', 'light', 'system'] as t}
            <button
              class="theme-btn"
              class:active={s.settings.theme === t}
              onclick={() => { s.updateSetting('theme', t as any); save(); }}
            >
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          {/each}
        </div>
      </div>
    </div>
  </FadeContent>

  <!-- Advanced -->
  <FadeContent duration={0.4} delay={0.15}>
    <div class="card-base settings-card">
      <h2>Advanced</h2>

      <div class="setting-row">
        <div class="setting-label">
          <span>CPU Threads</span>
          <span class="setting-desc">Number of threads for inference</span>
        </div>
        <input
          type="number"
          value={s.settings.cpuThreads}
          oninput={(e) => s.updateSetting('cpuThreads', parseInt(e.currentTarget.value) || 4)}
          class="number-input"
          min="1"
          max="64"
        />
      </div>
    </div>
  </FadeContent>

  <!-- About -->
  <FadeContent duration={0.4} delay={0.2}>
    <div class="card-base settings-card">
      <h2>About</h2>
      <div class="about-info">
        <div class="about-row">
          <span>Version</span>
          <span class="value">v1.0.0</span>
        </div>
        <div class="about-row">
          <span>Engine</span>
          <span class="value">Terllama (ternary LLM) — Animations by React Bits</span>
        </div>
        <div class="about-row">
          <span>Repository</span>
          <a href="https://github.com/MrHyplex9511/Terllama" target="_blank" class="link">github.com/MrHyplex9511/Terllama</a>
        </div>
        <div class="about-row">
          <span></span>
          <button class="check-update-btn" onclick={checkNow}>Check for Updates</button>
        </div>
        {#if updateStatus}
          <div class="about-row">
            <span></span>
            <span class:update-available={updateStatus.available} class:up-to-date={!updateStatus.available}>
              {updateStatus.text}
            </span>
          </div>
        {/if}
      </div>
    </div>
  </FadeContent>

  {#if saveStatus}
    <div class="save-toast">{saveStatus}</div>
  {/if}
</div>

<style>
  .settings-page {
    padding: 24px;
    max-width: 720px;
    margin: 0 auto;
  }

  .settings-header {
    position: relative;
    margin-bottom: 24px;
    overflow: hidden;
    border-radius: var(--radius);
    min-height: 48px;
  }

  :global(.settings-title) {
    margin: 0;
    font-size: 24px;
    font-weight: 700;
    color: hsl(var(--content));
    position: relative;
    z-index: 1;
  }

  .settings-card {
    padding: 24px;
    margin-bottom: 16px;
  }

  .settings-card h2 {
    margin: 0 0 4px;
    font-size: 16px;
    font-weight: 600;
    color: hsl(var(--content));
  }

  .card-desc {
    margin: 0 0 20px;
    font-size: 13px;
    color: hsl(var(--content-muted));
  }

  .mode-cards {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }

  .mode-card {
    padding: 16px;
    border: 1px solid hsla(var(--border), 0.5);
    border-radius: var(--radius-sm);
    background: hsla(var(--surface-tertiary), 0.4);
  }

  .mode-card.selected {
    border-color: hsl(var(--brand));
    background: hsla(var(--brand), 0.08);
  }

  .mode-card.disabled {
    opacity: 0.5;
  }

  .mode-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
  }

  .mode-icon {
    color: hsl(var(--brand));
  }

  .mode-title {
    font-size: 14px;
    font-weight: 600;
    flex: 1;
    color: hsl(var(--content));
  }

  .mode-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
  }

  .mode-badge:not(.planned) {
    background: hsla(var(--success), 0.15);
    color: hsl(var(--success));
  }

  .mode-badge.planned {
    background: hsla(var(--warning), 0.15);
    color: hsl(var(--warning));
  }

  .mode-desc {
    font-size: 12px;
    color: hsl(var(--content-muted));
    line-height: 1.5;
    margin: 0 0 10px;
  }

  .mode-features {
    list-style: none;
    padding: 0;
    margin: 0;
    font-size: 12px;
    color: hsl(var(--content-muted));
  }

  .mode-features li {
    padding: 2px 0;
  }

  .setting-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 0;
    border-bottom: 1px solid hsl(var(--border));
    gap: 16px;
  }

  .setting-row:last-of-type {
    border-bottom: none;
  }

  .setting-label {
    display: flex;
    flex-direction: column;
    gap: 2px;
    font-size: 14px;
    font-weight: 500;
    color: hsl(var(--content));
  }

  .setting-desc {
    font-size: 12px;
    font-weight: 400;
    color: hsl(var(--content-muted));
  }

  .setting-control {
    display: flex;
    gap: 8px;
    align-items: center;
  }

  .text-input {
    background: hsla(var(--surface-tertiary), 0.6);
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius-sm);
    padding: 8px 12px;
    color: hsl(var(--content));
    font-size: 13px;
    width: 240px;
    font-family: monospace;
  }

  .text-input:focus {
    border-color: hsl(var(--brand));
    outline: none;
  }

  .number-input {
    background: hsla(var(--surface-tertiary), 0.6);
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius-sm);
    padding: 8px 12px;
    color: hsl(var(--content));
    font-size: 13px;
    width: 100px;
  }

  .number-input:focus {
    border-color: hsl(var(--brand));
    outline: none;
  }

  .browse-btn {
    padding: 8px 14px;
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius-sm);
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content));
    font-size: 13px;
    cursor: pointer;
  }

  .browse-btn:hover {
    background: hsla(var(--border), 0.6);
  }

  .theme-selector {
    display: flex;
    gap: 4px;
  }

  .theme-btn {
    padding: 6px 14px;
    border: 1px solid hsl(var(--border));
    border-radius: var(--radius-sm);
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content-muted));
    font-size: 13px;
    cursor: pointer;
  }

  .theme-btn.active {
    border-color: hsl(var(--brand));
    color: hsl(var(--brand-hover));
    background: hsla(var(--brand), 0.1);
  }

  .check-update-btn {
    padding: 6px 14px;
    border: 1px solid hsl(var(--brand));
    border-radius: var(--radius-sm);
    background: transparent;
    color: hsl(var(--brand-hover));
    font-size: 12px;
    cursor: pointer;
    transition: all 0.15s;
  }

  .check-update-btn:hover {
    background: hsla(var(--brand), 0.1);
  }

  .update-available {
    color: hsl(var(--brand-hover));
    font-weight: 500;
    font-size: 13px;
  }

  .up-to-date {
    color: hsl(var(--success));
    font-size: 13px;
  }

  .about-info {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .about-row {
    display: flex;
    justify-content: space-between;
    font-size: 14px;
    color: hsl(var(--content-muted));
  }

  .about-row .value {
    color: hsl(var(--content));
  }

  .link {
    color: hsl(var(--brand-hover));
    text-decoration: none;
  }

  .link:hover {
    text-decoration: underline;
  }

  .save-toast {
    position: fixed;
    bottom: 24px;
    right: 24px;
    padding: 10px 20px;
    background: hsla(var(--success), 0.15);
    border: 1px solid hsla(var(--success), 0.3);
    border-radius: var(--radius-sm);
    color: hsl(var(--success));
    font-size: 13px;
    font-weight: 500;
    backdrop-filter: blur(8px);
  }
</style>

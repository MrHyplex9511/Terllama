<script lang="ts">
  import { onMount } from 'svelte';
  import { getSettingsState } from '../lib/stores/settings.svelte';

  const settings = getSettingsState();

  let isOnboarded = false;

  onMount(() => {
    isOnboarded = !!localStorage.getItem('terllama-onboarded');
    settings.loadSettings();
  });
</script>

{#if isOnboarded}
  <div class="splash">
    <div class="logo">Terllama</div>
    <p class="subtitle">CPU-first ternary LLM inference</p>
    <div class="actions">
      <a href="/library" class="btn">Browse Models</a>
      <a href="/chat" class="btn secondary">Open Chat</a>
    </div>
  </div>
{/if}

<script lang="ts" context="module">
  // Empty module script for redirect logic
</script>

<style>
  .splash {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 16px;
  }

  .logo {
    font-size: 36px;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .subtitle {
    color: var(--text-secondary);
    font-size: 16px;
  }

  .actions {
    display: flex;
    gap: 12px;
    margin-top: 24px;
  }

  .btn {
    padding: 12px 24px;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 600;
    text-decoration: none;
    background: var(--accent-gradient);
    color: white;
  }

  .btn.secondary {
    background: var(--bg-tertiary);
    color: var(--text-primary);
    border: 1px solid var(--border);
  }
</style>

<script lang="ts">
  import { getSettingsState } from '../lib/stores/settings.svelte';
  import { getModelsState } from '../lib/stores/models.svelte';
  import { getChatState } from '../lib/stores/chat.svelte';
  import Sidebar from '../lib/components/Sidebar.svelte';
  import Onboarding from '../lib/components/Onboarding.svelte';
  import LibraryPage from './library/+page.svelte';
  import ChatPage from './chat/+page.svelte';
  import ConvertPage from './convert/+page.svelte';
  import SettingsPage from './settings/+page.svelte';
  import Dither from '../lib/components/bg/Dither.svelte';
  import BlobCursor from '../lib/components/ui/BlobCursor.svelte';
  import '../app.css';

  const settings = getSettingsState();
  const models = getModelsState();
  const chat = getChatState();

  let currentRoute = $state('library');
  let showOnboarding = $state(false);
  let updateInfo = $state<{ latest: string; current: string; url: string } | null>(null);

  let viewTitle = $derived(
    currentRoute === 'library' ? 'Model Library'
    : currentRoute === 'chat' ? 'Chat'
    : currentRoute === 'settings' ? 'Settings'
    : 'Convert'
  );

  // Load settings on mount
  $effect(() => {
    settings.loadSettings();
    chat.loadSessions();

    // Check if first launch
    const onboarded = localStorage.getItem('terllama-onboarded');
    if (!onboarded) {
      showOnboarding = true;
    }

    // Check for updates on startup
    checkForUpdates();
  });

  async function checkForUpdates() {
    try {
      const { invoke } = await import('@tauri-apps/api/core');
      const info: any = await invoke('check_update');
      if (info.update_available) {
        updateInfo = {
          latest: info.latest_version,
          current: info.current_version,
          url: info.download_url,
        };
      }
    } catch {
      // Silently fail
    }
  }

  // Server status polling
  let serverStatus = $state('Stopped');
  $effect(() => {
    const interval = setInterval(async () => {
      try {
        const { invoke } = await import('@tauri-apps/api/core');
        const status: any = await invoke('server_status');
        serverStatus = typeof status === 'string' ? status : 'Running';
      } catch {
        serverStatus = 'Stopped';
      }
    }, 5000);
    return () => clearInterval(interval);
  });
</script>

{#if showOnboarding}
  <Onboarding onComplete={() => (showOnboarding = false)} />
{/if}

{#if updateInfo}
  <div class="update-banner">
    <span>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align: middle; margin-right: 8px;">
        <polyline points="23 4 23 10 17 10" />
        <polyline points="1 20 1 14 7 14" />
        <path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" />
      </svg>
      Update <strong>v{updateInfo.latest}</strong> available (you have v{updateInfo.current})
    </span>
    <div class="update-actions">
      <a href={updateInfo.url} target="_blank" rel="noopener noreferrer" class="update-btn">Download</a>
      <button class="update-dismiss" onclick={() => (updateInfo = null)}>×</button>
    </div>
  </div>
{/if}

<div class="app-layout">
  <Sidebar bind:currentRoute />

  <div class="main-area">
    <!-- Subtle noise texture + cursor glow -->
    <Dither opacity={0.02} />
    <BlobCursor size={300} blur={100} followSpeed={0.08} />

    <!-- Header bar (status only — no navigation) -->
    <header class="header-bar">
      <div class="header-left">
        <h1 class="header-title">{viewTitle}</h1>
      </div>
      <div class="header-right">
        {#if models.activeModel}
          <span class="active-model-badge" title="Active model">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
              <polyline points="17 6 23 6 23 12" />
            </svg>
            {models.activeModel}
          </span>
        {/if}

        <span class="server-status" class:running={serverStatus === 'Running'}>
          <span class="status-dot"></span>
          {serverStatus}
        </span>

        <button
          class="mode-toggle"
          class:gguf={settings.settings.ggufMode}
          onclick={() => settings.updateSetting('ggufMode', !settings.settings.ggufMode)}
          title={settings.settings.ggufMode ? 'Switch to Ternary mode' : 'Switch to GGUF mode'}
        >
          <span class="mode-dot" class:active={!settings.settings.ggufMode}></span>
          Ternary
          <span class="mode-divider"></span>
          <span class="mode-dot" class:active={settings.settings.ggufMode}></span>
          GGUF
        </button>
      </div>
    </header>

    <!-- Content -->
    <main class="content">
      {#if currentRoute === 'library'}
        <LibraryPage />
      {:else if currentRoute === 'chat'}
        <ChatPage />
      {:else if currentRoute === 'convert'}
        <ConvertPage />
      {:else if currentRoute === 'settings'}
        <SettingsPage />
      {/if}
    </main>
  </div>
</div>

<style>
  .update-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 16px;
    padding: 8px 20px;
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    color: white;
    font-size: 13px;
    z-index: 100;
    position: relative;
  }
  .update-actions {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .update-btn {
    padding: 4px 14px;
    background: white;
    color: #7c3aed;
    font-weight: 600;
    font-size: 12px;
    border-radius: 6px;
    text-decoration: none;
    transition: opacity 0.15s;
  }
  .update-btn:hover { opacity: 0.9; }
  .update-dismiss {
    background: none;
    border: none;
    color: white;
    font-size: 20px;
    cursor: pointer;
    padding: 0 4px;
    opacity: 0.7;
  }
  .update-dismiss:hover { opacity: 1; }

  .app-layout {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  .main-area {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
  }

  .header-bar {
    height: 48px;
    border-bottom: 1px solid hsl(var(--border));
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    flex-shrink: 0;
    background: hsla(var(--surface-secondary), 0.6);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    gap: 16px;
  }

  .header-left {
    display: flex;
    align-items: center;
  }

  .header-title {
    margin: 0;
    font-size: 16px;
    font-weight: 600;
    color: hsl(var(--content));
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .active-model-badge {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: hsl(var(--brand-hover));
    padding: 4px 10px;
    background: hsla(var(--brand), 0.1);
    border-radius: 6px;
    max-width: 200px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .server-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 11px;
    color: hsl(var(--content-muted));
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: hsl(var(--danger));
    flex-shrink: 0;
  }

  .server-status.running .status-dot {
    background: hsl(var(--success));
    box-shadow: 0 0 6px hsla(var(--success), 0.5);
  }

  .mode-toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border: 1px solid hsl(var(--border));
    border-radius: 20px;
    background: hsla(var(--surface-tertiary), 0.5);
    color: hsl(var(--content-muted));
    font-size: 11px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.02em;
  }

  .mode-toggle:hover {
    border-color: hsl(var(--brand));
  }

  .mode-toggle.gguf {
    border-color: hsl(var(--gpu));
  }

  .mode-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: hsl(var(--content-muted));
    transition: all 0.2s;
  }

  .mode-dot.active {
    background: hsl(var(--brand));
    box-shadow: 0 0 6px hsla(var(--brand), 0.5);
  }

  .mode-toggle.gguf .mode-dot.active {
    background: hsl(var(--gpu));
    box-shadow: 0 0 6px hsla(var(--gpu), 0.5);
  }

  .mode-divider {
    width: 1px;
    height: 12px;
    background: hsl(var(--border));
  }

  .content {
    flex: 1;
    overflow-y: auto;
  }
</style>

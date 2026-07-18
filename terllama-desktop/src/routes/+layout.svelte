<script lang="ts">
  import { getSettingsState } from '../lib/stores/settings';
  import { getModelsState } from '../lib/stores/models';
  import { getChatState } from '../lib/stores/chat';
  import Sidebar from '../lib/components/Sidebar.svelte';
  import Onboarding from '../lib/components/Onboarding.svelte';
  import LibraryPage from './library/+page.svelte';
  import ChatPage from './chat/+page.svelte';
  import SettingsPage from './settings/+page.svelte';
  import '../app.css';

  const settings = getSettingsState();
  const models = getModelsState();
  const chat = getChatState();

  let currentRoute = $state('library');
  let sidebarCollapsed = $state(false);
  let showOnboarding = $state(false);
  let updateInfo = $state<{ latest: string; current: string; url: string } | null>(null);

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
      // Silently fail — offline or rate-limited
    }
  }

  function navigate(route: string) {
    currentRoute = route;
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
      <a href={updateInfo.url} target="_blank" rel="noopener noreferrer" class="update-btn">
        Download
      </a>
      <button class="update-dismiss" onclick={() => (updateInfo = null)}>×</button>
    </div>
  </div>
{/if}

<div class="app-layout">
  <Sidebar bind:currentRoute bind:collapsed={sidebarCollapsed} />

  <div class="main-area">
    <!-- Top bar -->
    <header class="topbar">
      <nav class="tabs">
        <button
          class="tab"
          class:active={currentRoute === 'library'}
          onclick={() => (currentRoute = 'library')}
        >Library</button>
        <button
          class="tab"
          class:active={currentRoute === 'chat'}
          onclick={() => (currentRoute = 'chat')}
        >Chat</button>
        <button
          class="tab"
          class:active={currentRoute === 'settings'}
          onclick={() => (currentRoute = 'settings')}
        >Settings</button>
      </nav>

      <div class="status-bar">
        <span class="server-status" class:running={serverStatus === 'Running'}>
          <span class="status-dot"></span>
          {serverStatus}
        </span>
        {#if models.activeModel}
          <span class="active-model">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
              <polyline points="17 6 23 6 23 12" />
            </svg>
            {models.activeModel}
          </span>
        {/if}
      </div>
    </header>

    <!-- Content -->
    <main class="content">
      {#if currentRoute === 'library'}
        <LibraryPage />
      {:else if currentRoute === 'chat'}
        <ChatPage />
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

  .update-btn:hover {
    opacity: 0.9;
  }

  .update-dismiss {
    background: none;
    border: none;
    color: white;
    font-size: 20px;
    cursor: pointer;
    padding: 0 4px;
    opacity: 0.7;
  }

  .update-dismiss:hover {
    opacity: 1;
  }

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

  .topbar {
    height: 48px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 20px;
    flex-shrink: 0;
    background: var(--bg-secondary);
  }

  .tabs {
    display: flex;
    gap: 4px;
  }

  .tab {
    padding: 8px 16px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    border-radius: var(--radius-sm);
    transition: all 0.15s;
  }

  .tab:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .tab.active {
    background: rgba(124, 58, 237, 0.15);
    color: var(--accent-hover);
  }

  .status-bar {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .server-status {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--danger);
  }

  .server-status.running .status-dot {
    background: var(--success);
  }

  .active-model {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--accent-hover);
  }

  .content {
    flex: 1;
    overflow-y: auto;
  }
</style>

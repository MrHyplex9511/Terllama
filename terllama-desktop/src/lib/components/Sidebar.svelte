<script lang="ts">
  import { getChatState } from '../stores/chat.svelte';

  let { currentRoute = $bindable('library') } = $props();

  const chat = getChatState();

  const navItems = [
    { id: 'chat', label: 'Chat', icon: 'chat' },
    { id: 'library', label: 'Models', icon: 'library' },
    { id: 'convert', label: 'Convert', icon: 'convert' },
    { id: 'settings', label: 'Settings', icon: 'settings' },
  ];
</script>

<aside class="sidebar">
  <!-- Logo -->
  <div class="logo">
    <span class="logo-icon">T</span>
  </div>

  <!-- Navigation (icon-only) -->
  <nav class="nav">
    {#each navItems as item}
      <button
        class="nav-item"
        class:active={currentRoute === item.id}
        onclick={() => (currentRoute = item.id)}
        title={item.label}
      >
        {#if item.icon === 'chat'}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
          </svg>
        {:else if item.icon === 'library'}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="3" y="14" width="7" height="7" rx="1" />
            <rect x="14" y="14" width="7" height="7" rx="1" />
          </svg>
        {:else if item.icon === 'convert'}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <polyline points="1 20 1 14 7 14" />
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
          </svg>
        {:else if item.icon === 'settings'}
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.32 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
          </svg>
        {/if}
      </button>
    {/each}
  </nav>

  <!-- Divider -->
  <div class="divider"></div>

  <!-- New Chat Button -->
  <button class="new-chat-btn" onclick={() => chat.newSession()} title="New Chat">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  </button>

  <!-- Chat Sessions -->
  <div class="sessions-list">
    {#each chat.sessions as session}
      <button
        class="session-item"
        class:active={session.id === chat.activeSessionId}
        onclick={() => chat.switchSession(session.id)}
        title={session.title}
      >
        <span class="session-indicator" class:active={session.id === chat.activeSessionId}></span>
      </button>
    {/each}
  </div>
</aside>

<style>
  .sidebar {
    width: 56px;
    height: 100vh;
    background: hsla(var(--surface-secondary), 0.8);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid hsla(var(--border), 0.5);
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
    overflow: hidden;
    z-index: 10;
  }

  .logo {
    padding: 16px 0 12px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .logo-icon {
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 14px;
    color: #fff;
    flex-shrink: 0;
  }

  .nav {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    padding: 0 8px;
    width: 100%;
  }

  .nav-item {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 10px;
    border: none;
    background: transparent;
    color: hsl(var(--content-muted));
    cursor: pointer;
    transition: all 0.15s;
    position: relative;
  }

  .nav-item:hover {
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content));
  }

  .nav-item.active {
    background: hsla(var(--brand), 0.15);
    color: hsl(var(--brand-hover));
    box-shadow: 0 0 12px hsla(var(--brand), 0.15);
  }

  .divider {
    width: 24px;
    height: 1px;
    background: hsla(var(--border), 0.6);
    margin: 8px 0;
  }

  .new-chat-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 10px;
    border: none;
    background: transparent;
    color: hsl(var(--content-muted));
    cursor: pointer;
    transition: all 0.15s;
    margin-bottom: 2px;
  }

  .new-chat-btn:hover {
    background: hsla(var(--surface-tertiary), 0.6);
    color: hsl(var(--content));
  }

  .sessions-list {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    padding: 0 8px 8px;
    width: 100%;
  }

  .session-item {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    border: none;
    background: transparent;
    cursor: pointer;
    transition: all 0.15s;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .session-item:hover {
    background: hsla(var(--surface-tertiary), 0.6);
  }

  .session-item.active {
    background: hsla(var(--brand), 0.12);
  }

  .session-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: hsl(var(--content-muted));
    transition: all 0.15s;
  }

  .session-indicator.active {
    background: hsl(var(--brand));
    box-shadow: 0 0 8px hsla(var(--brand), 0.4);
  }
</style>

<script lang="ts">
  import { getChatState } from '../stores/chat';

  let { currentRoute = 'library', collapsed = false } = $props();

  const chat = getChatState();

  function navigate(route: string) {
    currentRoute = route;
  }
</script>

<aside
  class="sidebar"
  class:collapsed
  style="width: {collapsed ? '60px' : '260px'}"
>
  <!-- Logo -->
  <div class="logo">
    <span class="logo-icon">T</span>
    {#if !collapsed}
      <span class="logo-text">Terllama</span>
    {/if}
  </div>

  <!-- Nav -->
  <nav class="nav">
    <button
      class="nav-item"
      class:active={currentRoute === 'library'}
      onclick={() => navigate('library')}
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </svg>
      {#if !collapsed}<span>Library</span>{/if}
    </button>

    <button
      class="nav-item"
      class:active={currentRoute === 'chat'}
      onclick={() => navigate('chat')}
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
      </svg>
      {#if !collapsed}<span>Chat</span>{/if}
    </button>

    <button
      class="nav-item"
      class:active={currentRoute === 'settings'}
      onclick={() => navigate('settings')}
    >
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-2 2 2 2 0 01-2-2v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83 0 2 2 0 010-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 01-2-2 2 2 0 012-2h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 010-2.83 2 2 0 012.83 0l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 012-2 2 2 0 012 2v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 0 2 2 0 010 2.83l-.06.06A1.65 1.65 0 0019.32 9a1.65 1.65 0 001.51 1H21a2 2 0 012 2 2 2 0 01-2 2h-.09a1.65 1.65 0 00-1.51 1z" />
      </svg>
      {#if !collapsed}<span>Settings</span>{/if}
    </button>
  </nav>

  <!-- Chat Sessions (only in chat route) -->
  {#if currentRoute === 'chat' && !collapsed}
    <div class="sessions-header">
      <span>Chats</span>
      <button class="new-chat-btn" onclick={() => chat.newSession()}>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
        </svg>
      </button>
    </div>
    <div class="sessions-list">
      {#each chat.sessions as session}
        <button
          class="session-item"
          class:active={session.id === chat.activeSessionId}
          onclick={() => chat.switchSession(session.id)}
        >
          <span class="session-title">{session.title}</span>
        </button>
      {/each}
    </div>
  {/if}

  <!-- Collapse toggle -->
  <button class="collapse-btn" onclick={() => (collapsed = !collapsed)}>
    <svg
      width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
      style="transform: rotate({collapsed ? 180 : 0}deg)"
    >
      <polyline points="15 18 9 12 15 6" />
    </svg>
  </button>
</aside>

<style>
  .sidebar {
    height: 100vh;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    transition: width 0.2s ease;
    overflow: hidden;
    flex-shrink: 0;
  }

  .logo {
    padding: 20px 16px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 1px solid var(--border);
  }

  .logo-icon {
    width: 32px;
    height: 32px;
    background: var(--accent-gradient);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 16px;
    flex-shrink: 0;
  }

  .logo-text {
    font-size: 18px;
    font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .nav {
    padding: 12px 8px;
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 14px;
    transition: all 0.15s;
  }

  .nav-item:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .nav-item.active {
    background: rgba(124, 58, 237, 0.15);
    color: var(--accent-hover);
  }

  .sessions-header {
    padding: 12px 16px 8px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .new-chat-btn {
    background: none;
    border: none;
    color: var(--text-secondary);
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
  }

  .new-chat-btn:hover {
    color: var(--accent-hover);
    background: var(--bg-tertiary);
  }

  .sessions-list {
    flex: 1;
    overflow-y: auto;
    padding: 0 8px;
  }

  .session-item {
    width: 100%;
    text-align: left;
    padding: 8px 12px;
    border-radius: var(--radius-sm);
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    font-size: 13px;
    transition: all 0.15s;
    margin-bottom: 2px;
  }

  .session-item:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .session-item.active {
    background: rgba(124, 58, 237, 0.1);
    color: var(--accent-hover);
  }

  .session-title {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
  }

  .collapse-btn {
    margin: 8px;
    padding: 8px;
    border: none;
    background: transparent;
    color: var(--text-secondary);
    cursor: pointer;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .collapse-btn:hover {
    background: var(--bg-tertiary);
    color: var(--text-primary);
  }

  .collapsed .logo-text,
  .collapsed .nav-item span,
  .collapsed .sessions-header,
  .collapsed .sessions-list,
  .collapsed .session-item span {
    display: none;
  }

  .collapsed .nav-item {
    justify-content: center;
    padding: 10px;
  }

  .collapsed .logo {
    justify-content: center;
  }
</style>

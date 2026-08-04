<script lang="ts">
  import { getChatState } from '../stores/chat.svelte';
  import { getModelsState } from '../stores/models.svelte';

  let { currentRoute = $bindable('library') } = $props();

  const chat = getChatState();
  const models = getModelsState();

  const navItems = [
    { id: 'chat', label: 'Chat', icon: 'chat' },
    { id: 'library', label: 'Models', icon: 'library' },
    { id: 'convert', label: 'Convert', icon: 'convert' },
    { id: 'settings', label: 'Settings', icon: 'settings' },
  ];

  // ── Progress helpers ────────────────────────────────────────────────
  const RING_R = 13;
  const RING_C = 2 * Math.PI * RING_R;

  function ringOffset(pct: number): number {
    const p = Math.max(0, Math.min(100, pct || 0));
    return RING_C * (1 - p / 100);
  }

  function downloadPct(): number {
    const p = models.downloadProgress;
    if (!p || p.total === 0) return 0;
    return Math.min(100, Math.round((p.downloaded / p.total) * 100));
  }

  function formatMb(v: number): string {
    if (v >= 1024) return (v / 1024).toFixed(1) + ' GB';
    return Math.round(v) + ' MB';
  }

  function formatSpeed(bps: number): string {
    if (!bps || bps <= 0) return '—';
    return (bps / 1e6).toFixed(1) + ' MB/s';
  }

  function formatElapsed(sec: number): string {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  }

  // Tick elapsed time while a conversion is running.
  let elapsed = $state(0);
  let tickId: ReturnType<typeof setInterval> | null = null;
  $effect(() => {
    if (models.isConverting) {
      elapsed = 0;
      tickId = setInterval(() => (elapsed += 1), 1000);
    } else if (tickId) {
      clearInterval(tickId);
      tickId = null;
    }
  });
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

    <!-- Download progress (icon with circular ring, real %/speed on hover) -->
    {#if models.isDownloading && models.downloadProgress}
      <div class="progress-item" role="status" aria-label="Downloading {models.downloadProgress.model_id}">
        <div class="ring-wrap">
          <svg width="34" height="34" viewBox="0 0 34 34" class="ring">
            <circle class="ring-track" cx="17" cy="17" r={RING_R} />
            <circle
              class="ring-fill"
              cx="17" cy="17" r={RING_R}
              stroke-dasharray={RING_C}
              stroke-dashoffset={ringOffset(downloadPct())}
            />
          </svg>
          <svg class="ring-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        </div>
        <div class="tooltip">
          <span class="tooltip-title">Downloading</span>
          <span class="tooltip-model">{models.downloadProgress.model_id}</span>
          <span class="tooltip-stats">
            {downloadPct()}% · {formatMb(models.downloadProgress.downloaded)} / {formatMb(models.downloadProgress.total)} · {formatSpeed(models.downloadProgress.speed)}
          </span>
        </div>
      </div>
    {/if}

    <!-- Conversion progress (icon with circular ring, real %/elapsed on hover) -->
    {#if models.isConverting}
      <div class="progress-item" role="status" aria-label="Converting {models.convertProgress?.model}">
        <div class="ring-wrap">
          <svg width="34" height="34" viewBox="0 0 34 34" class="ring">
            <circle class="ring-track" cx="17" cy="17" r={RING_R} />
            <circle
              class="ring-fill"
              cx="17" cy="17" r={RING_R}
              stroke-dasharray={RING_C}
              stroke-dashoffset={ringOffset(models.convertProgress?.pct || 0)}
            />
          </svg>
          <svg class="ring-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="23 4 23 10 17 10" />
            <polyline points="1 20 1 14 7 14" />
            <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15" />
          </svg>
        </div>
        <div class="tooltip">
          <span class="tooltip-title">Converting</span>
          <span class="tooltip-model">{models.convertProgress?.model || 'model'}</span>
          <span class="tooltip-stats">
            {models.convertProgress?.pct || 0}% · elapsed {formatElapsed(elapsed)}
          </span>
        </div>
      </div>
    {/if}
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

  /* ── Progress icon with circular ring + hover tooltip ── */
  .progress-item {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 44px;
    margin-top: 4px;
    cursor: default;
  }

  .ring-wrap {
    position: relative;
    width: 34px;
    height: 34px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .ring {
    position: absolute;
    inset: 0;
    transform: rotate(-90deg);
  }

  .ring-track {
    fill: none;
    stroke: hsla(var(--surface-tertiary), 0.9);
    stroke-width: 3;
  }

  .ring-fill {
    fill: none;
    stroke: hsl(var(--brand));
    stroke-width: 3;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.3s ease;
    filter: drop-shadow(0 0 4px hsla(var(--brand), 0.5));
  }

  .ring-icon {
    color: hsl(var(--brand-hover));
  }

  .tooltip {
    position: absolute;
    left: 48px;
    top: 50%;
    transform: translateY(-50%);
    min-width: 210px;
    max-width: 260px;
    padding: 10px 12px;
    background: hsl(var(--surface));
    border: 1px solid hsla(var(--border), 0.7);
    border-radius: 8px;
    box-shadow: var(--shadow);
    display: flex;
    flex-direction: column;
    gap: 2px;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.15s ease;
    z-index: 50;
  }

  .progress-item:hover .tooltip {
    opacity: 1;
  }

  .tooltip-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: hsl(var(--brand-hover));
  }

  .tooltip-model {
    font-size: 12px;
    font-weight: 600;
    color: hsl(var(--content));
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .tooltip-stats {
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    color: hsl(var(--content-muted));
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

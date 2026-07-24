<script lang="ts">
  import type { Message } from '../../types';

  let { message }: { message: Message } = $props();

  function renderMarkdown(content: string): string {
    let html = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>');
    html = html.replace(/\n/g, '<br/>');
    return html;
  }
</script>

<div class="message {message.role}">
  <div class="avatar">
    {#if message.role === 'user'}
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    {:else}
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
        <polyline points="17 6 23 6 23 12" />
      </svg>
    {/if}
  </div>

  <div class="bubble">
    <div class="role-label">{message.role === 'user' ? 'You' : 'Terllama'}</div>
    <div class="text">
      {#if message.role === 'assistant'}
        <div>{@html renderMarkdown(message.content)}</div>
      {:else}
        <div>{message.content}</div>
      {/if}
    </div>
  </div>
</div>

<style>
  .message {
    display: flex;
    gap: 12px;
    padding: 16px 24px;
    max-width: 100%;
  }

  .message.user {
    background: transparent;
  }

  .message.assistant {
    background: hsla(var(--surface-secondary), 0.5);
    border-bottom: 1px solid hsla(var(--border), 0.3);
    border-left: 3px solid hsl(var(--gpu));
  }

  .message.user {
    border-left: 3px solid hsl(var(--brand));
  }

  .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 2px;
  }

  .user .avatar {
    background: hsla(var(--surface-tertiary), 0.8);
    color: hsl(var(--content-muted));
  }

  .assistant .avatar {
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: white;
  }

  .bubble {
    flex: 1;
    min-width: 0;
  }

  .role-label {
    font-size: 11px;
    font-weight: 600;
    color: hsl(var(--content-muted));
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .text {
    font-size: 14px;
    line-height: 1.6;
    color: hsl(var(--content));
    word-wrap: break-word;
  }

  .text :global(pre) {
    margin: 12px 0;
    font-size: 13px;
    line-height: 1.5;
  }

  .text :global(code) {
    font-size: 13px;
  }

  .text :global(br) {
    content: '';
    display: block;
    margin: 4px 0;
  }
</style>

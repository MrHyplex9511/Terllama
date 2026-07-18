<script lang="ts">
  import type { Message } from '../../types';

  let { message }: { message: Message } = $props();

  function renderMarkdown(content: string): string {
    // Bold
    let html = content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code class="lang-$1">$2</code></pre>');
    // Line breaks
    html = html.replace(/\n/g, '<br/>');
    return html;
  }
</script>

<div class="message {message.role}">
  <div class="avatar">
    {#if message.role === 'user'}
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M20 21v-2a4 4 0 00-4-4H8a4 4 0 00-4 4v2" />
        <circle cx="12" cy="7" r="4" />
      </svg>
    {:else}
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
        <polyline points="17 6 23 6 23 12" />
      </svg>
    {/if}
  </div>

  <div class="content">
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
    padding: 16px 20px;
    max-width: 100%;
  }

  .message.user {
    background: transparent;
  }

  .message.assistant {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
  }

  .avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
  }

  .user .avatar {
    background: var(--bg-tertiary);
    color: var(--text-secondary);
  }

  .assistant .avatar {
    background: var(--accent-gradient);
    color: white;
  }

  .content {
    flex: 1;
    min-width: 0;
  }

  .role-label {
    font-size: 12px;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .text {
    font-size: 14px;
    line-height: 1.6;
    color: var(--text-primary);
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

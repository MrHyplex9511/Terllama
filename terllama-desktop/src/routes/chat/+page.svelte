<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { getChatState } from '../../lib/stores/chat';
  import { getModelsState } from '../../lib/stores/models';
  import { getSettingsState } from '../../lib/stores/settings';
  import ChatMessage from '../../lib/components/ChatMessage.svelte';

  const chat = getChatState();
  const models = getModelsState();
  const settings = getSettingsState();

  let inputText = $state('');
  let inputRef: HTMLTextAreaElement | undefined = $state();
  let messagesEndRef: HTMLDivElement | undefined = $state();

  // Create first session if none exist
  $effect(() => {
    if (chat.sessions.length === 0) {
      chat.newSession();
    }
  });

  // Auto-scroll to bottom
  $effect(() => {
    if (chat.messages.length > 0 || chat.currentResponse) {
      requestAnimationFrame(() => {
        messagesEndRef?.scrollIntoView({ behavior: 'smooth' });
      });
    }
  });

  // Auto-grow textarea
  $effect(() => {
    if (inputRef) {
      inputRef.style.height = 'auto';
      inputRef.style.height = Math.min(inputRef.scrollHeight, 200) + 'px';
    }
  });

  async function sendMessage() {
    const text = inputText.trim();
    if (!text || chat.isGenerating) return;

    chat.addMessage({ role: 'user', content: text });
    inputText = '';
    chat.setIsGenerating(true);
    chat.setCurrentResponse('');

    // Get all messages for context
    const messages = chat.messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
      // Listen for streaming events
      const unlistenToken = await listen<string>('chat-token', (event) => {
        try {
          const data = JSON.parse(event.payload);
          if (data.choices?.[0]?.delta?.content) {
            chat.setCurrentResponse(chat.currentResponse + data.choices[0].delta.content);
          }
        } catch {
          chat.setCurrentResponse(chat.currentResponse + event.payload);
        }
      });

      const unlistenDone = await listen('chat-done', () => {
        chat.addMessage({ role: 'assistant', content: chat.currentResponse });
        chat.setCurrentResponse('');
        chat.setIsGenerating(false);
        unlistenToken();
        unlistenDone();
      });

      const unlistenError = await listen<string>('chat-error', (event) => {
        chat.setCurrentResponse('Error: ' + event.payload);
        chat.setIsGenerating(false);
        unlistenToken();
        unlistenDone();
        unlistenError();
      });

      await invoke('stream_chat', {
        messages,
        port: settings.settings.port,
      });
    } catch (e) {
      chat.setCurrentResponse('Failed to send message: ' + String(e));
      chat.setIsGenerating(false);
    }
  }

  function handleKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  }

  function stopGeneration() {
    chat.setIsGenerating(false);
    if (chat.currentResponse) {
      chat.addMessage({ role: 'assistant', content: chat.currentResponse });
    }
    chat.setCurrentResponse('');
  }
</script>

<div class="chat-page">
  <!-- Header -->
  <div class="chat-header">
    <div class="header-left">
      <span class="model-name">
        {models.activeModel || 'No model loaded'}
      </span>
      <span class="mode-badge ternary">Ternary</span>
    </div>
    <div class="header-right">
      {#if chat.tokensPerSecond > 0}
        <span class="tps">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
            <polyline points="17 6 23 6 23 12" />
          </svg>
          {chat.tokensPerSecond.toFixed(1)} t/s
        </span>
      {/if}
    </div>
  </div>

  <!-- Messages -->
  <div class="messages-area">
    {#if chat.messages.length === 0 && !chat.isGenerating}
      <div class="empty-state">
        <div class="empty-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
          </svg>
        </div>
        <h2>Start a conversation</h2>
        <p>Type a message below to begin chatting with your model.</p>
      </div>
    {:else}
      {#each chat.messages as msg}
        <ChatMessage message={msg} />
      {/each}

      {#if chat.isGenerating && chat.currentResponse}
        <ChatMessage message={{ role: 'assistant', content: chat.currentResponse }} />
      {/if}

      {#if chat.isGenerating && !chat.currentResponse}
        <div class="loading-dots">
          <span class="dot"></span>
          <span class="dot"></span>
          <span class="dot"></span>
        </div>
      {/if}
    {/if}
    <div bind:this={messagesEndRef} />
  </div>

  <!-- Input -->
  <div class="input-area">
    <div class="input-wrapper">
      <textarea
        bind:this={inputRef}
        bind:value={inputText}
        placeholder="Type a message... (Shift+Enter for newline)"
        onkeydown={handleKeydown}
        disabled={chat.isGenerating}
        rows="1"
      ></textarea>

      {#if chat.isGenerating}
        <button class="stop-btn" onclick={stopGeneration}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
      {:else}
        <button class="send-btn" onclick={sendMessage} disabled={!inputText.trim()}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      {/if}
    </div>
    <div class="input-footer">
      <span class="token-count">{chat.messages.length} messages</span>
    </div>
  </div>
</div>

<style>
  .chat-page {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .chat-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--bg-secondary);
  }

  .header-left {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .model-name {
    font-size: 14px;
    font-weight: 600;
  }

  .mode-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 4px;
    text-transform: uppercase;
  }

  .mode-badge.ternary {
    background: rgba(124, 58, 237, 0.15);
    color: var(--accent-hover);
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .tps {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    color: var(--success);
  }

  .messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 8px 0;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: var(--text-secondary);
  }

  .empty-icon {
    opacity: 0.4;
  }

  .empty-state h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
  }

  .empty-state p {
    margin: 0;
    font-size: 14px;
  }

  .loading-dots {
    display: flex;
    gap: 6px;
    padding: 20px 24px;
    align-items: center;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-secondary);
    animation: bounce 1.4s infinite ease-in-out;
  }

  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
  }

  .input-area {
    border-top: 1px solid var(--border);
    padding: 16px 20px;
    background: var(--bg-secondary);
  }

  .input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 8px 12px;
  }

  .input-wrapper:focus-within {
    border-color: var(--accent);
  }

  textarea {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 14px;
    resize: none;
    font-family: inherit;
    line-height: 1.5;
    max-height: 200px;
    outline: none;
  }

  textarea::placeholder {
    color: var(--text-secondary);
  }

  textarea:disabled {
    opacity: 0.6;
  }

  .send-btn,
  .stop-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    flex-shrink: 0;
    transition: all 0.15s;
  }

  .send-btn {
    background: var(--accent-gradient);
    color: white;
  }

  .send-btn:hover:not(:disabled) {
    opacity: 0.9;
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .stop-btn {
    background: var(--danger);
    color: white;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  .input-footer {
    display: flex;
    justify-content: flex-end;
    padding: 6px 4px 0;
  }

  .token-count {
    font-size: 11px;
    color: var(--text-secondary);
  }
</style>

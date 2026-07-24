<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { getChatState } from '../../lib/stores/chat.svelte';
  import { getModelsState } from '../../lib/stores/models.svelte';
  import { getSettingsState } from '../../lib/stores/settings.svelte';
  import ChatMessage from '../../lib/components/ChatMessage.svelte';
  import FadeContent from '../../lib/components/ui/FadeContent.svelte';
  import Counter from '../../lib/components/ui/Counter.svelte';
  import LiquidEther from '../../lib/components/bg/LiquidEther.svelte';

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

    const messages = chat.messages.map((m) => ({
      role: m.role,
      content: m.content,
    }));

    try {
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
  <!-- Messages -->
  <div class="messages-area">
    {#if chat.messages.length === 0 && !chat.isGenerating}
      <div class="empty-state">
        <LiquidEther colors={['#7C3AED', '#3B82F6', '#EC4899']} speed={15} blur={60} />
        <FadeContent duration={0.6} direction="up">
          <div class="empty-icon">
            <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" />
            </svg>
          </div>
          <h2>Start a conversation</h2>
          <p>Type a message below to begin chatting with your model.</p>
          {#if !models.activeModel}
            <p class="hint">Tip: Load a model from the Library first</p>
          {/if}
        </FadeContent>
      </div>
    {:else}
      <div class="messages-scroll">
        {#each chat.messages as msg, i}
          <FadeContent duration={0.3} delay={i === chat.messages.length - 1 ? 0 : 0}>
            <ChatMessage message={msg} />
          </FadeContent>
        {/each}

        {#if chat.isGenerating && chat.currentResponse}
          <FadeContent duration={0.3}>
            <ChatMessage message={{ role: 'assistant', content: chat.currentResponse }} />
          </FadeContent>
        {/if}

        {#if chat.isGenerating && !chat.currentResponse}
          <div class="loading-dots">
            <span class="dot"></span>
            <span class="dot"></span>
            <span class="dot"></span>
          </div>
        {/if}
      </div>
    {/if}
    <div bind:this={messagesEndRef}></div>
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
        <button class="stop-btn" onclick={stopGeneration} aria-label="Stop generation">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
      {:else}
        <button class="send-btn" onclick={sendMessage} disabled={!inputText.trim()} aria-label="Send message">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      {/if}
    </div>
    <div class="input-footer">
      <span class="model-badge">
        {models.activeModel || 'No model loaded'}
        {#if models.activeModel}
          <span class="mode-indicator">Ternary</span>
        {/if}
      </span>
      <Counter value={chat.messages.length} duration={300} suffix=" messages" class="msg-count" />
      {#if chat.tokensPerSecond > 0}
        <Counter value={chat.tokensPerSecond} duration={500} decimals={1} suffix=" t/s" class="tps" />
      {/if}
    </div>
  </div>
</div>

<style>
  .chat-page {
    display: flex;
    flex-direction: column;
    height: 100%;
  }

  .messages-area {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }

  .messages-scroll {
    padding: 8px 0;
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: 12px;
    color: hsl(var(--content-muted));
    padding: 40px;
    position: relative;
    overflow: hidden;
  }

  .empty-state > *:not(.liquid-ether) {
    position: relative;
    z-index: 1;
  }

  .empty-icon {
    opacity: 0.3;
  }

  .empty-state h2 {
    margin: 0;
    font-size: 20px;
    font-weight: 600;
    color: hsl(var(--content));
  }

  .empty-state p {
    margin: 0;
    font-size: 14px;
  }

  .empty-state .hint {
    font-size: 12px;
    color: hsl(var(--brand-hover));
    margin-top: 4px;
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
    background: hsl(var(--content-muted));
    animation: bounce 1.4s infinite ease-in-out;
  }

  .dot:nth-child(2) { animation-delay: 0.2s; }
  .dot:nth-child(3) { animation-delay: 0.4s; }

  @keyframes bounce {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
  }

  .input-area {
    border-top: 1px solid hsl(var(--border));
    padding: 16px 24px;
    background: hsla(var(--surface-secondary), 0.4);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
  }

  .input-wrapper {
    display: flex;
    align-items: flex-end;
    gap: 10px;
    background: hsla(var(--surface-tertiary), 0.6);
    border: 1px solid hsla(var(--border), 0.6);
    border-radius: var(--radius);
    padding: 8px 12px;
    transition: border-color 0.15s;
  }

  .input-wrapper:focus-within {
    border-color: hsl(var(--brand));
    box-shadow: 0 0 12px hsla(var(--brand), 0.1);
  }

  textarea {
    flex: 1;
    background: transparent;
    border: none;
    color: hsl(var(--content));
    font-size: 14px;
    resize: none;
    font-family: inherit;
    line-height: 1.5;
    max-height: 200px;
    outline: none;
  }

  textarea::placeholder {
    color: hsl(var(--content-muted));
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
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: white;
  }

  .send-btn:hover:not(:disabled) {
    opacity: 0.9;
    box-shadow: 0 0 12px hsla(var(--brand), 0.3);
  }

  .send-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }

  .stop-btn {
    background: hsl(var(--danger));
    color: white;
    animation: pulse 1.5s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
  }

  .input-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 4px 0;
    gap: 12px;
  }

  .model-badge {
    font-size: 11px;
    color: hsl(var(--content-muted));
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .mode-indicator {
    font-size: 10px;
    font-weight: 600;
    padding: 1px 6px;
    border-radius: 3px;
    background: hsla(var(--brand), 0.15);
    color: hsl(var(--brand-hover));
    text-transform: uppercase;
  }

  :global(.msg-count) {
    font-size: 11px;
    color: hsl(var(--content-muted));
  }

  :global(.tps) {
    font-size: 11px;
    color: hsl(var(--success));
    font-weight: 500;
  }
</style>

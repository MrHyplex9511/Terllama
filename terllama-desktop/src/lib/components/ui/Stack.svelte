<script lang="ts">
  import { onMount } from 'svelte';
  import { flip } from 'svelte/animate';
  import { fly } from 'svelte/transition';

  let {
    children,
    class: className = '',
    gap = 8,
    limit = 5,
    randomRotation = false,
  } = $props();

  let items: Array<{ id: number; content: string }> = $state([]);
  let container: HTMLDivElement | undefined = $state();

  // Expose a push method for consumers
  let addItem = $state<(content: string) => void>();

  $effect(() => {
    addItem = (content: string) => {
      items = [{ id: Date.now(), content }, ...items].slice(0, limit);
    };
  });

  function removeItem(id: number) {
    items = items.filter(i => i.id !== id);
  }
</script>

<div bind:this={container} class="stack {className}" style="--gap: {gap}px;">
  {#each items as item (item.id)}
    {#key item.id}
      <div
        class="stack-item"
        animate:flip={{ duration: 300 }}
        transition:fly={{ y: 20, duration: 200 }}
        style="--rotation: {randomRotation ? (Math.random() - 0.5) * 6 : 0}deg;"
        onclick={() => removeItem(item.id)}
        role="button"
        tabindex="0"
      >
        {item.content}
      </div>
    {/key}
  {/each}

  {#if items.length === 0}
    <div class="stack-empty">
      <slot name="empty">
        <p class="empty-text">No items</p>
      </slot>
    </div>
  {/if}
</div>

<style>
  .stack {
    display: flex;
    flex-direction: column;
    gap: var(--gap);
    position: relative;
  }

  .stack-item {
    padding: 16px;
    background: hsla(var(--surface-secondary), 0.6);
    backdrop-filter: blur(12px);
    border: 1px solid hsla(var(--border), 0.5);
    border-radius: var(--radius-sm);
    cursor: pointer;
    transition: all 0.2s;
    transform: rotate(var(--rotation));
  }

  .stack-item:hover {
    border-color: hsl(var(--brand));
    box-shadow: var(--shadow-glow);
  }

  .stack-empty {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 40px;
    color: hsl(var(--content-muted));
  }

  .empty-text {
    font-size: 14px;
    opacity: 0.6;
  }
</style>

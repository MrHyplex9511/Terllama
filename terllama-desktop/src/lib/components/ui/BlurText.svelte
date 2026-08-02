<script lang="ts">
  import { onMount } from 'svelte';

  let {
    text = '',
    class: className = '',
    delay = 50,
    animateBy = 'words' as 'words' | 'letters',
    direction = 'top' as 'top' | 'bottom',
    threshold = 0.1,
    duration = 0.5,
  } = $props();

  let el: HTMLElement | undefined = $state();
  let mounted = $state(false);

  $effect(() => {
    if (!el || !mounted) return;
    let cancelled = false;
    import('gsap').then((gsap) => {
      if (cancelled || !el) return;
      const spans = el.querySelectorAll('.blur-char');
      const dir = direction === 'top' ? -30 : 30;
      gsap.default.set(spans, { filter: 'blur(10px)', opacity: 0, y: dir });
      gsap.default.to(spans, {
        filter: 'blur(0px)', opacity: 1, y: 0,
        duration, ease: 'power3.out',
        stagger: { each: delay / 1000, from: 'start' },
      });
    });
    return () => { cancelled = true; };
  });

  onMount(() => { mounted = true; });

  function renderText(): Array<{ text: string; isSpace: boolean }> {
    if (animateBy === 'letters') {
      return text.split('').map(c => ({ text: c === ' ' ? '\u00A0' : c, isSpace: c === ' ' }));
    }
    const parts = text.split(/(\s+)/);
    return parts.map(p => ({ text: p, isSpace: /^\s+$/.test(p) }));
  }
</script>

<p bind:this={el} class={className}>
  {#if mounted}
    {#each renderText() as part}
      <span class="blur-char" class:space={part.isSpace}>{part.text}</span>
    {/each}
  {:else}
    {text}
  {/if}
</p>

<style>
  .blur-char:not(.space) {
    display: inline-block;
  }
</style>

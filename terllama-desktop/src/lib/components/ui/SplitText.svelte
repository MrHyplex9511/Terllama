<script lang="ts">
  import { onMount } from 'svelte';

  let {
    text = '',
    class: className = '',
    delay = 30,
    animationFrom = { opacity: 0, transform: 'translateY(20px)' },
    animationTo = { opacity: 1, transform: 'translateY(0)' },
    duration = 0.4,
    threshold = 0.1,
    as = 'p' as 'p' | 'h1' | 'h2' | 'h3' | 'span' | 'div',
  } = $props();

  let el: HTMLElement | undefined = $state();
  let mounted = $state(false);

  $effect(() => {
    if (!el || !mounted) return;
    let cancelled = false;
    import('gsap').then((gsap) => {
      if (cancelled || !el) return;
      const chars = el.querySelectorAll('.split-char');
      gsap.default.set(chars, animationFrom);
      gsap.default.to(chars, {
        ...animationTo,
        duration,
        ease: 'power3.out',
        stagger: delay / 1000,
      });
    });
    return () => { cancelled = true; };
  });

  onMount(() => { mounted = true; });
</script>

<svelte:element this={as} bind:this={el} class={className}>
  {#if mounted}
    {#each text.split('') as char, i}
      <span class="split-char" style="display: inline-block;">{char === ' ' ? '\u00A0' : char}</span>
    {/each}
  {:else}
    {text}
  {/if}
</svelte:element>

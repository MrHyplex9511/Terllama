<script lang="ts">
  let {
    children,
    class: className = '',
    direction = 'up' as 'up' | 'down' | 'left' | 'right',
    duration = 0.5,
    delay = 0,
    threshold = 0.1,
    stagger = 0,
    margin = '0px',
  } = $props();

  let el: HTMLElement | undefined = $state();
  let ctx: gsap.Context | undefined = $state();

  $effect(() => {
    if (!el) return;
    let cancelled = false;
    import('gsap').then((gsap) => {
      if (cancelled || !el) return;
      if (stagger > 0) {
        const items = el.querySelectorAll('.fade-item');
        if (items.length > 0) {
          gsap.default.set(items, { autoAlpha: 0, y: direction === 'up' ? 20 : direction === 'down' ? -20 : 0, x: direction === 'left' ? 20 : direction === 'right' ? -20 : 0 });
          const tl = gsap.default.timeline({ delay });
          tl.to(items, { autoAlpha: 1, y: 0, x: 0, duration, ease: 'power2.out', stagger });
          return;
        }
      }
      gsap.default.set(el, { autoAlpha: 0, y: direction === 'up' ? 20 : direction === 'down' ? -20 : 0, x: direction === 'left' ? 20 : direction === 'right' ? -20 : 0 });
      gsap.default.to(el, { autoAlpha: 1, y: 0, x: 0, duration, delay, ease: 'power2.out' });
    });
    return () => { cancelled = true; ctx?.revert(); };
  });
</script>

<div bind:this={el} class={className}>
  {@render children()}
</div>

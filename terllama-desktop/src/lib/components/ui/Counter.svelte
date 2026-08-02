<script lang="ts">
  import { tweened } from 'svelte/motion';
  import { cubicOut } from 'svelte/easing';
  import { onMount } from 'svelte';

  let {
    value = 0,
    class: className = '',
    duration = 500,
    decimals = 0,
    prefix = '',
    suffix = '',
  } = $props();

  let displayVal = $state('0');
  let el: HTMLElement | undefined = $state();

  $effect(() => {
    let cancelled = false;
    import('gsap').then((gsap) => {
      if (cancelled) return;
      if (!el) return;
      const obj = { val: 0 };
      gsap.default.to(obj, {
        val: value,
        duration: duration / 1000,
        ease: 'power2.out',
        onUpdate: () => {
          el!.textContent = prefix + obj.val.toFixed(decimals) + suffix;
        },
      });
    });
    return () => { cancelled = true; };
  });

  $effect(() => {
    displayVal = prefix + Number(value).toFixed(decimals) + suffix;
  });
</script>

<span bind:this={el} class={className}>{displayVal}</span>

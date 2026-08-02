<script lang="ts">
  import { onMount } from 'svelte';

  let {
    text = '',
    class: className = '',
    speed = 0.05,
    duration = 1.2,
    chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()',
  } = $props();

  let el: HTMLElement | undefined = $state();
  let displayed = $state(text);
  let mounted = $state(false);

  $effect(() => {
    if (!mounted) return;
    let cancelled = false;

    import('gsap').then((gsap) => {
      if (cancelled) return;

      const obj = { progress: 0 };
      const targetLen = text.length;

      gsap.default.to(obj, {
        progress: 1,
        duration,
        ease: 'power2.out',
        onUpdate: () => {
          if (cancelled) return;
          const prog = obj.progress;
          const done = Math.floor(prog * targetLen);
          let result = '';
          for (let i = 0; i < targetLen; i++) {
            if (i < done) {
              result += text[i];
            } else {
              const randChar = chars[Math.floor(Math.random() * chars.length)];
              if (text[i] === ' ') result += ' ';
              else result += randChar;
            }
          }
          displayed = result;
        },
        onComplete: () => { displayed = text; },
      });
    });

    return () => { cancelled = true; displayed = text; };
  });

  onMount(() => { mounted = true; });
</script>

<span bind:this={el} class={className}>{displayed}</span>

<script lang="ts">
  import { onMount } from 'svelte';

  let {
    class: className = '',
    color = 'hsla(var(--brand), 0.12)',
    size = 200,
    blur = 80,
    followSpeed = 0.15,
  } = $props();

  let canvas: HTMLCanvasElement | undefined = $state();
  let ctx: CanvasRenderingContext2D | null = $state(null);

  let mouse = $state({ x: -999, y: -999 });
  let pos = $state({ x: -999, y: -999 });
  let rafId = $state(0);

  function handleMouseMove(e: MouseEvent) {
    mouse = { x: e.clientX, y: e.clientY };
  }

  function animate() {
    if (!canvas || !ctx) return;
    pos = {
      x: pos.x + (mouse.x - pos.x) * followSpeed,
      y: pos.y + (mouse.y - pos.y) * followSpeed,
    };

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, size);
    gradient.addColorStop(0, color);
    gradient.addColorStop(1, 'transparent');

    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    rafId = requestAnimationFrame(animate);
  }

  function resize() {
    if (!canvas) return;
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  onMount(() => {
    if (!canvas) return;
    ctx = canvas.getContext('2d');
    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('mousemove', handleMouseMove);
    pos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    rafId = requestAnimationFrame(animate);
    return () => {
      cancelAnimationFrame(rafId);
      window.removeEventListener('resize', resize);
      window.removeEventListener('mousemove', handleMouseMove);
    };
  });
</script>

<canvas
  bind:this={canvas}
  class="blob-cursor {className}"
  style="--blur: {blur}px;"
></canvas>

<style>
  canvas {
    position: fixed;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 9999;
    filter: blur(var(--blur));
    -webkit-filter: blur(var(--blur));
  }
</style>

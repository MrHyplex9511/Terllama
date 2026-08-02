<script lang="ts">
  let {
    class: className = '',
    colors = ['#7C3AED', '#3B82F6', '#EC4899', '#8B5CF6'],
    speed = 10,
    blur = 40,
  } = $props();
</script>

<div class="liquid-ether {className}" style="--blur: {blur}px; --speed: {speed}s;">
  <div class="liquid-bg">
    {#each colors as color, i}
      <div class="blob" style="--color: {color}; --i: {i};"></div>
    {/each}
  </div>
</div>

<style>
  .liquid-ether {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    z-index: 0;
  }

  .liquid-bg {
    position: absolute;
    inset: -50%;
    filter: blur(var(--blur));
    animation: morph var(--speed) ease-in-out infinite alternate;
  }

  .blob {
    position: absolute;
    width: 40%;
    height: 40%;
    border-radius: 50%;
    background: var(--color);
    opacity: 0.25;
    animation: drift var(--speed) ease-in-out infinite alternate;
    animation-delay: calc(var(--i) * -2s);
    top: calc(var(--i) * 20%);
    left: calc(var(--i) * 15%);
  }

  @keyframes drift {
    0% { transform: translate(0, 0) scale(1) rotate(0deg); }
    33% { transform: translate(10%, 10%) scale(1.2) rotate(120deg); }
    66% { transform: translate(-10%, 5%) scale(0.8) rotate(240deg); }
    100% { transform: translate(5%, -10%) scale(1.1) rotate(360deg); }
  }

  @keyframes morph {
    0% { border-radius: 60% 40% 30% 70% / 60% 30% 70% 40%; }
    50% { border-radius: 30% 60% 70% 40% / 50% 60% 30% 60%; }
    100% { border-radius: 40% 60% 30% 70% / 60% 40% 70% 30%; }
  }
</style>

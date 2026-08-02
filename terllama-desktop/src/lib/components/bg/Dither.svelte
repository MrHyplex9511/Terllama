<script lang="ts">
  let {
    class: className = '',
    opacity = 0.03,
    speed = 0.5,
  } = $props();
</script>

<div class="dither-overlay {className}" style="--opacity: {opacity}; --speed: {speed}s;"></div>

<style>
  .dither-overlay {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 1;
    opacity: var(--opacity);
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 256px 256px;
    animation: dither-shift var(--speed) steps(4) infinite;
    mix-blend-mode: overlay;
  }

  @keyframes dither-shift {
    0% { background-position: 0 0; }
    25% { background-position: 2px -2px; }
    50% { background-position: -1px 1px; }
    75% { background-position: 1px -1px; }
    100% { background-position: 0 0; }
  }
</style>

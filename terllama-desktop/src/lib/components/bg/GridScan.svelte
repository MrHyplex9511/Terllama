<script lang="ts">
  let {
    class: className = '',
    color = 'hsla(var(--brand), 0.08)',
    lineColor = 'hsla(var(--brand), 0.15)',
    speed = 4,
  } = $props();
</script>

<div class="grid-scan {className}" style="--color: {color}; --line-color: {lineColor}; --speed: {speed}s;">
  <div class="scan-line"></div>
</div>

<style>
  .grid-scan {
    position: absolute;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image:
      linear-gradient(var(--line-color) 1px, transparent 1px),
      linear-gradient(90deg, var(--line-color) 1px, transparent 1px);
    background-size: 60px 60px;
  }

  .scan-line {
    position: absolute;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, hsl(var(--brand)), transparent);
    opacity: 0.5;
    animation: scan var(--speed) linear infinite;
    box-shadow: 0 0 20px hsl(var(--brand));
  }

  @keyframes scan {
    0% { top: -2px; }
    100% { top: 100%; }
  }
</style>

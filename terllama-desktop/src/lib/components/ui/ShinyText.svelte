<script lang="ts">
  let {
    text = '',
    class: className = '',
    baseColor = 'currentColor',
    shimmerColor = 'rgba(255,255,255,0.4)',
    speed = 3,
    disabled = false,
  } = $props();
</script>

{#if disabled}
  <span class={className}>{text}</span>
{:else}
  <span
    class={className}
    style="--shimmer-color: {shimmerColor}; --base-color: {baseColor}; --speed: {speed}s;"
  >{text}</span>
{/if}

<style>
  span {
    display: inline-block;
    background: linear-gradient(
      90deg,
      var(--base-color) 0%,
      var(--base-color) 40%,
      var(--shimmer-color) 50%,
      var(--base-color) 60%,
      var(--base-color) 100%
    );
    background-size: 200% 100%;
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer var(--speed) linear infinite;
  }

  @keyframes shimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }
</style>

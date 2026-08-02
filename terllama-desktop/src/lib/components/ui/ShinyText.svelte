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

<!--
  Intentionally rendered as solid colored text (no background-clip: text + 
  -webkit-text-fill-color: transparent trick). That technique renders the
  text fully invisible in the WebKitGTK webview used by Tauri on Linux
  (model names in the library were reported missing). Solid color keeps the
  text visible on every platform. Props are kept for API compatibility.
-->
<span class={className} style="--base-color: {baseColor};">{text}</span>

<style>
  span {
    display: inline-block;
    color: var(--base-color);
  }
</style>

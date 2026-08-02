<script lang="ts">
  import Layout from './routes/+layout.svelte';
  import Welcome from './lib/components/Welcome.svelte';

  let entered = $state(false);
  let fadeApp = $state(false);
</script>

{#if !entered}
  <Welcome
    onEnter={() => {
      fadeApp = true;
      // wait for the fade-in before showing layout
      setTimeout(() => {
        entered = true;
      }, 240);
    }}
  />
{/if}

<div class="app-shell" class:show={entered && fadeApp}>
  <Layout />
</div>

<style>
  .app-shell {
    opacity: 0;
    transition: opacity 0.3s ease-in;
  }
  .app-shell.show {
    opacity: 1;
  }
</style>

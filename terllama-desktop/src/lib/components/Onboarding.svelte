<script lang="ts">
  import BlurText from './ui/BlurText.svelte';
  import FadeContent from './ui/FadeContent.svelte';

  let { onComplete }: { onComplete?: () => void } = $props();

  let step = $state(1);

  const steps = [
    {
      title: 'Download a Model',
      desc: 'Browse the model library and pick a quantization. Ternary (1.58-bit) is recommended for fastest CPU inference with the Terllama engine.',
      icon: '📦',
    },
    {
      title: 'Start Chatting',
      desc: 'Once a model is downloaded, load it and start chatting. The chat interface supports streaming responses, markdown, and code blocks.',
      icon: '💬',
    },
    {
      title: 'Configure Settings',
      desc: 'Adjust port, auto-start, theme, and more in Settings. Toggle between Ternary and GGUF inference modes as support expands.',
      icon: '⚙️',
    },
  ];

  function next() {
    if (step < 3) step++;
    else finish();
  }

  function prev() {
    if (step > 1) step--;
  }

  function finish() {
    localStorage.setItem('terllama-onboarded', 'true');
    onComplete?.();
  }
</script>

<div class="overlay">
  <div class="card">
    <div class="step-indicator">
      {#each steps as _, i}
        <span class="dot" class:active={i + 1 === step} class:done={i + 1 < step}></span>
      {/each}
    </div>

    <div class="content">
      <FadeContent duration={0.4} delay={0.05} direction="up">
        <div class="icon">{steps[step - 1].icon}</div>
        <BlurText
          text={steps[step - 1].title}
          animateBy="words"
          direction="top"
          delay={40}
          duration={0.5}
          class="onboarding-title"
        />
        <p>{steps[step - 1].desc}</p>
      </FadeContent>
    </div>

    <div class="nav-buttons">
      <button class="btn secondary" onclick={finish}>Skip</button>
      <div class="right">
        {#if step > 1}
          <button class="btn secondary" onclick={prev}>Back</button>
        {/if}
        <button class="btn primary" onclick={next}>
          {step === 3 ? 'Get Started' : 'Next'}
        </button>
      </div>
    </div>
  </div>
</div>

<style>
  .overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.7);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 200;
    backdrop-filter: blur(8px);
  }

  .card {
    background: hsla(var(--surface-secondary), 0.9);
    border: 1px solid hsla(var(--border), 0.5);
    border-radius: 16px;
    padding: 40px;
    width: 440px;
    max-width: 90vw;
    box-shadow: 0 8px 40px rgba(0, 0, 0, 0.4);
    background-image: linear-gradient(135deg, hsla(var(--brand), 0.08), transparent 60%);
  }

  .step-indicator {
    display: flex;
    justify-content: center;
    gap: 8px;
    margin-bottom: 32px;
  }

  .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: hsl(var(--surface-tertiary));
    transition: all 0.3s;
  }

  .dot.active {
    background: hsl(var(--brand));
    width: 24px;
    border-radius: 4px;
  }

  .dot.done {
    background: hsl(var(--brand));
    opacity: 0.5;
  }

  .content {
    text-align: center;
    margin-bottom: 32px;
  }

  .icon {
    font-size: 48px;
    margin-bottom: 16px;
  }

  :global(.onboarding-title) {
    font-size: 22px;
    font-weight: 700;
    margin: 0 0 12px;
    color: hsl(var(--content));
  }

  p {
    font-size: 14px;
    color: hsl(var(--content-muted));
    line-height: 1.6;
    margin: 0;
  }

  .nav-buttons {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .right {
    display: flex;
    gap: 10px;
  }

  .btn {
    padding: 10px 20px;
    border: none;
    border-radius: var(--radius-sm);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.15s;
  }

  .btn.primary {
    background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
    color: white;
  }

  .btn.primary:hover {
    opacity: 0.9;
  }

  .btn.secondary {
    background: hsla(var(--surface-tertiary), 0.8);
    color: hsl(var(--content));
    border: 1px solid hsla(var(--border), 0.5);
  }

  .btn.secondary:hover {
    background: hsl(var(--border));
  }
</style>

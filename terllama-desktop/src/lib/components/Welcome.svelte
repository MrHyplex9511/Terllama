<script lang="ts">
  import { onMount } from 'svelte';
  import { animate, createTimeline, stagger, createDrawable, createSpring } from 'animejs';

  let { onEnter }: { onEnter?: (target: string) => void } = $props();

  // ── Refs ─────────────────────────────────────────────────────────────
  let root: HTMLElement | undefined = $state();
  let orbA: HTMLElement | undefined = $state();
  let orbB: HTMLElement | undefined = $state();
  let halo: SVGCircleElement | undefined = $state();
  let logoWrap: HTMLElement | undefined = $state();
  let tagline: HTMLElement | undefined = $state();
  let ctaRow: HTMLElement | undefined = $state();
  let hint: HTMLElement | undefined = $state();
  let primaryBtn: HTMLButtonElement | undefined = $state();
  let secondaryBtn: HTMLButtonElement | undefined = $state();

  let exiting = $state(false);

  const brand = 'Terllama';

  // ── Gear geometry ────────────────────────────────────────────────────
  // Three equal gears in an equilateral triangle. Because every gear is
  // meshed with BOTH others, the loop is mechanically locked — the gears
  // cannot rotate independently. We honor that: no free spinning, only a
  // torque strain-and-jam wobble after the teeth draw in.
  const teeth = 12;
  const outerR = 17;
  const innerR = 11.5;
  const halfTooth = Math.PI / teeth; // 15°: one tooth spans 2*halfTooth
  const side = 2 * outerR - 3;       // center distance (slight overlap so teeth interleave)
  const circumR = side / Math.sqrt(3);

  // Triangle positions (gear centers), phases offset half a tooth so the
  // teeth visually interlock at the contact points.
  const gears = [
    { cx: 48, cy: 48 - circumR, phase: 0 },
    { cx: 48 - side / 2, cy: 48 + circumR / 2, phase: halfTooth * 0.5 },
    { cx: 48 + side / 2, cy: 48 + circumR / 2, phase: -halfTooth * 0.5 },
  ];

  function gearPath(R: number, r: number, ph: number): string {
    const parts: string[] = [];
    const step = halfTooth;
    for (let i = 0; i < teeth; i++) {
      const a0 = ph + i * 2 * step;
      const a1 = a0 + step * 0.6; // tooth top end
      const a2 = a0 + step * 1.0; // root start
      const a3 = a0 + step * 1.6; // root end
      const a4 = a0 + 2 * step;   // next tooth start
      parts.push(`M ${(R * Math.cos(a0)).toFixed(2)} ${(R * Math.sin(a0)).toFixed(2)}`);
      parts.push(`A ${R} ${R} 0 0 1 ${(R * Math.cos(a1)).toFixed(2)} ${(R * Math.sin(a1)).toFixed(2)}`);
      parts.push(`L ${(r * Math.cos(a2)).toFixed(2)} ${(r * Math.sin(a2)).toFixed(2)}`);
      parts.push(`A ${r} ${r} 0 0 1 ${(r * Math.cos(a3)).toFixed(2)} ${(r * Math.sin(a3)).toFixed(2)}`);
      parts.push(`L ${(R * Math.cos(a4)).toFixed(2)} ${(R * Math.sin(a4)).toFixed(2)}`);
    }
    return parts.join(' ') + ' Z';
  }

  function handleEnter(target: string) {
    if (exiting) return;
    exiting = true;

    // Outro: collapse the whole splash
    createTimeline()
      .add(logoWrap!, { y: -24, opacity: 0, duration: 380, ease: 'inOutQuad' }, 0)
      .add(tagline!, { y: -16, opacity: 0, duration: 320, ease: 'inOutQuad' }, 40)
      .add(ctaRow!, { y: -12, opacity: 0, duration: 320, ease: 'inOutQuad' }, 80)
      .add(root!, { opacity: 0, scale: 0.98, duration: 380, ease: 'inOutQuad' }, 160)
      .call(() => {
        localStorage.setItem('terllama-onboarded', 'true');
        onEnter?.(target);
      }, 420);
  }

  onMount(() => {
    // ── Keyboard: Enter continues (matches the hint) ──────────────────
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Enter') handleEnter('/library');
    };
    window.addEventListener('keydown', onKey);

    // ── Ambient orbs: slow drifting glow ─────────────────────────────
    animate(orbA!, {
      x: ['-8%', '8%', '-8%'],
      y: ['-6%', '6%', '-6%'],
      scale: [1, 1.15, 1],
      opacity: [0.5, 0.75, 0.5],
      duration: 14000,
      ease: 'inOutSine',
      loop: true,
    });
    animate(orbB!, {
      x: ['10%', '-10%', '10%'],
      y: ['8%', '-6%', '8%'],
      scale: [1.1, 0.95, 1.1],
      opacity: [0.45, 0.65, 0.45],
      duration: 17000,
      ease: 'inOutSine',
      loop: true,
    });

    // ── Logo: draw each gear's teeth, staggered ─────────────────────
    // Query the DOM directly (bind:this arrays aren't populated yet in
    // Svelte 5 onMount) — same pattern as the brand letters below.
    const spins = Array.from(root!.querySelectorAll<SVGGElement>('.gear-spin'));
    const paths = spins.flatMap((g) =>
      Array.from(g.querySelectorAll<SVGPathElement>('path'))
    );
    animate(
      paths.map((p) => createDrawable(p)),
      {
        draw: '0 1',
        duration: 1000,
        ease: 'inOutExpo',
        delay: stagger(120),
      }
    );

    // ── Halo: soft pulsing glow behind the gears ─────────────────────
    animate(halo!, {
      opacity: [0.25, 0.6, 0.25],
      scale: [1, 1.08, 1],
      duration: 4200,
      ease: 'inOutSine',
      loop: true,
    });

    // ── Torque strain: locked gears try to turn, jam, settle ────────
    // Each gear strains ±1.6° against its mesh then locks back at rest.
    // Note: timeline .add() takes (targets, params, position) — not
    // pre-built animations. The inner <g> is rotated (CSS transform),
    // while the outer <g> keeps the SVG translate attribute — otherwise
    // the CSS transform would override the translate and the gear would
    // jump to the SVG origin.
    createTimeline()
      .add(
        spins,
        { rotate: [0, 1.6, -1.4, 1.1, -0.7, 0.4, 0], duration: 900, ease: 'inOutSine' },
        1100
      )
      .add(
        spins,
        { rotate: [0, -1.6, 1.4, -1.1, 0.7, -0.4, 0], duration: 900, ease: 'inOutSine' },
        2100
      );

    // ── Brand letters: blur → focus stagger ─────────────────────────
    const letters = root!.querySelectorAll<HTMLElement>('.brand-letter');
    animate(letters, {
      filter: ['blur(12px)', 'blur(0px)'],
      opacity: [0, 1],
      y: [18, 0],
      duration: 650,
      ease: 'outExpo',
      delay: stagger(55, { from: 'center' }),
    });

    // ── Tagline + CTAs ───────────────────────────────────────────────
    animate(tagline!, {
      opacity: [0, 1],
      y: [12, 0],
      duration: 600,
      ease: 'outExpo',
      delay: 650,
    });
    animate(ctaRow!, {
      opacity: [0, 1],
      y: [16, 0],
      duration: 600,
      ease: 'outExpo',
      delay: 800,
    });
    animate(hint!, {
      opacity: [0, 1],
      duration: 800,
      ease: 'outQuad',
      delay: 1400,
    });

    // ── Spring hover on CTAs ─────────────────────────────────────────
    const springX = createSpring({ stiffness: 260, damping: 18, mass: 0.5 });
    const springY = createSpring({ stiffness: 260, damping: 18, mass: 0.5 });
    [primaryBtn, secondaryBtn].forEach((btn) => {
      if (!btn) return;
      btn.addEventListener('mouseenter', () => {
        animate(btn, { scale: 1.04, translateY: -1, ease: springX, duration: 300 });
      });
      btn.addEventListener('mouseleave', () => {
        animate(btn, { scale: 1, translateY: 0, ease: springY, duration: 300 });
      });
    });

    return () => window.removeEventListener('keydown', onKey);
  });

</script>
<svelte:head>
  <style>
    .welcome-root {
      position: fixed;
      inset: 0;
      z-index: 300;
      display: flex;
      align-items: center;
      justify-content: center;
      background: radial-gradient(120% 120% at 50% 20%, #14141f 0%, #0a0a12 55%, #06060c 100%);
      overflow: hidden;
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      -webkit-user-select: none;
      user-select: none;
    }

    .welcome-root::after {
      content: '';
      position: absolute;
      inset: 0;
      pointer-events: none;
      background-image: linear-gradient(hsla(var(--border), 0.22) 1px, transparent 1px),
        linear-gradient(90deg, hsla(var(--border), 0.22) 1px, transparent 1px);
      background-size: 56px 56px;
      -webkit-mask-image: radial-gradient(60% 60% at 50% 50%, #000 30%, transparent 100%);
      mask-image: radial-gradient(60% 60% at 50% 50%, #000 30%, transparent 100%);
    }

    .orb {
      position: absolute;
      border-radius: 50%;
      filter: blur(90px);
      pointer-events: none;
    }
    .orb-a {
      width: 480px;
      height: 480px;
      left: 12%;
      top: 10%;
      background: hsla(var(--brand), 0.18);
    }
    .orb-b {
      width: 420px;
      height: 420px;
      right: 10%;
      bottom: 6%;
      background: hsla(var(--gpu), 0.12);
    }

    .welcome-inner {
      position: relative;
      display: flex;
      flex-direction: column;
      align-items: center;
      text-align: center;
      padding: 24px;
      max-width: 520px;
      z-index: 1;
    }

    .logo-mark {
      width: 96px;
      height: 96px;
      margin-bottom: 28px;
      display: flex;
      align-items: center;
      justify-content: center;
      cursor: pointer;
      background: none;
      border: none;
      padding: 0;
    }
    .logo-mark svg {
      width: 100%;
      height: 100%;
      overflow: visible;
    }
    .logo-mark .gear-path {
      fill: rgba(15, 15, 26, 0.55);
      stroke: hsl(var(--brand-hover));
      stroke-width: 1.6;
      stroke-linecap: round;
      stroke-linejoin: round;
      filter: drop-shadow(0 0 10px hsla(var(--brand), 0.45));
    }
    .logo-mark .axle {
      fill: hsl(var(--brand-hover));
    }

    .brand {
      display: flex;
      justify-content: center;
      gap: 2px;
      margin: 0;
      font-size: 52px;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: hsl(var(--content));
      line-height: 1;
    }
    .brand-letter {
      display: inline-block;
      opacity: 0;
    }
    .brand-letter.accent {
      background: linear-gradient(135deg, hsl(var(--brand-hover)), hsl(var(--brand)));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }

    .tagline {
      margin: 16px 0 0;
      font-size: 15px;
      color: hsl(var(--content-muted));
      opacity: 0;
      font-weight: 400;
      letter-spacing: 0.01em;
    }

    .cta-row {
      display: flex;
      gap: 14px;
      margin-top: 40px;
      opacity: 0;
    }

    .btn {
      padding: 12px 26px;
      border: none;
      border-radius: var(--radius-sm);
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
      font-family: inherit;
      transition: box-shadow 0.2s ease, border-color 0.2s ease;
      will-change: transform;
    }

    .btn-primary {
      background: linear-gradient(135deg, hsl(var(--brand)), hsl(var(--brand-hover)));
      color: #fff;
      box-shadow: 0 0 0 1px hsla(var(--brand), 0.4), 0 8px 28px hsla(var(--brand), 0.28);
    }
    .btn-primary:hover {
      box-shadow: 0 0 0 1px hsla(var(--brand), 0.6), 0 12px 40px hsla(var(--brand), 0.38);
    }

    .btn-secondary {
      background: hsla(var(--surface-tertiary), 0.6);
      color: hsl(var(--content));
      border: 1px solid hsla(var(--border), 0.8);
      backdrop-filter: blur(8px);
    }
    .btn-secondary:hover {
      border-color: hsl(var(--brand));
    }

    .hint {
      margin-top: 56px;
      font-size: 11px;
      color: hsl(var(--content-muted));
      opacity: 0;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .hint kbd {
      font-family: inherit;
      font-size: 10px;
      padding: 2px 6px;
      border: 1px solid hsla(var(--border), 0.9);
      border-radius: 4px;
      background: hsla(var(--surface-tertiary), 0.5);
    }
  </style>
</svelte:head>

<div class="welcome-root" bind:this={root}>
  <div class="orb orb-a" bind:this={orbA}></div>
  <div class="orb orb-b" bind:this={orbB}></div>

  <div class="welcome-inner">
    <button class="logo-mark" bind:this={logoWrap} onclick={() => handleEnter('/chat')} aria-label="Enter Terllama">
      <svg viewBox="0 0 96 96" aria-label="Terllama gears">
        <circle bind:this={halo} class="halo" cx="48" cy="48" r="30" fill="none" stroke="hsla(270, 90%, 66%, 0.25)"
          stroke-width="1" opacity="0" />
        {#each gears as gear}
          <g transform={`translate(${gear.cx} ${gear.cy})`}>
            <g class="gear-spin">
              <path class="gear-path" d={gearPath(outerR, innerR, gear.phase)} />
              <circle class="axle" cx="0" cy="0" r="3" />
            </g>
          </g>
        {/each}
      </svg>
    </button>

    <div class="brand" aria-label="Terllama">
      {#each brand.split('') as letter, i}
        <span
          class="brand-letter"
          class:accent={letter.toLowerCase() === 't'}
        >{letter}</span>
      {/each}
    </div>

    <p class="tagline" bind:this={tagline}>
      CPU-first ternary LLM inference — tiny, local, fast.
    </p>

    <div class="cta-row" bind:this={ctaRow}>
      <button class="btn btn-primary" bind:this={primaryBtn} onclick={() => handleEnter('/library')}>
        Enter
      </button>
      <button class="btn btn-secondary" bind:this={secondaryBtn} onclick={() => handleEnter('/chat')}>
        Open Chat
      </button>
    </div>

    <div class="hint" bind:this={hint}>
      <kbd>Enter</kbd> to continue
    </div>
  </div>
</div>

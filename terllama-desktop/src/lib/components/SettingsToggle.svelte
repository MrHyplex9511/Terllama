<script lang="ts">
  let {
    checked = false,
    onChange,
    label = '',
    description = '',
    disabled = false,
  }: {
    checked?: boolean;
    onChange?: (v: boolean) => void;
    label?: string;
    description?: string;
    disabled?: boolean;
  } = $props();

  function toggle() {
    if (disabled) return;
    const newVal = !checked;
    checked = newVal;
    onChange?.(newVal);
  }
</script>

<label class="toggle-row" class:disabled>
  <div class="toggle-text">
    <span class="toggle-label">{label}</span>
    {#if description}
      <span class="toggle-desc">{description}</span>
    {/if}
  </div>
  <button
    class="toggle-switch"
    class:on={checked}
    onclick={toggle}
    disabled={disabled}
    role="switch"
    aria-checked={checked}
    aria-label={label || 'Toggle'}
  >
    <span class="toggle-knob"></span>
  </button>
</label>

<style>
  .toggle-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 0;
    cursor: pointer;
  }

  .toggle-row.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .toggle-text {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .toggle-label {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-primary);
  }

  .toggle-desc {
    font-size: 12px;
    color: var(--text-secondary);
  }

  .toggle-switch {
    width: 44px;
    height: 24px;
    border-radius: 12px;
    border: none;
    background: var(--bg-tertiary);
    cursor: pointer;
    position: relative;
    transition: background 0.2s;
    flex-shrink: 0;
    padding: 0;
  }

  .toggle-switch.on {
    background: var(--accent);
  }

  .toggle-knob {
    position: absolute;
    top: 2px;
    left: 2px;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: white;
    transition: transform 0.2s;
  }

  .toggle-switch.on .toggle-knob {
    transform: translateX(20px);
  }
</style>

import type { Settings } from '../../types';
import { invoke } from '@tauri-apps/api/core';

const defaultSettings: Settings = {
  modelDir: '',
  port: 8375,
  keepAlive: 300,
  autoStart: false,
  theme: 'dark',
  ggufMode: false,
  gpuLayers: 0,
  cpuThreads: 4,
};

let settings = $state<Settings>({ ...defaultSettings });
let loaded = $state(false);

export function getSettingsState() {
  return {
    get settings() {
      return settings;
    },
    get loaded() {
      return loaded;
    },
    loadSettings,
    saveSettings,
    updateSetting,
  };
}

async function loadSettings() {
  try {
    const s = await invoke<Settings>('get_settings');
    settings = { ...defaultSettings, ...s };
    loaded = true;
  } catch (e) {
    console.error('Failed to load settings:', e);
    settings = { ...defaultSettings };
    loaded = true;
  }
}

async function saveSettings() {
  try {
    await invoke('save_settings', { settings });
  } catch (e) {
    console.error('Failed to save settings:', e);
  }
}

function updateSetting<K extends keyof Settings>(key: K, value: Settings[K]) {
  settings = { ...settings, [key]: value };
}

import { invoke } from '@tauri-apps/api/core';
import type { RegistryModel, DownloadedModel, DownloadProgress, ConvertProgress } from '../../types';

let registry = $state<RegistryModel[]>([]);
let downloadedModels = $state<DownloadedModel[]>([]);
let activeModel = $state<string | null>(null);
let downloadProgress = $state<DownloadProgress | null>(null);
let isDownloading = $state(false);
let convertProgress = $state<ConvertProgress | null>(null);
let isConverting = $state(false);
let loading = $state(false);

export function getModelsState() {
  return {
    get registry() {
      return registry;
    },
    get downloadedModels() {
      return downloadedModels;
    },
    get activeModel() {
      return activeModel;
    },
    get downloadProgress() {
      return downloadProgress;
    },
    get isDownloading() {
      return isDownloading;
    },
    get convertProgress() {
      return convertProgress;
    },
    get isConverting() {
      return isConverting;
    },
    get loading() {
      return loading;
    },
    setRegistry,
    setDownloadedModels,
    setActiveModel,
    setDownloadProgress,
    setIsDownloading,
    setConvertProgress,
    setIsConverting,
    setLoading,
    refreshDownloaded,
  };
}

function setRegistry(v: RegistryModel[]) {
  registry = v;
}
function setDownloadedModels(v: DownloadedModel[]) {
  downloadedModels = v;
}
function setActiveModel(v: string | null) {
  activeModel = v;
}
function setDownloadProgress(v: DownloadProgress | null) {
  downloadProgress = v;
}
function setIsDownloading(v: boolean) {
  isDownloading = v;
}
function setConvertProgress(v: ConvertProgress | null) {
  convertProgress = v;
}
function setIsConverting(v: boolean) {
  isConverting = v;
}
function setLoading(v: boolean) {
  loading = v;
}

/** Re-query the backend for installed models (called when a download/conversion finishes). */
async function refreshDownloaded() {
  try {
    const downloaded = await invoke<DownloadedModel[]>('list_downloaded_models');
    setDownloadedModels(downloaded);
  } catch {
    // Non-fatal — the library page also refreshes on load.
  }
}

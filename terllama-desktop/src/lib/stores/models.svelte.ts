import type { RegistryModel, DownloadedModel, DownloadProgress } from '../../types';

let registry = $state<RegistryModel[]>([]);
let downloadedModels = $state<DownloadedModel[]>([]);
let activeModel = $state<string | null>(null);
let downloadProgress = $state<DownloadProgress | null>(null);
let isDownloading = $state(false);
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
    get loading() {
      return loading;
    },
    setRegistry,
    setDownloadedModels,
    setActiveModel,
    setDownloadProgress,
    setIsDownloading,
    setLoading,
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
function setLoading(v: boolean) {
  loading = v;
}

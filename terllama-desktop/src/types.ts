export interface RegistryModel {
  id: string;
  name: string;
  hf_repo: string;
  format: string;
  description: string;
  context: number;
  size_mb: number;
  quants: QuantOptions;
}

export interface QuantOptions {
  ternary: QuantInfo;
  q4_k_m: QuantInfo;
  q8_0: QuantInfo;
}

export interface QuantInfo {
  available: boolean;
  size_mb: number;
  filename: string;
}

export interface DownloadedModel {
  id: string;
  path: string;
  size_mb: number;
  quant: string;
}

export interface DownloadProgress {
  model_id: string;
  file: string;
  downloaded: number;
  total: number;
  speed: number;
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  created_at: string;
}

export interface Settings {
  modelDir: string;
  port: number;
  keepAlive: number;
  autoStart: boolean;
  theme: 'dark' | 'light' | 'system';
  ggufMode: boolean;
  gpuLayers: number;
  cpuThreads: number;
}

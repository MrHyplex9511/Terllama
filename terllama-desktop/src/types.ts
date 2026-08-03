export interface RegistryModel {
  id: string;
  name: string;
  hf_repo: string;
  format: string;
  description: string;
  context: number;
  size_mb: number;
  formats: Formats;
}

export interface Formats {
  fp: FormatInfo;
  q4: FormatInfo;
  ternary: FormatInfo;
}

export interface FormatInfo {
  available: boolean;
  size_mb: number;
  filename: string;
  files: string[];
  hf_repo: string;
  needs_conversion: boolean;
  note: string;
}

export type DownloadFormat = 'fp' | 'q4' | 'ternary';

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

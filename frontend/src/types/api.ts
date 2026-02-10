/**
 * API Types for PPT Translator
 */

export interface ModelInfo {
  id: string;
  name: string;
  provider: string;
}

export interface LanguageInfo {
  code: string;
  name: string;
}

export interface ConfigResponse {
  max_upload_size_mb: number;
  providers: string[];
  default_provider: string;
  default_model: string;
}

export interface JobCreateResponse {
  job_id: string;
  status: string;
}

export interface JobProgress {
  status: string;
  current_batch: number;
  total_batches: number;
  current_sentence: number;
  total_sentences: number;
  percent: number;
  message: string;
}

export interface JobStatusResponse {
  job_id: string;
  job_type: string;
  state: "pending" | "running" | "completed" | "failed" | "cancelled";
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
  progress: JobProgress | null;
  error_message: string | null;
}

export interface ExtractionResponse {
  markdown: string;
  slide_count: number;
}

export interface SSEEvent {
  type: "progress" | "complete" | "error" | "started" | "cancelled" | "keepalive";
  data: Record<string, unknown>;
  timestamp: number;
}

export interface FilenameSettings {
  mode: "auto" | "custom";
  includeLanguage: boolean;
  includeOriginalName: boolean;
  includeModel: boolean;
  includeDate: boolean;
  customName: string;
}

export type TextFitMode = "none" | "auto_shrink" | "expand_box" | "shrink_then_expand";

export type ImageCompression = "none" | "high" | "medium" | "low";

export interface TranslationSettings {
  sourceLang: string;
  targetLang: string;
  provider: string;
  model: string;
  context: string;
  instructions: string;
  preprocessRepetitions: boolean;
  translateNotes: boolean;
  filenameSettings: FilenameSettings;
  textFitMode: TextFitMode;
  minFontRatio: number;
  imageCompression: ImageCompression;
}

export interface ExtractionSettings {
  figures: "omit" | "placeholder";
  charts: "labels" | "placeholder" | "omit";
  withNotes: boolean;
  tableHeader: boolean;
}

export interface SummarizeRequest {
  markdown: string;
  provider?: string;
  model?: string;
}

export interface SummarizeResponse {
  summary: string;
}

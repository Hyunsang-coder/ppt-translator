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

// --- Review / edit loop (WP-C5) ---

export interface FragmentFinding {
  type: string;
  severity: "critical" | "major" | "minor";
  description: string;
  suggested_fix: string | null;
  related_location: Record<string, unknown> | null;
}

export interface FragmentItem {
  index: number;
  slide: number;
  shape: number;
  paragraph: number;
  slide_title: string | null;
  is_note: boolean;
  source: string;
  target: string;
  repeat_count: number;
  length_budget: number | null;
  findings: FragmentFinding[];
  edited: boolean;
}

export interface FragmentsResponse {
  job_id: string;
  total: number;
  fragments: FragmentItem[];
}

export interface PartialCandidate {
  index: number;
  slide: number;
  is_note: boolean;
  target: string;
}

export interface FragmentEditRequest {
  action: "edit" | "retranslate" | "ignore";
  target?: string;
  instruction?: string;
  propagate_identical?: boolean;
  finding_type?: string;
}

export interface FragmentEditResponse {
  index: number;
  target: string;
  changed_indices: number[];
  partial_candidates: PartialCandidate[];
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
  componentOrder: Array<"language" | "originalName" | "model" | "date">;
  customName: string;
}

export type TextFitMode = "none" | "auto_shrink" | "expand_box" | "shrink_then_expand";

export type ImageCompression = "none" | "high" | "medium" | "low";

export type LengthLimit = 110 | 130 | 150;

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
  lengthLimit: LengthLimit | null;
}

export interface ExtractionSettings {
  figures: "omit" | "placeholder";
  charts: "labels" | "placeholder" | "omit";
  withNotes: boolean;
  tableHeader: boolean;
}

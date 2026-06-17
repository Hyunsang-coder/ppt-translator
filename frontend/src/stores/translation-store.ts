/**
 * Zustand store for translation state management
 */

import { create } from "zustand";
import type { JobProgress, TranslationSettings } from "@/types/api";

export interface LogEntry {
  id: string;
  timestamp: Date;
  message: string;
  type: "info" | "success" | "error" | "warning";
}

export type TranslationStatus =
  | "idle"
  | "uploading"
  | "translating"
  | "completed"
  | "failed"
  | "cancelled";

interface TranslationState {
  // File state
  pptFile: File | null;
  glossaryFile: File | null;

  // Settings
  settings: TranslationSettings;

  // Job state
  jobId: string | null;
  status: TranslationStatus;
  progress: JobProgress | null;
  errorMessage: string | null;

  // Result
  resultFilename: string | null;

  // Logs
  logs: LogEntry[];

  // Actions
  setPptFile: (file: File | null) => void;
  setGlossaryFile: (file: File | null) => void;
  updateSettings: (settings: Partial<TranslationSettings>) => void;
  setJobId: (jobId: string | null) => void;
  setStatus: (status: TranslationStatus) => void;
  setProgress: (progress: JobProgress | null) => void;
  setErrorMessage: (message: string | null) => void;
  setResultFilename: (filename: string | null) => void;
  addLog: (message: string, type?: LogEntry["type"]) => void;
  clearLogs: () => void;
  resetJobState: () => void;
  reset: () => void;
}

const DEFAULT_SETTINGS: TranslationSettings = {
  sourceLang: "Auto",
  targetLang: "", // 타겟 언어는 필수 선택
  provider: "anthropic",
  model: "claude-sonnet-4-6",
  context: "",
  instructions: "",
  preprocessRepetitions: true,
  translateNotes: false,
  filenameSettings: {
    mode: "auto",
    includeLanguage: true,
    includeOriginalName: true,
    includeModel: false,
    includeDate: true,
    componentOrder: ["language", "originalName", "model", "date"],
    customName: "",
  },
  textFitMode: "expand_box",
  minFontRatio: 80,
  imageCompression: "medium",
  lengthLimit: null,
};

const MAX_LOGS = 400;

export const useTranslationStore = create<TranslationState>((set) => ({
  // Initial state
  pptFile: null,
  glossaryFile: null,
  settings: DEFAULT_SETTINGS,
  jobId: null,
  status: "idle",
  progress: null,
  errorMessage: null,
  resultFilename: null,
  logs: [],

  // Actions
  setPptFile: (file) =>
    set((state) => ({
      pptFile: file,
      ...(file !== state.pptFile
        ? {
            jobId: null,
            status: "idle" as const,
            progress: null,
            errorMessage: null,
            resultFilename: null,
            logs: [],
          }
        : {}),
    })),

  setGlossaryFile: (file) => set({ glossaryFile: file }),

  updateSettings: (newSettings) =>
    set((state) => ({
      settings: { ...state.settings, ...newSettings },
    })),

  setJobId: (jobId) => set({ jobId }),
  setStatus: (status) => set({ status }),
  setProgress: (progress) =>
    set((state) => {
      if (!progress) return { progress: null };

      const previous = state.progress;
      const incomingPercent = Math.max(0, Math.min(100, progress.percent ?? 0));

      if (previous && incomingPercent < (previous.percent ?? 0)) {
        return { progress: previous };
      }

      const next = {
        ...progress,
        percent: incomingPercent,
      };

      if (previous) {
        if (next.total_batches === previous.total_batches) {
          next.current_batch = Math.max(next.current_batch, previous.current_batch);
        }
        if (next.total_sentences === previous.total_sentences) {
          next.current_sentence = Math.max(next.current_sentence, previous.current_sentence);
        }
      }

      return { progress: next };
    }),
  setErrorMessage: (errorMessage) => set({ errorMessage }),
  setResultFilename: (resultFilename) => set({ resultFilename }),

  addLog: (message, type = "info") =>
    set((state) => {
      const newLog: LogEntry = {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        timestamp: new Date(),
        message,
        type,
      };
      const logs = [...state.logs, newLog];
      // Keep only the last MAX_LOGS entries
      return { logs: logs.slice(-MAX_LOGS) };
    }),

  clearLogs: () => set({ logs: [] }),

  resetJobState: () =>
    set({
      jobId: null,
      status: "idle",
      progress: null,
      errorMessage: null,
      resultFilename: null,
      logs: [],
    }),

  reset: () =>
    set({
      pptFile: null,
      glossaryFile: null,
      jobId: null,
      status: "idle",
      progress: null,
      errorMessage: null,
      resultFilename: null,
      logs: [],
    }),
}));

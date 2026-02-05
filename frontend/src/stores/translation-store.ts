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
  reset: () => void;
}

const DEFAULT_SETTINGS: TranslationSettings = {
  sourceLang: "Auto",
  targetLang: "Auto",
  provider: "openai",
  model: "gpt-5.2",
  userPrompt: "",
  preprocessRepetitions: true,
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
  setPptFile: (file) => set({ pptFile: file }),
  setGlossaryFile: (file) => set({ glossaryFile: file }),

  updateSettings: (newSettings) =>
    set((state) => ({
      settings: { ...state.settings, ...newSettings },
    })),

  setJobId: (jobId) => set({ jobId }),
  setStatus: (status) => set({ status }),
  setProgress: (progress) => set({ progress }),
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

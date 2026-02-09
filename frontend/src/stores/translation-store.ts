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

  // Markdown cache (for context/instructions generation)
  cachedMarkdown: string | null;
  cachedMarkdownFileKey: string | null; // file name + size + lastModified

  // Context generation
  generatedContext: string;
  isGeneratingContext: boolean;

  // Instructions generation
  generatedInstructions: string;
  isGeneratingInstructions: boolean;

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
  setCachedMarkdown: (markdown: string | null, fileKey: string | null) => void;
  setGeneratedContext: (context: string) => void;
  setIsGeneratingContext: (loading: boolean) => void;
  setGeneratedInstructions: (instructions: string) => void;
  setIsGeneratingInstructions: (loading: boolean) => void;
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
  provider: "openai",
  model: "gpt-5.2",
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
    customName: "",
  },
  textFitMode: "expand_box",
  minFontRatio: 80,
};

const MAX_LOGS = 400;

export const useTranslationStore = create<TranslationState>((set) => ({
  // Initial state
  pptFile: null,
  glossaryFile: null,
  settings: DEFAULT_SETTINGS,
  cachedMarkdown: null,
  cachedMarkdownFileKey: null,
  generatedContext: "",
  isGeneratingContext: false,
  generatedInstructions: "",
  isGeneratingInstructions: false,
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

  setCachedMarkdown: (markdown, fileKey) =>
    set({ cachedMarkdown: markdown, cachedMarkdownFileKey: fileKey }),

  setGeneratedContext: (context) => set({ generatedContext: context }),
  setIsGeneratingContext: (loading) => set({ isGeneratingContext: loading }),
  setGeneratedInstructions: (instructions) => set({ generatedInstructions: instructions }),
  setIsGeneratingInstructions: (loading) => set({ isGeneratingInstructions: loading }),

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
      cachedMarkdown: null,
      cachedMarkdownFileKey: null,
      generatedContext: "",
      isGeneratingContext: false,
      generatedInstructions: "",
      isGeneratingInstructions: false,
      jobId: null,
      status: "idle",
      progress: null,
      errorMessage: null,
      resultFilename: null,
      logs: [],
    }),
}));

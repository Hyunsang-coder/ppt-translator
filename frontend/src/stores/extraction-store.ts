/**
 * Zustand store for text extraction state management
 */

import { create } from "zustand";
import type { ExtractionSettings } from "@/types/api";

export type ExtractionStatus = "idle" | "extracting" | "completed" | "failed";

interface ExtractionState {
  // File state
  pptFile: File | null;

  // Settings
  settings: ExtractionSettings;

  // Job state
  status: ExtractionStatus;
  errorMessage: string | null;

  // Result
  markdown: string | null;
  slideCount: number | null;

  // Actions
  setPptFile: (file: File | null) => void;
  updateSettings: (settings: Partial<ExtractionSettings>) => void;
  setStatus: (status: ExtractionStatus) => void;
  setErrorMessage: (message: string | null) => void;
  setResult: (markdown: string, slideCount: number) => void;
  reset: () => void;
}

const DEFAULT_SETTINGS: ExtractionSettings = {
  figures: "omit",
  charts: "labels",
  withNotes: false,
  tableHeader: true,
};

export const useExtractionStore = create<ExtractionState>((set) => ({
  // Initial state
  pptFile: null,
  settings: DEFAULT_SETTINGS,
  status: "idle",
  errorMessage: null,
  markdown: null,
  slideCount: null,

  // Actions
  setPptFile: (file) => set({ pptFile: file }),

  updateSettings: (newSettings) =>
    set((state) => ({
      settings: { ...state.settings, ...newSettings },
    })),

  setStatus: (status) => set({ status }),
  setErrorMessage: (errorMessage) => set({ errorMessage }),

  setResult: (markdown, slideCount) =>
    set({
      markdown,
      slideCount,
      status: "completed",
    }),

  reset: () =>
    set({
      pptFile: null,
      status: "idle",
      errorMessage: null,
      markdown: null,
      slideCount: null,
    }),
}));

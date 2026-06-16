import { create } from "zustand";

interface SharedFileState {
  pptFile: File | null;
  setPptFile: (file: File | null) => void;
}

export function getFileKey(file: File | null): string | null {
  if (!file) return null;
  return `${file.name}|${file.size}|${file.lastModified}`;
}

export const useSharedFileStore = create<SharedFileState>((set) => ({
  pptFile: null,
  setPptFile: (pptFile) => set({ pptFile }),
}));

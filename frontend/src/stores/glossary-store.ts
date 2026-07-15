/**
 * Persisted glossary library for the in-app terminology editor.
 *
 * Independent from translation-store so reset()/retranslate keep the library.
 */

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  createId,
  MAX_TERM_CHARS,
  normalizeTerm,
  upsertEntries,
  type Glossary,
  type GlossaryEntry,
} from "@/lib/glossary";

const DEFAULT_GLOSSARY_NAME = "기본 용어집";

interface GlossaryState {
  glossaries: Glossary[];
  activeGlossaryId: string | null;

  ensureDefaultGlossary: () => string;
  setActiveGlossaryId: (id: string | null) => void;
  createGlossary: (name: string) => string;
  renameGlossary: (id: string, name: string) => void;
  deleteGlossary: (id: string) => void;
  addEntry: (glossaryId: string, source: string, target: string, notes?: string) => void;
  updateEntry: (
    glossaryId: string,
    entryId: string,
    patch: Partial<Pick<GlossaryEntry, "source" | "target" | "notes">>
  ) => void;
  deleteEntry: (glossaryId: string, entryId: string) => void;
  importEntries: (glossaryId: string, incoming: Omit<GlossaryEntry, "id">[]) => {
    inserted: number;
    updated: number;
  };
  getActiveGlossary: () => Glossary | null;
  getActiveEntries: () => GlossaryEntry[];
}

function touch(glossary: Glossary, entries: GlossaryEntry[]): Glossary {
  return { ...glossary, entries, updatedAt: Date.now() };
}

function requireGlossary(glossaries: Glossary[], id: string): Glossary {
  const found = glossaries.find((g) => g.id === id);
  if (!found) throw new Error("용어집을 찾을 수 없습니다.");
  return found;
}

export const useGlossaryStore = create<GlossaryState>()(
  persist(
    (set, get) => ({
      glossaries: [],
      activeGlossaryId: null,

      ensureDefaultGlossary: () => {
        const state = get();
        if (state.activeGlossaryId) {
          const exists = state.glossaries.some((g) => g.id === state.activeGlossaryId);
          if (exists) return state.activeGlossaryId;
          // Stale id after delete — fall back without treating null as "cleared".
          if (state.glossaries.length > 0) {
            const id = state.glossaries[0].id;
            set({ activeGlossaryId: id });
            return id;
          }
        }
        // Empty library: create a default and activate it.
        if (state.glossaries.length === 0) {
          const id = createId("glossary");
          const glossary: Glossary = {
            id,
            name: DEFAULT_GLOSSARY_NAME,
            entries: [],
            updatedAt: Date.now(),
          };
          set({ glossaries: [glossary], activeGlossaryId: id });
          return id;
        }
        // Library exists but activeGlossaryId is null → user cleared for this job.
        // Return a glossary id for editing without re-activating.
        return state.glossaries[0].id;
      },

      setActiveGlossaryId: (id) => set({ activeGlossaryId: id }),

      createGlossary: (name) => {
        const trimmed = name.trim() || DEFAULT_GLOSSARY_NAME;
        const id = createId("glossary");
        const glossary: Glossary = {
          id,
          name: trimmed,
          entries: [],
          updatedAt: Date.now(),
        };
        set((state) => ({
          glossaries: [...state.glossaries, glossary],
          activeGlossaryId: id,
        }));
        return id;
      },

      renameGlossary: (id, name) => {
        const trimmed = name.trim();
        if (!trimmed) return;
        set((state) => ({
          glossaries: state.glossaries.map((g) =>
            g.id === id ? { ...g, name: trimmed, updatedAt: Date.now() } : g
          ),
        }));
      },

      deleteGlossary: (id) => {
        set((state) => {
          const glossaries = state.glossaries.filter((g) => g.id !== id);
          const activeGlossaryId =
            state.activeGlossaryId === id
              ? glossaries[0]?.id ?? null
              : state.activeGlossaryId;
          return { glossaries, activeGlossaryId };
        });
      },

      addEntry: (glossaryId, source, target, notes) => {
        const src = normalizeTerm(source);
        const tgt = normalizeTerm(target);
        if (!src || !tgt) return;
        if (src.length > MAX_TERM_CHARS || tgt.length > MAX_TERM_CHARS) {
          throw new Error(`용어 길이는 ${MAX_TERM_CHARS}자를 초과할 수 없습니다.`);
        }

        const glossary = requireGlossary(get().glossaries, glossaryId);
        const { entries } = upsertEntries(glossary.entries, [
          { source: src, target: tgt, notes: notes?.trim() || undefined },
        ]);
        set((state) => ({
          glossaries: state.glossaries.map((g) =>
            g.id === glossaryId ? touch(g, entries) : g
          ),
          activeGlossaryId: glossaryId,
        }));
      },

      updateEntry: (glossaryId, entryId, patch) => {
        const glossary = requireGlossary(get().glossaries, glossaryId);
        const entries = glossary.entries.map((entry) => {
          if (entry.id !== entryId) return entry;
          const next = {
            ...entry,
            source: patch.source !== undefined ? normalizeTerm(patch.source) : entry.source,
            target: patch.target !== undefined ? normalizeTerm(patch.target) : entry.target,
            notes:
              patch.notes !== undefined
                ? patch.notes.trim() || undefined
                : entry.notes,
          };
          if (!next.source || !next.target) {
            throw new Error("원문과 번역을 모두 입력해주세요.");
          }
          if (next.source.length > MAX_TERM_CHARS || next.target.length > MAX_TERM_CHARS) {
            throw new Error(`용어 길이는 ${MAX_TERM_CHARS}자를 초과할 수 없습니다.`);
          }
          return next;
        });
        set((state) => ({
          glossaries: state.glossaries.map((g) =>
            g.id === glossaryId ? touch(g, entries) : g
          ),
        }));
      },

      deleteEntry: (glossaryId, entryId) => {
        set((state) => ({
          glossaries: state.glossaries.map((g) =>
            g.id === glossaryId
              ? touch(
                  g,
                  g.entries.filter((e) => e.id !== entryId)
                )
              : g
          ),
        }));
      },

      importEntries: (glossaryId, incoming) => {
        const state = get();
        const glossary = requireGlossary(state.glossaries, glossaryId);
        const { entries, inserted, updated } = upsertEntries(glossary.entries, incoming);
        set({
          glossaries: state.glossaries.map((g) =>
            g.id === glossaryId ? touch(g, entries) : g
          ),
          activeGlossaryId: glossaryId,
        });
        return { inserted, updated };
      },

      getActiveGlossary: () => {
        const { glossaries, activeGlossaryId } = get();
        // null means intentionally unused for the current job (library kept).
        if (!activeGlossaryId) return null;
        return glossaries.find((g) => g.id === activeGlossaryId) ?? null;
      },

      getActiveEntries: () => get().getActiveGlossary()?.entries ?? [],
    }),
    {
      name: "ppt-translator-glossary",
      partialize: (state) => ({
        glossaries: state.glossaries,
        activeGlossaryId: state.activeGlossaryId,
      }),
    }
  )
);

/**
 * Browser-persisted glossary library.
 *
 * Editing selection is intentionally kept in the UI. This store persists only
 * the reusable library and the ordered glossary ids applied to new jobs.
 */

import { create } from "zustand";
import {
  createId,
  glossaryTermKey,
  MAX_GLOSSARY_ENTRIES,
  MAX_GLOSSARY_STORAGE_BYTES,
  MAX_TOTAL_GLOSSARY_ENTRIES,
  mergeGlossaryEntries,
  upsertEntries,
  utf8ByteLength,
  validateGlossaryEntry,
  validateGlossaryName,
  type Glossary,
  type GlossaryEntry,
} from "@/lib/glossary";

export const GLOSSARY_STORAGE_KEY = "ppt-translator-glossary";
export const GLOSSARY_RECOVERY_KEY = "ppt-translator-glossary-recovery";
export const GLOSSARY_STORAGE_VERSION = 2;

const DEFAULT_GLOSSARY_NAME = "기본 용어집";

export interface PersistedGlossaryState {
  glossaries: Glossary[];
  activeGlossaryIds: string[];
}

interface StoredEnvelope {
  state: PersistedGlossaryState;
  version: number;
}

export interface GlossaryStoreState extends PersistedGlossaryState {
  hydrated: boolean;
  storageWarning: string | null;

  hydrate: () => void;
  syncFromStorage: (raw: string | null) => void;
  dismissStorageWarning: () => void;
  ensureDefaultGlossary: () => string;
  createGlossary: (name: string) => string;
  renameGlossary: (id: string, name: string) => void;
  deleteGlossary: (id: string) => void;
  setActiveGlossaryIds: (ids: string[]) => void;
  toggleGlossaryActive: (id: string) => void;
  moveActiveGlossary: (id: string, direction: -1 | 1) => void;
  addEntry: (
    glossaryId: string,
    source: string,
    target: string,
    notes?: string
  ) => { entry: GlossaryEntry; activated: boolean };
  updateEntry: (
    glossaryId: string,
    entryId: string,
    patch: Partial<Pick<GlossaryEntry, "source" | "target" | "notes">>
  ) => GlossaryEntry;
  deleteEntry: (glossaryId: string, entryId: string) => void;
  importEntries: (glossaryId: string, incoming: Omit<GlossaryEntry, "id">[]) => {
    inserted: number;
    updated: number;
  };
  getActiveGlossaries: () => Glossary[];
  getActiveEntries: () => GlossaryEntry[];
}

export type GlossaryStorageProvider = () => Storage | null;

function defaultStorageProvider(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function countEntries(glossaries: Glossary[]): number {
  return glossaries.reduce((sum, glossary) => sum + glossary.entries.length, 0);
}

function sanitizeGlossaryState(input: unknown): PersistedGlossaryState {
  const value = input && typeof input === "object" ? input as Record<string, unknown> : {};
  const rawGlossaries = Array.isArray(value.glossaries) ? value.glossaries : [];
  const glossaries: Glossary[] = [];
  const seenGlossaryIds = new Set<string>();

  for (const rawGlossary of rawGlossaries) {
    if (!rawGlossary || typeof rawGlossary !== "object") continue;
    const item = rawGlossary as Record<string, unknown>;
    const id = typeof item.id === "string" && item.id.trim() ? item.id : createId("glossary");
    if (seenGlossaryIds.has(id)) continue;
    let name: string;
    try {
      name = validateGlossaryName(typeof item.name === "string" ? item.name : DEFAULT_GLOSSARY_NAME);
    } catch {
      name = DEFAULT_GLOSSARY_NAME;
    }

    const incoming: Omit<GlossaryEntry, "id">[] = [];
    const entryIds: string[] = [];
    for (const rawEntry of Array.isArray(item.entries) ? item.entries : []) {
      if (!rawEntry || typeof rawEntry !== "object") continue;
      const entry = rawEntry as Record<string, unknown>;
      try {
        incoming.push(validateGlossaryEntry({
          source: typeof entry.source === "string" ? entry.source : "",
          target: typeof entry.target === "string" ? entry.target : "",
          notes: typeof entry.notes === "string" ? entry.notes : undefined,
        }));
        entryIds.push(typeof entry.id === "string" && entry.id ? entry.id : createId("term"));
      } catch {
        // Corrupt individual rows are ignored while the rest of the library is recovered.
      }
    }

    const deduped = new Map<string, GlossaryEntry>();
    incoming.forEach((entry, index) => {
      const key = glossaryTermKey(entry.source);
      const existing = deduped.get(key);
      deduped.set(key, { id: existing?.id ?? entryIds[index]!, ...entry });
    });
    if (deduped.size > MAX_GLOSSARY_ENTRIES) {
      throw new Error(`용어집 항목은 최대 ${MAX_GLOSSARY_ENTRIES}개까지 지원합니다.`);
    }

    seenGlossaryIds.add(id);
    glossaries.push({
      id,
      name,
      entries: Array.from(deduped.values()),
      updatedAt: typeof item.updatedAt === "number" && Number.isFinite(item.updatedAt)
        ? item.updatedAt
        : Date.now(),
    });
  }

  if (countEntries(glossaries) > MAX_TOTAL_GLOSSARY_ENTRIES) {
    throw new Error(`전체 용어는 최대 ${MAX_TOTAL_GLOSSARY_ENTRIES}개까지 저장할 수 있습니다.`);
  }

  const knownIds = new Set(glossaries.map((glossary) => glossary.id));
  const legacyActiveId = typeof value.activeGlossaryId === "string" ? value.activeGlossaryId : null;
  const requestedActiveIds = Array.isArray(value.activeGlossaryIds)
    ? value.activeGlossaryIds
    : legacyActiveId ? [legacyActiveId] : [];
  const activeGlossaryIds: string[] = [];
  for (const id of requestedActiveIds) {
    if (typeof id !== "string" || !knownIds.has(id) || activeGlossaryIds.includes(id)) continue;
    activeGlossaryIds.push(id);
  }
  // Also validates the merged active count after migration.
  mergeGlossaryEntries(glossaries, activeGlossaryIds);
  return { glossaries, activeGlossaryIds };
}

export function parseGlossaryStorage(raw: string): PersistedGlossaryState {
  const parsed = JSON.parse(raw) as unknown;
  if (!parsed || typeof parsed !== "object") {
    throw new Error("저장된 용어집 데이터가 올바르지 않습니다.");
  }
  const record = parsed as Record<string, unknown>;
  const version = typeof record.version === "number" ? record.version : 0;
  if (version > GLOSSARY_STORAGE_VERSION) {
    throw new Error("더 새로운 앱에서 저장한 용어집이라 현재 버전에서 열 수 없습니다.");
  }
  return sanitizeGlossaryState(record.state ?? record);
}

export function serializeGlossaryStorage(state: PersistedGlossaryState): string {
  const sanitized = sanitizeGlossaryState(state);
  const serialized = JSON.stringify({
    state: sanitized,
    version: GLOSSARY_STORAGE_VERSION,
  } satisfies StoredEnvelope);
  if (utf8ByteLength(serialized) > MAX_GLOSSARY_STORAGE_BYTES) {
    throw new Error("용어집 저장 공간 한도를 초과했습니다. 일부 용어를 삭제하거나 내보내 백업해주세요.");
  }
  return serialized;
}

function storageErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  return "용어집을 브라우저에 저장하지 못했습니다.";
}

export function createGlossaryStore(storageProvider: GlossaryStorageProvider = defaultStorageProvider) {
  return create<GlossaryStoreState>((set, get) => {
    const persistAndSet = (next: PersistedGlossaryState): void => {
      const serialized = serializeGlossaryStorage(next);
      const storage = storageProvider();
      if (!storage) {
        const message = "브라우저 저장소를 사용할 수 없어 용어집을 저장하지 못했습니다.";
        set({ storageWarning: message });
        throw new Error(message);
      }
      try {
        storage.setItem(GLOSSARY_STORAGE_KEY, serialized);
      } catch (error) {
        const errorName = error && typeof error === "object" && "name" in error
          ? String(error.name)
          : "";
        const message = errorName === "QuotaExceededError"
          ? "브라우저 저장 공간이 부족합니다. 용어를 내보낸 뒤 일부 항목을 삭제해주세요."
          : "브라우저가 용어집 저장을 허용하지 않았습니다. 저장소 설정을 확인해주세요.";
        set({ storageWarning: message });
        throw new Error(message, { cause: error });
      }
      set({ ...next, storageWarning: null });
    };

    const ensureHydrated = (): void => {
      if (!get().hydrated) get().hydrate();
    };

    const findGlossary = (id: string): Glossary => {
      const glossary = get().glossaries.find((item) => item.id === id);
      if (!glossary) throw new Error("용어집을 찾을 수 없습니다.");
      return glossary;
    };

    return {
      glossaries: [],
      activeGlossaryIds: [],
      hydrated: false,
      storageWarning: null,

      hydrate: () => {
        if (get().hydrated) return;
        const storage = storageProvider();
        if (!storage) {
          set({ hydrated: true, storageWarning: "브라우저 저장소를 사용할 수 없습니다." });
          return;
        }
        let raw: string | null = null;
        try {
          raw = storage.getItem(GLOSSARY_STORAGE_KEY);
          if (!raw) {
            set({ hydrated: true });
            return;
          }
          const restored = parseGlossaryStorage(raw);
          set({ ...restored, hydrated: true, storageWarning: null });
          // Rewrite legacy envelopes only after successful parsing.
          const migrated = serializeGlossaryStorage(restored);
          if (migrated !== raw) {
            try {
              storage.setItem(GLOSSARY_STORAGE_KEY, migrated);
            } catch {
              set({
                ...restored,
                hydrated: true,
                storageWarning: "기존 용어집은 불러왔지만 새 저장 형식으로 변환하지 못했습니다.",
              });
            }
          }
        } catch (error) {
          if (raw) {
            try {
              storage.setItem(GLOSSARY_RECOVERY_KEY, raw);
              storage.removeItem(GLOSSARY_STORAGE_KEY);
            } catch {
              // Keep the original key if even the recovery copy cannot be written.
            }
          }
          set({
            glossaries: [],
            activeGlossaryIds: [],
            hydrated: true,
            storageWarning: `저장된 용어집을 읽지 못해 복구용 사본을 보관했습니다. ${storageErrorMessage(error)}`,
          });
        }
      },

      syncFromStorage: (raw) => {
        if (!raw) {
          set({ glossaries: [], activeGlossaryIds: [], hydrated: true });
          return;
        }
        try {
          set({ ...parseGlossaryStorage(raw), hydrated: true, storageWarning: null });
        } catch (error) {
          set({ storageWarning: storageErrorMessage(error), hydrated: true });
        }
      },

      dismissStorageWarning: () => set({ storageWarning: null }),

      ensureDefaultGlossary: () => {
        ensureHydrated();
        const state = get();
        const active = state.activeGlossaryIds.find((id) => state.glossaries.some((g) => g.id === id));
        if (active) return active;
        if (state.glossaries[0]) return state.glossaries[0].id;
        const id = createId("glossary");
        const glossary: Glossary = {
          id,
          name: DEFAULT_GLOSSARY_NAME,
          entries: [],
          updatedAt: Date.now(),
        };
        persistAndSet({ glossaries: [glossary], activeGlossaryIds: [] });
        return id;
      },

      createGlossary: (name) => {
        ensureHydrated();
        const id = createId("glossary");
        const glossary: Glossary = {
          id,
          name: validateGlossaryName(name),
          entries: [],
          updatedAt: Date.now(),
        };
        const state = get();
        persistAndSet({
          glossaries: [...state.glossaries, glossary],
          activeGlossaryIds: state.activeGlossaryIds,
        });
        return id;
      },

      renameGlossary: (id, name) => {
        ensureHydrated();
        findGlossary(id);
        const state = get();
        persistAndSet({
          glossaries: state.glossaries.map((glossary) => glossary.id === id
            ? { ...glossary, name: validateGlossaryName(name), updatedAt: Date.now() }
            : glossary),
          activeGlossaryIds: state.activeGlossaryIds,
        });
      },

      deleteGlossary: (id) => {
        ensureHydrated();
        findGlossary(id);
        const state = get();
        persistAndSet({
          glossaries: state.glossaries.filter((glossary) => glossary.id !== id),
          activeGlossaryIds: state.activeGlossaryIds.filter((activeId) => activeId !== id),
        });
      },

      setActiveGlossaryIds: (ids) => {
        ensureHydrated();
        const state = get();
        const knownIds = new Set(state.glossaries.map((glossary) => glossary.id));
        const uniqueIds = ids.filter((id, index) => knownIds.has(id) && ids.indexOf(id) === index);
        mergeGlossaryEntries(state.glossaries, uniqueIds);
        persistAndSet({ glossaries: state.glossaries, activeGlossaryIds: uniqueIds });
      },

      toggleGlossaryActive: (id) => {
        ensureHydrated();
        findGlossary(id);
        const state = get();
        const activeGlossaryIds = state.activeGlossaryIds.includes(id)
          ? state.activeGlossaryIds.filter((activeId) => activeId !== id)
          : [...state.activeGlossaryIds, id];
        mergeGlossaryEntries(state.glossaries, activeGlossaryIds);
        persistAndSet({ glossaries: state.glossaries, activeGlossaryIds });
      },

      moveActiveGlossary: (id, direction) => {
        ensureHydrated();
        const state = get();
        const index = state.activeGlossaryIds.indexOf(id);
        const destination = index + direction;
        if (index < 0 || destination < 0 || destination >= state.activeGlossaryIds.length) return;
        const activeGlossaryIds = [...state.activeGlossaryIds];
        [activeGlossaryIds[index], activeGlossaryIds[destination]] = [
          activeGlossaryIds[destination]!,
          activeGlossaryIds[index]!,
        ];
        persistAndSet({ glossaries: state.glossaries, activeGlossaryIds });
      },

      addEntry: (glossaryId, source, target, notes) => {
        ensureHydrated();
        const glossary = findGlossary(glossaryId);
        const normalized = validateGlossaryEntry({ source, target, notes });
        if (glossary.entries.some((entry) => glossaryTermKey(entry.source) === glossaryTermKey(normalized.source))) {
          throw new Error("같은 원문을 가진 용어가 이미 이 용어집에 있습니다.");
        }
        if (countEntries(get().glossaries) >= MAX_TOTAL_GLOSSARY_ENTRIES) {
          throw new Error(`전체 용어는 최대 ${MAX_TOTAL_GLOSSARY_ENTRIES}개까지 저장할 수 있습니다.`);
        }
        if (glossary.entries.length >= MAX_GLOSSARY_ENTRIES) {
          throw new Error(`용어집 항목은 최대 ${MAX_GLOSSARY_ENTRIES}개까지 지원합니다.`);
        }
        const entry: GlossaryEntry = { id: createId("term"), ...normalized };
        const state = get();
        const activated = !state.activeGlossaryIds.includes(glossaryId);
        const glossaries = state.glossaries.map((item) => item.id === glossaryId
          ? { ...item, entries: [...item.entries, entry], updatedAt: Date.now() }
          : item);
        const activeGlossaryIds = activated
          ? [...state.activeGlossaryIds, glossaryId]
          : state.activeGlossaryIds;
        mergeGlossaryEntries(glossaries, activeGlossaryIds);
        persistAndSet({ glossaries, activeGlossaryIds });
        return { entry, activated };
      },

      updateEntry: (glossaryId, entryId, patch) => {
        ensureHydrated();
        const glossary = findGlossary(glossaryId);
        const current = glossary.entries.find((entry) => entry.id === entryId);
        if (!current) throw new Error("용어를 찾을 수 없습니다.");
        const updated: GlossaryEntry = {
          id: current.id,
          ...validateGlossaryEntry({ ...current, ...patch }),
        };
        if (glossary.entries.some((entry) => (
          entry.id !== entryId && glossaryTermKey(entry.source) === glossaryTermKey(updated.source)
        ))) {
          throw new Error("같은 원문을 가진 용어가 이미 이 용어집에 있습니다.");
        }
        const state = get();
        const glossaries = state.glossaries.map((item) => item.id === glossaryId
          ? {
              ...item,
              entries: item.entries.map((entry) => entry.id === entryId ? updated : entry),
              updatedAt: Date.now(),
            }
          : item);
        mergeGlossaryEntries(glossaries, state.activeGlossaryIds);
        persistAndSet({ glossaries, activeGlossaryIds: state.activeGlossaryIds });
        return updated;
      },

      deleteEntry: (glossaryId, entryId) => {
        ensureHydrated();
        const glossary = findGlossary(glossaryId);
        if (!glossary.entries.some((entry) => entry.id === entryId)) return;
        const state = get();
        persistAndSet({
          glossaries: state.glossaries.map((item) => item.id === glossaryId
            ? {
                ...item,
                entries: item.entries.filter((entry) => entry.id !== entryId),
                updatedAt: Date.now(),
              }
            : item),
          activeGlossaryIds: state.activeGlossaryIds,
        });
      },

      importEntries: (glossaryId, incoming) => {
        ensureHydrated();
        const glossary = findGlossary(glossaryId);
        const result = upsertEntries(glossary.entries, incoming);
        const state = get();
        const nextTotal = countEntries(state.glossaries) - glossary.entries.length + result.entries.length;
        if (nextTotal > MAX_TOTAL_GLOSSARY_ENTRIES) {
          throw new Error(`전체 용어는 최대 ${MAX_TOTAL_GLOSSARY_ENTRIES}개까지 저장할 수 있습니다.`);
        }
        const glossaries = state.glossaries.map((item) => item.id === glossaryId
          ? { ...item, entries: result.entries, updatedAt: Date.now() }
          : item);
        mergeGlossaryEntries(glossaries, state.activeGlossaryIds);
        persistAndSet({ glossaries, activeGlossaryIds: state.activeGlossaryIds });
        return { inserted: result.inserted, updated: result.updated };
      },

      getActiveGlossaries: () => {
        const state = get();
        const byId = new Map(state.glossaries.map((glossary) => [glossary.id, glossary]));
        return state.activeGlossaryIds
          .map((id) => byId.get(id))
          .filter((glossary): glossary is Glossary => Boolean(glossary));
      },

      getActiveEntries: () => {
        const state = get();
        return mergeGlossaryEntries(state.glossaries, state.activeGlossaryIds);
      },
    };
  });
}

export const useGlossaryStore = createGlossaryStore();

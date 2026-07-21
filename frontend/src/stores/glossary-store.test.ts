import { beforeEach, describe, expect, it } from "vitest";
import {
  createGlossaryStore,
  GLOSSARY_RECOVERY_KEY,
  GLOSSARY_STORAGE_KEY,
  GLOSSARY_STORAGE_VERSION,
  parseGlossaryStorage,
} from "@/stores/glossary-store";

class MemoryStorage implements Storage {
  private data = new Map<string, string>();
  failWrites = false;

  get length(): number {
    return this.data.size;
  }

  clear(): void {
    this.data.clear();
  }

  getItem(key: string): string | null {
    return this.data.get(key) ?? null;
  }

  key(index: number): string | null {
    return Array.from(this.data.keys())[index] ?? null;
  }

  removeItem(key: string): void {
    this.data.delete(key);
  }

  setItem(key: string, value: string): void {
    if (this.failWrites) throw new DOMException("quota", "QuotaExceededError");
    this.data.set(key, value);
  }
}

describe("glossary storage migration", () => {
  it("migrates the legacy single active id into an ordered list", () => {
    const restored = parseGlossaryStorage(JSON.stringify({
      state: {
        glossaries: [{ id: "g1", name: "기존", entries: [], updatedAt: 1 }],
        activeGlossaryId: "g1",
      },
      version: 0,
    }));

    expect(restored.activeGlossaryIds).toEqual(["g1"]);
  });
});

describe("glossary store", () => {
  let storage: MemoryStorage;

  beforeEach(() => {
    storage = new MemoryStorage();
  });

  it("keeps creation separate from activation and auto-activates on a manual term add", () => {
    const store = createGlossaryStore(() => storage);
    store.getState().hydrate();
    const id = store.getState().createGlossary("제품");

    expect(store.getState().activeGlossaryIds).toEqual([]);
    const result = store.getState().addEntry(id, "API", "인터페이스");
    expect(result.activated).toBe(true);
    expect(store.getState().activeGlossaryIds).toEqual([id]);
  });

  it("preserves active priority and first-wins conflict resolution", () => {
    const store = createGlossaryStore(() => storage);
    store.getState().hydrate();
    const first = store.getState().createGlossary("첫째");
    const second = store.getState().createGlossary("둘째");
    store.getState().addEntry(first, "API", "첫 값");
    store.getState().addEntry(second, "api", "둘째 값");

    expect(store.getState().getActiveEntries()[0]?.target).toBe("첫 값");
    store.getState().moveActiveGlossary(second, -1);
    expect(store.getState().activeGlossaryIds).toEqual([second, first]);
    expect(store.getState().getActiveEntries()[0]?.target).toBe("둘째 값");
  });

  it("does not mutate in-memory state when localStorage rejects a write", () => {
    const store = createGlossaryStore(() => storage);
    store.getState().hydrate();
    storage.failWrites = true;

    expect(() => store.getState().createGlossary("저장 실패")).toThrow("저장 공간");
    expect(store.getState().glossaries).toEqual([]);
    expect(store.getState().storageWarning).toContain("저장 공간");
  });

  it("does not pretend to persist when browser storage is unavailable", () => {
    const store = createGlossaryStore(() => null);
    store.getState().hydrate();

    expect(() => store.getState().createGlossary("저장 불가")).toThrow("저장소");
    expect(store.getState().glossaries).toEqual([]);
  });

  it("keeps migrated data in memory when rewriting the legacy envelope fails", () => {
    storage.setItem(GLOSSARY_STORAGE_KEY, JSON.stringify({
      state: {
        glossaries: [{ id: "legacy", name: "기존", entries: [], updatedAt: 1 }],
        activeGlossaryId: "legacy",
      },
      version: 0,
    }));
    storage.failWrites = true;
    const store = createGlossaryStore(() => storage);
    store.getState().hydrate();

    expect(store.getState().glossaries[0]?.name).toBe("기존");
    expect(store.getState().activeGlossaryIds).toEqual(["legacy"]);
    expect(store.getState().storageWarning).toContain("변환하지 못했습니다");
  });

  it("backs up corrupt JSON and starts with an empty, usable library", () => {
    storage.setItem(GLOSSARY_STORAGE_KEY, "{broken");
    const store = createGlossaryStore(() => storage);
    store.getState().hydrate();

    expect(store.getState().glossaries).toEqual([]);
    expect(store.getState().storageWarning).toContain("복구용 사본");
    expect(storage.getItem(GLOSSARY_RECOVERY_KEY)).toBe("{broken");
    expect(storage.getItem(GLOSSARY_STORAGE_KEY)).toBeNull();

    const id = store.getState().createGlossary("복구 후");
    const envelope = JSON.parse(storage.getItem(GLOSSARY_STORAGE_KEY) ?? "{}") as { version: number };
    expect(id).toBeTruthy();
    expect(envelope.version).toBe(GLOSSARY_STORAGE_VERSION);
  });
});

/**
 * Client-side glossary helpers (flatten, CSV parse, id generation).
 */

export interface GlossaryEntry {
  id: string;
  source: string;
  target: string;
  notes?: string;
}

export interface Glossary {
  id: string;
  name: string;
  entries: GlossaryEntry[];
  updatedAt: number;
}

export const MAX_GLOSSARY_ENTRIES = 5000;
export const MAX_TERM_CHARS = 500;

const HEADER_SOURCE = new Set(["원문", "source", "소스", "용어", "term"]);
const HEADER_TARGET = new Set(["번역", "target", "타겟", "번역어", "translation"]);

export function createId(prefix: string): string {
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export function normalizeTerm(value: string): string {
  return value.trim();
}

export function isHeaderPair(source: string, target: string): boolean {
  return (
    HEADER_SOURCE.has(source.trim().toLocaleLowerCase()) &&
    HEADER_TARGET.has(target.trim().toLocaleLowerCase())
  );
}

/** Flatten active glossary entries to {source: target} for the job API. */
export function flattenGlossary(entries: GlossaryEntry[]): Record<string, string> {
  const map: Record<string, string> = {};
  for (const entry of entries) {
    const source = normalizeTerm(entry.source);
    const target = normalizeTerm(entry.target);
    if (!source || !target) continue;
    map[source] = target;
  }
  return map;
}

export function glossaryToJsonPayload(entries: GlossaryEntry[]): string | null {
  const map = flattenGlossary(entries);
  if (Object.keys(map).length === 0) return null;
  return JSON.stringify(map);
}

/**
 * Parse a simple CSV with source,target[,notes].
 * Supports quoted fields and skips a header row when present.
 */
export function parseGlossaryCsv(text: string): Omit<GlossaryEntry, "id">[] {
  const rows = parseCsvRows(text);
  const results: Omit<GlossaryEntry, "id">[] = [];
  let started = false;

  for (const cols of rows) {
    if (cols.length < 2) continue;
    const source = normalizeTerm(cols[0] ?? "");
    const target = normalizeTerm(cols[1] ?? "");
    const notes = normalizeTerm(cols[2] ?? "") || undefined;
    if (!source || !target) continue;
    if (!started && isHeaderPair(source, target)) {
      started = true;
      continue;
    }
    started = true;
    if (source.length > MAX_TERM_CHARS || target.length > MAX_TERM_CHARS) {
      throw new Error(`용어 길이는 ${MAX_TERM_CHARS}자를 초과할 수 없습니다.`);
    }
    results.push({ source, target, notes });
    if (results.length > MAX_GLOSSARY_ENTRIES) {
      throw new Error(`용어집 항목은 최대 ${MAX_GLOSSARY_ENTRIES}개까지 지원합니다.`);
    }
  }

  return results;
}

function parseCsvRows(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"') {
      inQuotes = true;
    } else if (ch === "," || ch === "\t") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      field = "";
      if (row.some((cell) => cell.trim())) rows.push(row);
      row = [];
    } else if (ch === "\r") {
      // ignore CR; LF handles the break
    } else {
      field += ch;
    }
  }

  row.push(field);
  if (row.some((cell) => cell.trim())) rows.push(row);
  return rows;
}

/** Upsert imported pairs into an entry list (match by exact trimmed source). */
export function upsertEntries(
  existing: GlossaryEntry[],
  incoming: Omit<GlossaryEntry, "id">[]
): { entries: GlossaryEntry[]; inserted: number; updated: number } {
  const bySource = new Map<string, GlossaryEntry>();
  for (const entry of existing) {
    bySource.set(normalizeTerm(entry.source), entry);
  }

  let inserted = 0;
  let updated = 0;
  for (const item of incoming) {
    const source = normalizeTerm(item.source);
    const target = normalizeTerm(item.target);
    if (!source || !target) continue;
    const prev = bySource.get(source);
    if (prev) {
      bySource.set(source, {
        ...prev,
        source,
        target,
        notes: item.notes ?? prev.notes,
      });
      updated++;
    } else {
      bySource.set(source, {
        id: createId("term"),
        source,
        target,
        notes: item.notes,
      });
      inserted++;
    }
  }

  if (bySource.size > MAX_GLOSSARY_ENTRIES) {
    throw new Error(`용어집 항목은 최대 ${MAX_GLOSSARY_ENTRIES}개까지 지원합니다.`);
  }

  return {
    entries: Array.from(bySource.values()),
    inserted,
    updated,
  };
}

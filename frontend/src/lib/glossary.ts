/**
 * Client-side glossary helpers.
 *
 * Glossaries are stored locally, while only the ordered, active entries are
 * snapshotted and sent to a translation job.
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
export const MAX_TOTAL_GLOSSARY_ENTRIES = 10000;
export const MAX_TERM_CHARS = 500;
export const MAX_NOTES_CHARS = 2000;
export const MAX_GLOSSARY_NAME_CHARS = 100;
export const MAX_GLOSSARY_JSON_CHARS = 1_000_000;
export const MAX_GLOSSARY_STORAGE_BYTES = 3_500_000;

const HEADER_SOURCE = new Set(["원문", "source", "소스", "용어", "term"]);
const HEADER_TARGET = new Set(["번역", "target", "타겟", "번역어", "translation"]);
const SPREADSHEET_FORMULA_PREFIX = /^[=+\-@]/;

export function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}_${crypto.randomUUID()}`;
  }
  return `${prefix}_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

export function normalizeTerm(value: string): string {
  return value.trim().normalize("NFC");
}

/** Case-insensitive key used for duplicate/conflict resolution. */
export function glossaryTermKey(value: string): string {
  return normalizeTerm(value).normalize("NFKC").toLowerCase();
}

export function normalizeNotes(value?: string): string | undefined {
  const normalized = value?.trim().normalize("NFC");
  return normalized || undefined;
}

export function validateGlossaryName(value: string): string {
  const name = value.trim().normalize("NFC");
  if (!name) throw new Error("용어집 이름을 입력해주세요.");
  if (name.length > MAX_GLOSSARY_NAME_CHARS) {
    throw new Error(`용어집 이름은 ${MAX_GLOSSARY_NAME_CHARS}자를 초과할 수 없습니다.`);
  }
  return name;
}

export function validateGlossaryEntry(
  entry: Pick<GlossaryEntry, "source" | "target" | "notes">
): Omit<GlossaryEntry, "id"> {
  const source = normalizeTerm(entry.source);
  const target = normalizeTerm(entry.target);
  const notes = normalizeNotes(entry.notes);
  if (!source || !target) throw new Error("원문과 번역을 모두 입력해주세요.");
  if (source.length > MAX_TERM_CHARS || target.length > MAX_TERM_CHARS) {
    throw new Error(`용어 길이는 ${MAX_TERM_CHARS}자를 초과할 수 없습니다.`);
  }
  if (notes && notes.length > MAX_NOTES_CHARS) {
    throw new Error(`메모는 ${MAX_NOTES_CHARS}자를 초과할 수 없습니다.`);
  }
  return { source, target, notes };
}

export function isHeaderPair(source: string, target: string): boolean {
  return (
    HEADER_SOURCE.has(source.replace(/^\uFEFF/, "").trim().toLowerCase()) &&
    HEADER_TARGET.has(target.trim().toLowerCase())
  );
}

/**
 * Merge active glossaries in priority order. The first glossary wins when the
 * same normalized source exists in more than one glossary.
 */
export function mergeGlossaryEntries(
  glossaries: Glossary[],
  activeGlossaryIds: string[]
): GlossaryEntry[] {
  const byId = new Map(glossaries.map((glossary) => [glossary.id, glossary]));
  const seen = new Set<string>();
  const merged: GlossaryEntry[] = [];

  for (const glossaryId of activeGlossaryIds) {
    const glossary = byId.get(glossaryId);
    if (!glossary) continue;
    for (const entry of glossary.entries) {
      const key = glossaryTermKey(entry.source);
      if (!key || seen.has(key)) continue;
      seen.add(key);
      merged.push(entry);
      if (merged.length > MAX_GLOSSARY_ENTRIES) {
        throw new Error(
          `이번 번역에 적용할 용어는 최대 ${MAX_GLOSSARY_ENTRIES}개까지 지원합니다.`
        );
      }
    }
  }
  return merged;
}

/** Flatten entries to {source: target} for the existing job API. */
export function flattenGlossary(entries: GlossaryEntry[]): Record<string, string> {
  const map: Record<string, string> = {};
  for (const entry of entries) {
    const { source, target } = validateGlossaryEntry(entry);
    map[source] = target;
  }
  return map;
}

export function glossaryToJsonPayload(entries: GlossaryEntry[]): string | null {
  if (entries.length > MAX_GLOSSARY_ENTRIES) {
    throw new Error(`용어집 항목은 최대 ${MAX_GLOSSARY_ENTRIES}개까지 적용할 수 있습니다.`);
  }
  const map = flattenGlossary(entries);
  if (Object.keys(map).length === 0) return null;
  const payload = JSON.stringify(map);
  if (payload.length > MAX_GLOSSARY_JSON_CHARS) {
    throw new Error("선택한 용어집이 번역 전송 한도를 초과합니다. 용어 수나 길이를 줄여주세요.");
  }
  return payload;
}

function detectDelimiter(text: string): "," | "\t" | ";" {
  const counts = { ",": 0, "\t": 0, ";": 0 };
  let inQuotes = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (ch === '"') {
      if (inQuotes && text[i + 1] === '"') {
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }
    if (!inQuotes && (ch === "\n" || ch === "\r")) break;
    if (!inQuotes && (ch === "," || ch === "\t" || ch === ";")) counts[ch] += 1;
  }
  return (Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] as "," | "\t" | ";") || ",";
}

function parseCsvRows(text: string, delimiter: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let inQuotes = false;

  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i += 1;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
      continue;
    }

    if (ch === '"' && field.length === 0) {
      inQuotes = true;
    } else if (ch === delimiter) {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field);
      field = "";
      if (row.some((cell) => cell.trim())) rows.push(row);
      row = [];
    } else if (ch !== "\r") {
      field += ch;
    }
  }

  if (inQuotes) throw new Error("CSV 따옴표가 올바르게 닫히지 않았습니다.");
  row.push(field);
  if (row.some((cell) => cell.trim())) rows.push(row);
  return rows;
}

/** Parse CSV/TSV with source,target[,notes], including quoted newlines. */
export function parseGlossaryCsv(text: string): Omit<GlossaryEntry, "id">[] {
  const normalizedText = text.replace(/^\uFEFF/, "");
  const rows = parseCsvRows(normalizedText, detectDelimiter(normalizedText));
  const results: Omit<GlossaryEntry, "id">[] = [];
  let started = false;

  for (const cols of rows) {
    if (cols.length < 2) continue;
    const rawSource = normalizeTerm(cols[0] ?? "");
    const rawTarget = normalizeTerm(cols[1] ?? "");
    if (!rawSource || !rawTarget) continue;
    const candidate = validateGlossaryEntry({
      source: rawSource,
      target: rawTarget,
      notes: cols[2],
    });
    if (!started && isHeaderPair(candidate.source, candidate.target)) {
      started = true;
      continue;
    }
    started = true;
    results.push(candidate);
    if (results.length > MAX_GLOSSARY_ENTRIES) {
      throw new Error(`용어집 항목은 최대 ${MAX_GLOSSARY_ENTRIES}개까지 지원합니다.`);
    }
  }

  return results;
}

/** Upsert imported pairs (case-insensitive normalized source, last value wins). */
export function upsertEntries(
  existing: GlossaryEntry[],
  incoming: Omit<GlossaryEntry, "id">[]
): { entries: GlossaryEntry[]; inserted: number; updated: number } {
  const bySource = new Map<string, GlossaryEntry>();
  for (const entry of existing) {
    const normalized = validateGlossaryEntry(entry);
    bySource.set(glossaryTermKey(normalized.source), { ...entry, ...normalized });
  }

  let inserted = 0;
  let updated = 0;
  for (const item of incoming) {
    const normalized = validateGlossaryEntry(item);
    const key = glossaryTermKey(normalized.source);
    const previous = bySource.get(key);
    if (previous) {
      bySource.set(key, {
        ...previous,
        ...normalized,
        notes: normalized.notes ?? previous.notes,
      });
      updated += 1;
    } else {
      bySource.set(key, { id: createId("term"), ...normalized });
      inserted += 1;
    }
  }

  if (bySource.size > MAX_GLOSSARY_ENTRIES) {
    throw new Error(`용어집 항목은 최대 ${MAX_GLOSSARY_ENTRIES}개까지 지원합니다.`);
  }
  return { entries: Array.from(bySource.values()), inserted, updated };
}

function spreadsheetSafeCell(value: string): string {
  return SPREADSHEET_FORMULA_PREFIX.test(value) ? `\t${value}` : value;
}

function csvCell(value: string): string {
  const safe = spreadsheetSafeCell(value);
  return `"${safe.replaceAll('"', '""')}"`;
}

export function glossaryToCsv(entries: GlossaryEntry[]): string {
  const lines = ["원문,번역,메모"];
  for (const entry of entries) {
    const normalized = validateGlossaryEntry(entry);
    lines.push([
      csvCell(normalized.source),
      csvCell(normalized.target),
      csvCell(normalized.notes ?? ""),
    ].join(","));
  }
  return `\uFEFF${lines.join("\r\n")}\r\n`;
}

export function safeGlossaryFilename(name: string): string {
  const cleaned = name.trim().replace(/[\\/:*?"<>|]+/g, "_").slice(0, 80);
  return cleaned || "glossary";
}

export function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

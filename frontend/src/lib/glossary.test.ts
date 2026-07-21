import { describe, expect, it } from "vitest";
import {
  glossaryTermKey,
  glossaryToCsv,
  glossaryToJsonPayload,
  mergeGlossaryEntries,
  parseGlossaryCsv,
  upsertEntries,
  type Glossary,
  type GlossaryEntry,
} from "@/lib/glossary";

function entry(id: string, source: string, target: string): GlossaryEntry {
  return { id, source, target };
}

function glossary(id: string, entries: GlossaryEntry[]): Glossary {
  return { id, name: id, entries, updatedAt: 1 };
}

describe("parseGlossaryCsv", () => {
  it("handles a BOM, semicolon delimiter, escaped quotes, and quoted newlines", () => {
    const parsed = parseGlossaryCsv(
      '\uFEFF원문;번역;메모\r\n"Alpha";"알파";"첫 줄\n둘째 줄"\r\n"A""B";"값";""\r\n'
    );

    expect(parsed).toEqual([
      { source: "Alpha", target: "알파", notes: "첫 줄\n둘째 줄" },
      { source: 'A"B', target: "값", notes: undefined },
    ]);
  });

  it("round-trips spreadsheet-formula-looking cells without exporting formulas", () => {
    const csv = glossaryToCsv([
      { id: "1", source: "=SUM(A1:A2)", target: "+danger", notes: "@note" },
    ]);

    expect(csv).toContain('"\t=SUM(A1:A2)"');
    expect(parseGlossaryCsv(csv)).toEqual([
      { source: "=SUM(A1:A2)", target: "+danger", notes: "@note" },
    ]);
  });
});

describe("glossary merging", () => {
  it("normalizes Unicode and case for imports while the last imported value wins", () => {
    const result = upsertEntries(
      [entry("old", "ＡＰＩ", "기존")],
      [{ source: "api", target: "새 값" }]
    );

    expect(glossaryTermKey("ＡＰＩ")).toBe(glossaryTermKey("api"));
    expect(result.inserted).toBe(0);
    expect(result.updated).toBe(1);
    expect(result.entries).toEqual([{ id: "old", source: "api", target: "새 값", notes: undefined }]);
  });

  it("uses active glossary order as the duplicate-source priority", () => {
    const glossaries = [
      glossary("first", [entry("1", "API", "첫 번째"), entry("2", "SDK", "키트")]),
      glossary("second", [entry("3", "api", "두 번째"), entry("4", "CLI", "도구")]),
    ];

    expect(mergeGlossaryEntries(glossaries, ["first", "second"]).map((item) => item.target))
      .toEqual(["첫 번째", "키트", "도구"]);
    expect(mergeGlossaryEntries(glossaries, ["second", "first"]).map((item) => item.target))
      .toEqual(["두 번째", "도구", "키트"]);
  });

  it("produces the existing backend JSON contract", () => {
    expect(glossaryToJsonPayload([entry("1", "API", "인터페이스")]))
      .toBe('{"API":"인터페이스"}');
  });
});

import { describe, expect, it } from "vitest";
import {
  buildBlocks,
  buildQueue,
  focusedKey,
  initialQueueState,
  isReviewComplete,
  lastAction,
  primaryFinding,
  queueReducer,
  remainingCount,
  suggestFix,
  type QueueAction,
  type QueueState,
  type ReviewLogEntry,
  type ReviewOutcome,
} from "@/lib/review-queue";
import type { FragmentFinding, FragmentItem } from "@/types/api";

function fragment(overrides: Partial<FragmentItem> & { index: number }): FragmentItem {
  return {
    slide: 1,
    shape: 0,
    paragraph: overrides.index,
    slide_title: null,
    is_note: false,
    source: "source",
    target: "번역",
    repeat_count: 1,
    length_budget: null,
    findings: [],
    edited: false,
    style_segments: [],
    style_status: "single_style",
    container_id: "s0/sh0",
    container_kind: "textbox",
    ...overrides,
  };
}

function finding(overrides: Partial<FragmentFinding> = {}): FragmentFinding {
  return {
    type: "terminology.violation",
    severity: "major",
    description: "용어집과 다릅니다.",
    suggested_fix: null,
    term_source: null,
    related_location: null,
    ...overrides,
  };
}

describe("buildBlocks", () => {
  it("merges the consecutive paragraphs of a hand-wrapped sentence", () => {
    const blocks = buildBlocks([
      fragment({ index: 0, paragraph: 0, source: "A sentence that keeps" }),
      fragment({ index: 1, paragraph: 1, source: "going on the next line" }),
    ]);

    expect(blocks).toHaveLength(1);
    expect(blocks[0].key).toBe("s0/sh0#0");
    expect(blocks[0].items.map((item) => item.index)).toEqual([0, 1]);
  });

  it("keeps bullet points and speaker notes as separate items", () => {
    const bullets = buildBlocks([
      fragment({ index: 0, paragraph: 0, container_kind: "body", source: "a bullet that keeps" }),
      fragment({ index: 1, paragraph: 1, container_kind: "body", source: "going on the next line" }),
    ]);
    const notes = buildBlocks([
      fragment({ index: 0, paragraph: 0, container_id: "s0/notes", container_kind: "notes", is_note: true }),
      fragment({ index: 1, paragraph: 1, container_id: "s0/notes", container_kind: "notes", is_note: true }),
    ]);

    expect(bullets).toHaveLength(2);
    expect(notes).toHaveLength(2);
  });

  it("treats a skipped paragraph index as a separator", () => {
    // An empty paragraph between the two is dropped during extraction, so the
    // gap in paragraph_index is the only trace it left.
    const blocks = buildBlocks([
      fragment({ index: 0, paragraph: 0, source: "a sentence that keeps" }),
      fragment({ index: 1, paragraph: 2, source: "going on the next line" }),
    ]);

    expect(blocks).toHaveLength(2);
  });

  it("merges across a comma, where letter case cannot help", () => {
    // The only continuation signal that survives into Korean: 개조식 list lines
    // close on a noun or `~함`, so a trailing comma is a deliberate break.
    const blocks = buildBlocks([
      fragment({ index: 0, paragraph: 0, source: "발전속도가 매우 빠르니," }),
      fragment({ index: 1, paragraph: 1, source: "한번 써봐도 좋을 것 같다." }),
    ]);

    expect(blocks).toHaveLength(1);
  });

  it("keeps a heading, a label/value pair, and a text-box bullet list apart", () => {
    // Measured on real decks: these are what plain text boxes are actually
    // full of, and only the continuation's case tells them from a wrapped
    // sentence. Scripts without letter case therefore never merge.
    const blocks = buildBlocks([
      fragment({ index: 0, paragraph: 0, source: "UIUX" }),
      fragment({ index: 1, paragraph: 1, source: "Match the military characteristics" }),
      fragment({ index: 2, paragraph: 2, source: "Squad Size" }),
      fragment({ index: 3, paragraph: 3, source: "4 players" }),
      fragment({ index: 4, paragraph: 4, source: "안정화 빌드" }),
      fragment({ index: 5, paragraph: 5, source: "코어 플레이 완성" }),
    ]);

    expect(blocks).toHaveLength(6);
  });

  it("stops merging after a sentence-ending punctuation", () => {
    const blocks = buildBlocks([
      fragment({ index: 0, paragraph: 0, source: "First sentence." }),
      fragment({ index: 1, paragraph: 1, source: "Second sentence" }),
    ]);

    expect(blocks.map((block) => block.items.length)).toEqual([1, 1]);
  });

  it("caps a block at four paragraphs", () => {
    const blocks = buildBlocks(
      [0, 1, 2, 3, 4, 5].map((n) => fragment({ index: n, paragraph: n, source: `line ${n}` }))
    );

    expect(blocks.map((block) => block.items.length)).toEqual([4, 2]);
  });

  it("never merges across table cells that share a shape index", () => {
    // Every cell reuses the table's shape_index and restarts paragraph_index,
    // so only container_id keeps them apart.
    const blocks = buildBlocks([
      fragment({ index: 0, paragraph: 0, container_id: "s0/sh2/r0c0", container_kind: "table_cell", source: "셀 하나" }),
      fragment({ index: 1, paragraph: 0, container_id: "s0/sh2/r0c1", container_kind: "table_cell", source: "셀 둘" }),
    ]);

    expect(blocks).toHaveLength(2);
  });
});

describe("buildQueue", () => {
  const critical = finding({ type: "accuracy.omission", severity: "critical" });
  const minor = finding({ type: "style.mapping_dropped", severity: "minor" });

  it("keeps only flagged blocks, worst first, then deck order", () => {
    const queue = buildQueue(
      [
        fragment({ index: 0, slide: 1, container_id: "s0/sh0", findings: [minor] }),
        fragment({ index: 1, slide: 1, container_id: "s0/sh1" }),
        fragment({ index: 2, slide: 2, container_id: "s1/sh0", findings: [critical] }),
        fragment({ index: 3, slide: 3, container_id: "s2/sh0", findings: [finding()] }),
      ],
      {}
    );

    expect(queue.map((block) => block.items[0].index)).toEqual([2, 3, 0]);
  });

  it("keeps a handled block in the queue after its findings clear", () => {
    const fragments = [fragment({ index: 0, container_id: "s0/sh0" })];

    expect(buildQueue(fragments, {})).toHaveLength(0);
    expect(buildQueue(fragments, { "s0/sh0#0": "applied" })).toHaveLength(1);
  });

  it("takes in an unflagged block the user opened from the full list", () => {
    const fragments = [fragment({ index: 0, container_id: "s0/sh0" })];

    expect(buildQueue(fragments, {}, ["s0/sh0#0"])).toHaveLength(1);
  });

  it("speaks about the worst finding in the block", () => {
    const [block] = buildBlocks([
      fragment({ index: 0, paragraph: 0, source: "wrapped", findings: [minor] }),
      fragment({ index: 1, paragraph: 1, source: "sentence", findings: [critical] }),
    ]);

    expect(primaryFinding(block)).toEqual({ index: 1, finding: critical });
  });
});

describe("suggestFix", () => {
  it("replaces a source term the translator left untranslated", () => {
    const suggestion = suggestFix(
      "Battle Pass를 구매하세요",
      finding({ suggested_fix: "배틀 패스", term_source: "Battle Pass" })
    );

    expect(suggestion?.target).toBe("배틀 패스를 구매하세요");
    expect(suggestion?.confidence).toBe("certain");
    expect(suggestion?.replaced).toEqual({ start: 0, end: 11 });
    expect(suggestion?.span).toEqual({ start: 0, end: 5 });
  });

  it("fixes a case or spacing variant of a locked term", () => {
    const suggestion = suggestFix(
      "Buy the battlepass now",
      finding({ suggested_fix: "Battle Pass" })
    );

    expect(suggestion?.target).toBe("Buy the Battle Pass now");
    expect(suggestion?.confidence).toBe("certain");
  });

  it("guesses a near-miss phrase and labels it as one", () => {
    const suggestion = suggestFix(
      "Buy the Combat Pass now",
      finding({ suggested_fix: "Battle Pass", term_source: "Battle Pass" })
    );

    expect(suggestion?.target).toBe("Buy the Battle Pass now");
    expect(suggestion?.confidence).toBe("estimated");
    expect(suggestion?.basis).toContain("추정");
  });

  it("guesses through a particle glued onto the term", () => {
    const suggestion = suggestFix(
      "컴뱃 패스를 구매하세요",
      finding({ suggested_fix: "배틀 패스", term_source: "Battle Pass" })
    );

    expect(suggestion?.target).toBe("배틀 패스를 구매하세요");
    expect(suggestion?.confidence).toBe("estimated");
  });

  it("offers nothing when the wrong wording cannot be located", () => {
    expect(
      suggestFix("전혀 다른 문장입니다", finding({ suggested_fix: "배틀 패스" }))
    ).toBeNull();
    // The required term is already there — the finding is stale.
    expect(
      suggestFix("배틀 패스 구매", finding({ suggested_fix: "배틀 패스" }))
    ).toBeNull();
  });

  it("offers nothing for findings that carry no replacement wording", () => {
    expect(
      suggestFix(
        "너무 긴 번역문",
        finding({ type: "fit.overflow", suggested_fix: null })
      )
    ).toBeNull();
    expect(
      suggestFix(
        "번역이 빠졌습니다",
        finding({ type: "accuracy.omission", severity: "critical" })
      )
    ).toBeNull();
  });
});

describe("queueReducer", () => {
  const KEYS = ["a", "b", "c"];

  function reduce(state: QueueState, ...actions: QueueAction[]): QueueState {
    return actions.reduce(queueReducer, state);
  }

  const synced = reduce(initialQueueState, { type: "sync", keys: KEYS });

  it("keeps the cursor on the same item when the list is reloaded", () => {
    const moved = reduce(synced, { type: "move", delta: 1 });
    // "a" was fixed elsewhere and dropped out; "b" must stay under the cursor.
    const reloaded = reduce(moved, { type: "sync", keys: ["b", "c"] });

    expect(focusedKey(reloaded)).toBe("b");
    expect(reloaded.order).toEqual(["b", "c"]);
  });

  it("appends newly surfaced findings instead of reshuffling", () => {
    const reloaded = reduce(synced, { type: "sync", keys: ["d", "c", "b", "a"] });

    expect(reloaded.order).toEqual(["a", "b", "c", "d"]);
    expect(focusedKey(reloaded)).toBe("a");
  });

  it("clamps navigation to the queue bounds", () => {
    expect(reduce(synced, { type: "move", delta: -1 }).cursor).toBe(0);
    expect(reduce(synced, { type: "move", delta: 9 }).cursor).toBe(2);
  });

  it("moves to the next unhandled item after an apply", () => {
    const applied = reduce(synced, {
      type: "resolve",
      entry: { kind: "edit", keys: ["a"], revision: 4 },
    });

    expect(focusedKey(applied)).toBe("b");
    expect(applied.resolved).toEqual({ a: "applied" } satisfies Record<string, ReviewOutcome>);
    expect(lastAction(applied)).toEqual({ kind: "edit", keys: ["a"], revision: 4 });
  });

  it("skips over items that were already handled", () => {
    const state = reduce(
      synced,
      { type: "resolve", entry: { kind: "dismiss", keys: ["b"], entries: [] } },
      { type: "focus", key: "a" },
      { type: "resolve", entry: { kind: "edit", keys: ["a"], revision: 5 } }
    );

    expect(focusedKey(state)).toBe("c");
  });

  it("undoes a bulk skip in one step", () => {
    const skipped = reduce(synced, {
      type: "resolve",
      entry: {
        kind: "dismiss",
        keys: KEYS,
        entries: KEYS.map((_, index) => ({ index, finding_type: "fit.overflow" })),
      },
    });
    expect(isReviewComplete(skipped)).toBe(true);
    expect(remainingCount(skipped)).toBe(0);

    const restored = reduce(skipped, { type: "undo" });

    expect(restored.resolved).toEqual({});
    expect(restored.log).toEqual([]);
    expect(focusedKey(restored)).toBe("a");
  });

  it("takes back a failed optimistic apply without disturbing later ones", () => {
    // The cursor advances before the server answers, so by the time an apply
    // fails the user may already have handled the next item.
    const failed: ReviewLogEntry = { kind: "edit", keys: ["a"], revision: 3 };
    const later: ReviewLogEntry = { kind: "dismiss", keys: ["b"], entries: [] };
    const state = reduce(
      synced,
      { type: "resolve", entry: failed },
      { type: "resolve", entry: later },
      { type: "rollback", entry: failed }
    );

    expect(state.resolved).toEqual({ b: "skipped" });
    expect(state.log).toEqual([later]);
    expect(focusedKey(state)).toBe("a");
  });

  it("focuses a block pulled in from the full list once it joins the queue", () => {
    // A pinned block has no findings, so it only enters `order` on the next
    // sync — the focus has to wait for it.
    const pinned = reduce(synced, { type: "pin", key: "d" });
    expect(pinned.pinned).toEqual(["d"]);
    expect(pinned.mode).toBe("queue");

    const state = reduce(pinned, { type: "sync", keys: [...KEYS, "d"] });

    expect(focusedKey(state)).toBe("d");
    expect(state.pendingFocus).toBeNull();
  });

  it("just moves the cursor when the pinned block is already queued", () => {
    const state = reduce(synced, { type: "pin", key: "c" });

    expect(state.pinned).toEqual([]);
    expect(focusedKey(state)).toBe("c");
  });

  it("leaves the state alone when there is nothing to undo", () => {
    expect(reduce(synced, { type: "undo" })).toBe(synced);
  });

  it("closes the editor whenever the item changes", () => {
    const editing = reduce(synced, { type: "editor", editor: "manual" });

    expect(editing.editor).toBe("manual");
    expect(reduce(editing, { type: "move", delta: 1 }).editor).toBe("none");
    expect(reduce(editing, { type: "focus", key: "c" }).editor).toBe("none");
    expect(
      reduce(editing, {
        type: "resolve",
        entry: { kind: "edit", keys: ["a"], revision: 1 },
      }).editor
    ).toBe("none");
  });
});

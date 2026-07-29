/**
 * Pure helpers for the review queue (docs/REVIEW_QUEUE_PLAN.md Step 4).
 *
 * Everything the queue screen decides on its own lives here: which paragraphs
 * read as one item, what order they are handled in, what the recommended fix
 * says, and how the cursor moves. The server does not know what a block is —
 * it only serves `container_id` / `container_kind` and a bulk apply endpoint —
 * so these rules stay in TypeScript where vitest can cover them and the
 * thresholds can be tuned without a Python round trip.
 */

import type {
  ContainerKind,
  FragmentFinding,
  FragmentItem,
  ReviewDismissalEntry,
} from "@/types/api";
import { glossaryTermKey } from "@/lib/glossary";

// --- block merging (§3.2) ---------------------------------------------------

/**
 * Several paragraphs in a bullet list are separate points, not one sentence;
 * speaker notes are prose where each paragraph stands alone.
 */
export const NON_MERGING_KINDS: ReadonlySet<ContainerKind> = new Set([
  "body",
  "notes",
]);

/** A longer run is a list of some kind, whatever the container claims. */
export const MAX_MERGE_PARAGRAPHS = 4;

/** If the previous line already closed the sentence, the next one is new. */
const SENTENCE_END = /[.!?。！？…:;]['"”’)\]]*$/;

/**
 * The next line only continues the sentence when it starts mid-sentence — a
 * lowercase letter. Measured over four real decks (377 candidate merges,
 * `scripts/analyze_fragments.py`): every other shape was a heading with its
 * description, a label/value pair, or a bullet list living in a plain text
 * box. This signal kept all 7 genuine hard-wrapped sentences and rejected all
 * 370 wrong merges.
 *
 * Scripts without letter case (Korean, Japanese, Chinese) therefore never
 * merge. The measured decks held no genuine wrap in those scripts to calibrate
 * a replacement signal against, and one review item per paragraph — today's
 * behaviour — is the safe side to fail to.
 */
const CONTINUES_SENTENCE = /^\p{Ll}/u;

/**
 * A line broken on a comma continues whatever the next line looks like. This
 * is the one signal that survives into scripts without letter case: Korean
 * decks write their lists 개조식 (each line closing on a noun or `~함`), so a
 * trailing comma is a deliberate mid-clause break, not a list item.
 */
const ENDS_MID_CLAUSE = /[,،、]$/;

/** Paragraphs the queue shows as a single item. */
export interface ReviewBlock {
  /**
   * Stable for the session: paragraphs are fixed once the deck is parsed, so
   * the container path plus the first paragraph's index never shifts.
   */
  key: string;
  containerId: string;
  kind: ContainerKind;
  slide: number;
  items: FragmentItem[];
}

/**
 * Mirrors `_merge_block` in `scripts/analyze_fragments.py` so thresholds
 * measured against real decks transfer here unchanged.
 */
function mergesWithPrevious(
  previous: FragmentItem,
  current: FragmentItem,
  size: number
): boolean {
  if (current.container_id !== previous.container_id) return false;
  // An empty paragraph between the two reads as a separator.
  if (current.paragraph !== previous.paragraph + 1) return false;
  if (NON_MERGING_KINDS.has(current.container_kind)) return false;
  if (size >= MAX_MERGE_PARAGRAPHS) return false;
  const previousText = previous.source.trim();
  if (
    !CONTINUES_SENTENCE.test(current.source.trimStart()) &&
    !ENDS_MID_CLAUSE.test(previousText)
  ) {
    return false;
  }
  return !SENTENCE_END.test(previousText);
}

export function buildBlocks(fragments: readonly FragmentItem[]): ReviewBlock[] {
  const blocks: ReviewBlock[] = [];
  for (const fragment of fragments) {
    const current = blocks[blocks.length - 1];
    if (
      current &&
      mergesWithPrevious(
        current.items[current.items.length - 1],
        fragment,
        current.items.length
      )
    ) {
      current.items.push(fragment);
      continue;
    }
    blocks.push({
      key: `${fragment.container_id}#${fragment.index}`,
      containerId: fragment.container_id,
      kind: fragment.container_kind,
      slide: fragment.slide,
      items: [fragment],
    });
  }
  return blocks;
}

// --- queue ordering ---------------------------------------------------------

/** A finding plus the fragment inside the block that carries it. */
export interface BlockFinding {
  index: number;
  finding: FragmentFinding;
}

const SEVERITY_RANK: Record<string, number> = { critical: 0, major: 1, minor: 2 };
const NO_FINDING_RANK = 3;

export function blockFindings(block: ReviewBlock): BlockFinding[] {
  return block.items.flatMap((item) =>
    item.findings.map((finding) => ({ index: item.index, finding }))
  );
}

/** The finding the queue card speaks about: worst severity, else the first. */
export function primaryFinding(block: ReviewBlock): BlockFinding | null {
  let best: BlockFinding | null = null;
  let bestRank = NO_FINDING_RANK;
  for (const candidate of blockFindings(block)) {
    const rank = SEVERITY_RANK[candidate.finding.severity] ?? NO_FINDING_RANK;
    if (rank < bestRank) {
      best = candidate;
      bestRank = rank;
    }
  }
  return best;
}

function blockRank(block: ReviewBlock): number {
  const primary = primaryFinding(block);
  return primary ? SEVERITY_RANK[primary.finding.severity] ?? NO_FINDING_RANK : NO_FINDING_RANK;
}

/** Worst problems first, then deck order. */
export function compareBlocks(a: ReviewBlock, b: ReviewBlock): number {
  return (
    blockRank(a) - blockRank(b) ||
    a.slide - b.slide ||
    a.items[0].index - b.items[0].index
  );
}

/**
 * Blocks that need a decision, plus the ones already handled — a processed
 * item stays in the queue so `이전` can walk back to it. `queueReducer` freezes
 * the order it is first given, so a block that loses its findings does not
 * jump to the end mid-session.
 */
export function buildQueue(
  fragments: readonly FragmentItem[],
  resolved: ResolvedMap
): ReviewBlock[] {
  return buildBlocks(fragments)
    .filter((block) => blockFindings(block).length > 0 || block.key in resolved)
    .sort(compareBlocks);
}

// --- recommended fix (D-1) --------------------------------------------------

export const FIX_BASIS_GLOSSARY = "용어집 기준";
export const FIX_BASIS_GLOSSARY_GUESS = "용어집 기준 (추정)";

/** Bigram overlap a guessed span must reach before it is offered at all. */
export const MIN_FUZZY_SIMILARITY = 0.5;

export type FixConfidence = "certain" | "estimated";

export interface FixSuggestion {
  /** The whole proposed target — what `적용하고 다음` sends. */
  target: string;
  /** Span inside `target` that changed, for highlighting the new wording. */
  span: { start: number; end: number };
  /** Span inside the current target it replaces, for highlighting the problem. */
  replaced: { start: number; end: number };
  confidence: FixConfidence;
  /** Label shown beside `추천 수정`. */
  basis: string;
}

/**
 * `suggested_fix` is the wording that *should be there*, not the wording to
 * replace — the sweep only knows the required term is missing, never where the
 * wrong one sits. So a card is only offered when the wrong span can be located:
 * an untranslated source term, a case/space variant of the required term, or a
 * close-enough neighbouring phrase (labelled as a guess). Everything else —
 * `fit.*`, `accuracy.omission`, `consistency.phrase` — returns null and the
 * caller promotes `AI에게 다시 맡기기` instead.
 */
export function suggestFix(
  target: string,
  finding: FragmentFinding
): FixSuggestion | null {
  if (!finding.type.startsWith("terminology.")) return null;
  const fix = finding.suggested_fix?.trim();
  if (!fix || !target) return null;
  // Already worded correctly — nothing to propose.
  if (target.includes(fix)) return null;

  const source = finding.term_source?.trim();
  if (source) {
    const at = indexOfIgnoreCase(target, source);
    if (at >= 0) {
      return replaceSpan(target, at, at + source.length, fix, "certain", FIX_BASIS_GLOSSARY);
    }
  }

  const variant = findVariant(target, fix);
  if (variant) {
    return replaceSpan(target, variant.start, variant.end, fix, "certain", FIX_BASIS_GLOSSARY);
  }

  const guess = findSimilarSpan(target, fix);
  if (guess) {
    return replaceSpan(
      target,
      guess.start,
      guess.end,
      fix,
      "estimated",
      FIX_BASIS_GLOSSARY_GUESS
    );
  }
  return null;
}

function replaceSpan(
  target: string,
  start: number,
  end: number,
  replacement: string,
  confidence: FixConfidence,
  basis: string
): FixSuggestion {
  return {
    target: target.slice(0, start) + replacement + target.slice(end),
    span: { start, end: start + replacement.length },
    replaced: { start, end },
    confidence,
    basis,
  };
}

/** NFKC + lowercase (shared with the glossary) plus collapsed whitespace. */
function compareKey(value: string): string {
  return glossaryTermKey(value).replace(/\s+/g, " ");
}

function indexOfIgnoreCase(haystack: string, needle: string): number {
  return haystack.toLowerCase().indexOf(needle.toLowerCase());
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** The required term written with different case or spacing. */
function findVariant(
  target: string,
  fix: string
): { start: number; end: number } | null {
  const pattern = fix
    .trim()
    .split(/\s+/)
    .map(escapeRegExp)
    .join("\\s*");
  const match = new RegExp(pattern, "i").exec(target);
  return match ? { start: match.index, end: match.index + match[0].length } : null;
}

interface Token {
  text: string;
  start: number;
  end: number;
}

function tokenize(value: string): Token[] {
  const tokens: Token[] = [];
  const pattern = /\S+/g;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(value)) !== null) {
    tokens.push({ text: match[0], start: match.index, end: match.index + match[0].length });
  }
  return tokens;
}

function bigrams(value: string): string[] {
  const grams: string[] = [];
  for (let i = 0; i + 1 < value.length; i += 1) grams.push(value.slice(i, i + 2));
  return grams;
}

/** Sørensen–Dice over character bigrams. */
function similarity(a: string, b: string): number {
  const left = bigrams(a);
  const right = bigrams(b);
  if (left.length === 0 || right.length === 0) return a === b ? 1 : 0;
  const pool = new Map<string, number>();
  for (const gram of left) pool.set(gram, (pool.get(gram) ?? 0) + 1);
  let shared = 0;
  for (const gram of right) {
    const count = pool.get(gram) ?? 0;
    if (count > 0) {
      shared += 1;
      pool.set(gram, count - 1);
    }
  }
  return (2 * shared) / (left.length + right.length);
}

/**
 * The wrong wording is usually a near-miss of the required term (`Combat Pass`
 * for `Battle Pass`), so score runs of whole words against it. Requiring a
 * shared token keeps unrelated phrases of similar shape out.
 *
 * Targets without spaces (or with particles glued on) only match through the
 * trimmed variant below; when nothing clears the bar the caller falls back to
 * a re-translation, which is the safe outcome.
 */
function findSimilarSpan(
  target: string,
  fix: string
): { start: number; end: number } | null {
  const fixKey = compareKey(fix);
  const fixTokens = fixKey.split(" ").filter((token) => token.length >= 2);
  const tokens = tokenize(target);
  const maxSpan = fixKey.split(" ").length + 1;

  let best: { start: number; end: number } | null = null;
  let bestScore = 0;

  for (let start = 0; start < tokens.length; start += 1) {
    for (let size = 1; size <= maxSpan && start + size <= tokens.length; size += 1) {
      const from = tokens[start].start;
      const to = tokens[start + size - 1].end;
      const spanText = target.slice(from, to);
      const spanKey = compareKey(spanText);
      const sharesToken =
        fixTokens.some((token) => spanKey.includes(token)) ||
        spanKey.split(" ").some((token) => token.length >= 2 && fixKey.includes(token));
      if (!sharesToken) continue;

      // Also score the span trimmed to the term's length: agglutinative
      // targets glue particles onto the last word ("패스를"), which drags the
      // whole-token score below the bar even when the term is right there.
      const candidates: Array<{ end: number; key: string }> = [{ end: to, key: spanKey }];
      if (spanText.length > fix.length) {
        const trimmedTo = from + fix.length;
        candidates.push({ end: trimmedTo, key: compareKey(target.slice(from, trimmedTo)) });
      }

      for (const candidate of candidates) {
        const score = similarity(candidate.key, fixKey);
        if (score >= MIN_FUZZY_SIMILARITY && score > bestScore) {
          bestScore = score;
          best = { start: from, end: candidate.end };
        }
      }
    }
  }
  return best;
}

// --- queue state (§3.3) -----------------------------------------------------

export type ReviewOutcome = "applied" | "skipped";
export type ResolvedMap = Readonly<Record<string, ReviewOutcome>>;
export type ReviewMode = "queue" | "list";
export type EditorMode = "none" | "manual" | "ai";

/**
 * One undoable step, newest last. The server keeps two independent stacks —
 * dismissals never touch the draft history — so the client is the only place
 * that knows what `되돌리기` should undo next.
 */
export type ReviewLogEntry =
  | { kind: "edit"; keys: string[]; revision: number }
  | { kind: "dismiss"; keys: string[]; entries: ReviewDismissalEntry[] };

export interface QueueState {
  /** Queue order, frozen on first sight so handled items keep their place. */
  order: readonly string[];
  resolved: ResolvedMap;
  log: readonly ReviewLogEntry[];
  cursor: number;
  mode: ReviewMode;
  editor: EditorMode;
}

export const initialQueueState: QueueState = {
  order: [],
  resolved: {},
  log: [],
  cursor: 0,
  mode: "queue",
  editor: "none",
};

export type QueueAction =
  /** Reconcile with a freshly loaded fragment list (keys in severity order). */
  | { type: "sync"; keys: readonly string[] }
  | { type: "move"; delta: number }
  | { type: "focus"; key: string }
  | { type: "resolve"; entry: ReviewLogEntry }
  /** Take back an optimistic `resolve` whose server call then failed. */
  | { type: "rollback"; entry: ReviewLogEntry }
  | { type: "undo" }
  | { type: "editor"; editor: EditorMode }
  | { type: "mode"; mode: ReviewMode };

function clamp(value: number, max: number): number {
  return Math.max(0, Math.min(value, max));
}

function nextUnresolved(
  order: readonly string[],
  resolved: ResolvedMap,
  from: number
): number {
  for (let step = 1; step <= order.length; step += 1) {
    const at = (from + step) % order.length;
    if (!(order[at] in resolved)) return at;
  }
  return from;
}

export function queueReducer(state: QueueState, action: QueueAction): QueueState {
  switch (action.type) {
    case "sync": {
      const incoming = new Set(action.keys);
      const kept = state.order.filter((key) => incoming.has(key));
      const known = new Set(kept);
      // New findings (a partial apply can raise one) land at the end rather
      // than shuffling the item under the cursor.
      const order = [...kept, ...action.keys.filter((key) => !known.has(key))];
      const focused = state.order[state.cursor];
      const at = focused ? order.indexOf(focused) : -1;
      return {
        ...state,
        order,
        cursor: at >= 0 ? at : clamp(state.cursor, order.length - 1),
      };
    }
    case "move":
      return {
        ...state,
        cursor: clamp(state.cursor + action.delta, state.order.length - 1),
        editor: "none",
      };
    case "focus": {
      const at = state.order.indexOf(action.key);
      if (at < 0) return state;
      return { ...state, cursor: at, mode: "queue", editor: "none" };
    }
    case "resolve": {
      const outcome: ReviewOutcome = action.entry.kind === "edit" ? "applied" : "skipped";
      const resolved = { ...state.resolved };
      for (const key of action.entry.keys) resolved[key] = outcome;
      const focused = state.order[state.cursor];
      return {
        ...state,
        resolved,
        log: [...state.log, action.entry],
        cursor:
          focused && action.entry.keys.includes(focused)
            ? nextUnresolved(state.order, resolved, state.cursor)
            : state.cursor,
        editor: "none",
      };
    }
    case "rollback": {
      // Identity, not position: the cursor moved on optimistically, so a later
      // action may already sit on top of the one that failed.
      const log = state.log.filter((item) => item !== action.entry);
      if (log.length === state.log.length) return state;
      const resolved = { ...state.resolved };
      for (const key of action.entry.keys) delete resolved[key];
      const at = state.order.indexOf(action.entry.keys[0]);
      return {
        ...state,
        log,
        resolved,
        cursor: at >= 0 ? at : state.cursor,
        editor: "none",
      };
    }
    case "undo": {
      const entry = state.log[state.log.length - 1];
      if (!entry) return state;
      const resolved = { ...state.resolved };
      for (const key of entry.keys) delete resolved[key];
      const at = state.order.indexOf(entry.keys[0]);
      return {
        ...state,
        resolved,
        log: state.log.slice(0, -1),
        cursor: at >= 0 ? at : state.cursor,
        editor: "none",
      };
    }
    case "editor":
      return { ...state, editor: action.editor };
    case "mode":
      return { ...state, mode: action.mode, editor: "none" };
  }
}

export function focusedKey(state: QueueState): string | null {
  return state.order[state.cursor] ?? null;
}

export function remainingCount(state: QueueState): number {
  return state.order.filter((key) => !(key in state.resolved)).length;
}

export function isReviewComplete(state: QueueState): boolean {
  return state.order.length > 0 && remainingCount(state) === 0;
}

/** What `되돌리기` would undo, if anything. */
export function lastAction(state: QueueState): ReviewLogEntry | null {
  return state.log[state.log.length - 1] ?? null;
}

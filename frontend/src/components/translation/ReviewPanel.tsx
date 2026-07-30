"use client";

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { ApiError, apiClient } from "@/lib/api-client";
import { DoneScreen } from "@/components/translation/review/DoneScreen";
import { FinishBar } from "@/components/translation/review/FinishBar";
import { FragmentList } from "@/components/translation/review/FragmentList";
import { GlossaryPane } from "@/components/translation/review/GlossaryPane";
import { PartialMatchCard } from "@/components/translation/review/PartialMatchCard";
import { QueueItem } from "@/components/translation/review/QueueItem";
import { SlideRail, type SlideProgress } from "@/components/translation/review/SlideRail";
import { StepHeader } from "@/components/translation/review/StepHeader";
import {
  blockFindings,
  buildBlocks,
  buildQueue,
  initialQueueState,
  isReviewComplete,
  lastAction,
  primaryFinding,
  queueReducer,
  remainingCount,
  suggestFix,
  type EditorMode,
  type ReviewLogEntry,
  type ReviewProposal,
} from "@/lib/review-queue";
import type {
  FragmentItem,
  PartialCandidate,
  ReviewDismissalEntry,
} from "@/types/api";
import { AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface ReviewPanelProps {
  jobId: string;
  onClose: () => void;
  onDownload: () => void;
}

/**
 * The review screen: one flagged item at a time, in a fixed order, with the
 * deck's remaining work on the left and the glossary on the right. Everything
 * about *what* an item is and *where* the cursor goes lives in
 * `lib/review-queue`; this component only talks to the server and renders.
 */
export function ReviewPanel({ jobId, onClose, onDownload }: ReviewPanelProps) {
  const [fragments, setFragments] = useState<FragmentItem[]>([]);
  const [outputFilename, setOutputFilename] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [revision, setRevision] = useState(0);
  const [dirty, setDirty] = useState(false);
  const [busy, setBusy] = useState(false);
  const [saving, setSaving] = useState(false);
  // A re-translation belongs to the item it was made for: navigating away must
  // not offer it for the next one.
  const [proposalFor, setProposalFor] = useState<
    { key: string; proposal: ReviewProposal } | null
  >(null);
  const [pendingKey, setPendingKey] = useState<string | null>(null);
  const [partialCandidates, setPartialCandidates] = useState<PartialCandidate[]>([]);
  const [selectedPartial, setSelectedPartial] = useState<Set<number>>(new Set());
  const [applyingPartial, setApplyingPartial] = useState(false);
  const [editTexts, setEditTexts] = useState<Record<number, string>>({});
  const [instruction, setInstruction] = useState("");
  const [propagate, setPropagate] = useState(true);
  const [queueState, dispatch] = useReducer(queueReducer, initialQueueState);

  const fetchFragments = useCallback(async () => {
    const resp = await apiClient.getJobFragments(jobId);
    setFragments(resp.fragments);
    setOutputFilename(resp.output_filename);
    setRevision(resp.revision);
    setDirty(resp.dirty);
  }, [jobId]);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      await fetchFragments();
    } catch {
      setError("검토 목록을 불러오지 못했습니다.");
    } finally {
      setLoading(false);
    }
  }, [fetchFragments]);

  // Every mutation re-reads the list (the server re-sweeps findings), so the
  // re-read must not flash the whole screen the way the first load does.
  const refresh = useCallback(async () => {
    try {
      await fetchFragments();
    } catch {
      toast.error("목록을 새로 불러오지 못했습니다.");
    }
  }, [fetchFragments]);

  useEffect(() => {
    void load();
  }, [load]);

  const queue = useMemo(
    () => buildQueue(fragments, queueState.resolved, queueState.pinned),
    [fragments, queueState.resolved, queueState.pinned]
  );
  const allBlocks = useMemo(() => buildBlocks(fragments), [fragments]);
  const blocksByKey = useMemo(
    () => new Map(queue.map((block) => [block.key, block])),
    [queue]
  );
  const queueKeys = useMemo(() => queue.map((block) => block.key), [queue]);
  // Blocks the latest sweep still flags — an applied edit that left one behind
  // has not finished the item, so `sync` puts it back in the queue.
  const flaggedKeys = useMemo(
    () => queue.filter((block) => blockFindings(block).length > 0).map((block) => block.key),
    [queue]
  );

  useEffect(() => {
    dispatch({ type: "sync", keys: queueKeys, flagged: flaggedKeys });
  }, [queueKeys, flaggedKeys]);

  const currentKey = queueState.order[queueState.cursor] ?? null;
  const current = currentKey ? blocksByKey.get(currentKey) ?? null : null;
  const currentFinding = current ? primaryFinding(current) : null;
  const subject = current
    ? current.items.find((item) => item.index === currentFinding?.index) ?? current.items[0]
    : null;
  const total = queueState.order.length;
  const remaining = remainingCount(queueState);
  const undoable = lastAction(queueState) !== null;
  const complete = isReviewComplete(queueState);
  // The done screen is a destination, not a wall: `처리한 항목 다시 보기` steps
  // back into the queue, and new findings put it away on their own.
  const [reopened, setReopened] = useState(false);
  useEffect(() => {
    if (!complete) setReopened(false);
  }, [complete]);
  const outcomes = Object.values(queueState.resolved);
  const suggestion = useMemo(
    () => (subject && currentFinding ? suggestFix(subject.target, currentFinding.finding) : null),
    [subject, currentFinding]
  );

  const slides = useMemo<SlideProgress[]>(() => {
    const bySlide = new Map<number, SlideProgress>();
    for (const key of queueState.order) {
      if (key in queueState.resolved) continue;
      const block = blocksByKey.get(key);
      if (!block) continue;
      const entry = bySlide.get(block.slide);
      if (entry) {
        entry.remaining += 1;
      } else {
        bySlide.set(block.slide, {
          slide: block.slide,
          title: block.items[0].slide_title,
          remaining: 1,
        });
      }
    }
    return [...bySlide.values()].sort((a, b) => a.slide - b.slide);
  }, [queueState.order, queueState.resolved, blocksByKey]);

  const doneSlides = useMemo(() => {
    const left = new Set(slides.map((entry) => entry.slide));
    const all = new Set(queue.map((block) => block.slide));
    return [...all].filter((slide) => !left.has(slide)).length;
  }, [queue, slides]);

  const selectSlide = (slide: number) => {
    const key = queueState.order.find((candidate) => (
      !(candidate in queueState.resolved) && blocksByKey.get(candidate)?.slide === slide
    ));
    if (key) dispatch({ type: "focus", key });
  };

  const setEditor = (editor: EditorMode) => {
    if (editor === "manual" && current) {
      setEditTexts(
        Object.fromEntries(current.items.map((item) => [item.index, item.target]))
      );
    }
    if (editor === "ai") setInstruction("");
    if (editor === "none") setProposalFor(null);
    dispatch({ type: "editor", editor });
  };

  const receivePartials = (list: PartialCandidate[]) => {
    setPartialCandidates(list);
    // 후보가 하나면 고를 것이 없다 — 체크박스 없이 적용/건너뛰기만 남긴다.
    setSelectedPartial(new Set(list.length === 1 ? [list[0].index] : []));
  };

  // `busy` guards the shortcut as much as the button: a second `s` before the
  // first dismissal returns would log an undo step the server knows nothing of.
  const skip = async () => {
    if (!current || busy) return;
    const entries = blockFindings(current).map(({ index, finding }) => ({
      index,
      finding_type: finding.type,
    }));
    if (entries.length === 0) {
      // A block opened from the full list carries nothing to dismiss — the
      // server has no work, but the queue still has to let it be handled.
      dispatch({
        type: "resolve",
        entry: { kind: "dismiss", keys: [current.key], entries: [] },
      });
      return;
    }
    setBusy(true);
    try {
      const resp = await apiClient.updateReviewDismissals(jobId, "dismiss", entries);
      setDirty(resp.dirty);
      dispatch({
        type: "resolve",
        entry: { kind: "dismiss", keys: [current.key], entries: resp.changed },
      });
      await refresh();
    } catch {
      toast.error("처리에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  /**
   * One request for every finding still open, so `되돌리기` brings them all
   * back in one step. Dismissals only add to a set, so no lock or revision
   * check is involved — 40 items would otherwise be 40 round trips.
   */
  const skipAllRemaining = async () => {
    const keys = queueState.order.filter((key) => !(key in queueState.resolved));
    const entries = keys.flatMap((key) => {
      const block = blocksByKey.get(key);
      return block
        ? blockFindings(block).map(({ index, finding }) => ({
            index,
            finding_type: finding.type,
          }))
        : [];
    });
    if (keys.length === 0) return;
    setBusy(true);
    try {
      // Blocks pulled in from the full list have no findings to dismiss; they
      // still belong to the batch so `되돌리기` brings the whole step back.
      let changed: ReviewDismissalEntry[] = [];
      if (entries.length > 0) {
        const resp = await apiClient.updateReviewDismissals(jobId, "dismiss", entries);
        setDirty(resp.dirty);
        changed = resp.changed;
      }
      dispatch({ type: "resolve", entry: { kind: "dismiss", keys, entries: changed } });
      await refresh();
    } catch {
      toast.error("남은 항목을 넘기지 못했습니다.");
    } finally {
      setBusy(false);
    }
  };

  /**
   * propose → apply with no comparison modal in between. A 409 means another
   * change landed first; the proposal is bound to the revision it was made
   * against and cannot be reused, so re-read and propose once more.
   */
  const proposeAndApply = async (index: number, target: string) => {
    let expected = revision;
    for (let attempt = 0; attempt < 2; attempt += 1) {
      const proposed = await apiClient.proposeJobFragment(jobId, index, {
        action: "edit",
        target,
        propagate_identical: propagate,
      });
      try {
        return await apiClient.applyJobFragmentProposal(jobId, proposed.proposal_id, expected);
      } catch (err) {
        if (attempt > 0 || !(err instanceof ApiError) || err.status !== 409) throw err;
        const fresh = await apiClient.getJobFragments(jobId);
        setFragments(fresh.fragments);
        setRevision(fresh.revision);
        setDirty(fresh.dirty);
        expected = fresh.revision;
      }
    }
    throw new Error("apply conflicted twice");
  };

  const applySuggestion = async () => {
    if (!current || !subject || !suggestion) return;
    const entry: ReviewLogEntry = { kind: "edit", keys: [current.key], revision };
    // 적용 1회에 서버가 스윕을 다시 돌아 큰 덱에서 1~2초가 걸린다. 커서는 응답을
    // 기다리지 않고 넘기고, 실패하면 그 항목만 되돌린다.
    dispatch({ type: "resolve", entry });
    try {
      const resp = await proposeAndApply(subject.index, suggestion.target);
      setRevision(resp.revision);
      setDirty(resp.dirty);
      receivePartials(resp.partial_candidates);
      await refresh();
    } catch {
      dispatch({ type: "rollback", entry });
      toast.error("추천 수정을 적용하지 못했습니다.");
      await refresh();
    }
  };

  const applyEdits = async () => {
    if (!current) return;
    const edits: Record<number, string> = {};
    for (const item of current.items) {
      const next = editTexts[item.index] ?? item.target;
      if (next !== item.target) edits[item.index] = next;
    }
    const indices = Object.keys(edits).map(Number);
    if (indices.length === 0) {
      dispatch({ type: "editor", editor: "none" });
      return;
    }
    const entry: ReviewLogEntry = { kind: "edit", keys: [current.key], revision };
    dispatch({ type: "resolve", entry });
    try {
      if (indices.length === 1) {
        // 문단 하나면 propose 경로가 낫다 — 동일 문구 전파와 부분 일치 후보를 준다.
        const resp = await proposeAndApply(indices[0], edits[indices[0]]);
        setRevision(resp.revision);
        setDirty(resp.dirty);
        receivePartials(resp.partial_candidates);
      } else {
        // 여러 문단이면 한 요청으로 — 되돌리기 한 번에 문장 전체가 복구돼야 한다.
        // 동일 문구 전파는 이 경로에도 있고, 부분 일치 후보만 없다.
        const resp = await apiClient.applyReviewBlockEdit(jobId, edits, revision, propagate);
        setRevision(resp.revision);
        setDirty(resp.dirty);
      }
      await refresh();
    } catch {
      dispatch({ type: "rollback", entry });
      toast.error("수정을 적용하지 못했습니다.");
      await refresh();
    }
  };

  const retranslate = async () => {
    if (!current || !subject) return;
    const trimmed = instruction.trim();
    const used = current.items.reduce((sum, item) => sum + item.target.length, 0);
    const overBudget =
      subject.length_budget !== null && !subject.is_note && used > subject.length_budget;
    const hint = trimmed || (overBudget ? "더 짧게" : undefined);
    setProposalFor(null);
    setPendingKey(current.key);
    try {
      // A merged block is one sentence: re-translating a single paragraph of it
      // is what left the wording under review, so the whole item goes together.
      let proposal: ReviewProposal;
      if (current.items.length > 1) {
        const resp = await apiClient.retranslateReviewBlock(
          jobId,
          current.items.map((item) => item.index),
          hint
        );
        proposal = { kind: "block", edits: resp.edits, overBudget: resp.over_budget };
      } else {
        proposal = {
          kind: "fragment",
          response: await apiClient.proposeJobFragment(jobId, subject.index, {
            action: "retranslate",
            instruction: hint,
            propagate_identical: propagate,
          }),
        };
      }
      setProposalFor({ key: current.key, proposal });
    } catch {
      toast.error("재번역에 실패했습니다.");
    } finally {
      setPendingKey(null);
    }
  };

  const applyProposal = async () => {
    if (!current || proposalFor?.key !== current.key) return;
    const pending = proposalFor.proposal;
    const entry: ReviewLogEntry = { kind: "edit", keys: [current.key], revision };
    setProposalFor(null);
    dispatch({ type: "resolve", entry });
    try {
      if (pending.kind === "block") {
        const resp = await apiClient.applyReviewBlockEdit(
          jobId,
          pending.edits,
          revision,
          propagate
        );
        setRevision(resp.revision);
        setDirty(resp.dirty);
      } else {
        const resp = await apiClient.applyJobFragmentProposal(
          jobId,
          pending.response.proposal_id,
          revision
        );
        setRevision(resp.revision);
        setDirty(resp.dirty);
        receivePartials(resp.partial_candidates);
      }
      await refresh();
    } catch {
      // A proposal is bound to the revision it was made against, so a conflict
      // means re-running the model — the user decides whether that is worth it.
      dispatch({ type: "rollback", entry });
      toast.error("AI 번역 결과를 적용하지 못했습니다. 다시 시도해주세요.");
      await refresh();
    }
  };

  const applySelectedPartial = async () => {
    if (applyingPartial || selectedPartial.size === 0 || partialCandidates.length === 0) {
      return;
    }
    const first = partialCandidates[0];
    setApplyingPartial(true);
    try {
      const resp = await apiClient.applyPartialCandidates(jobId, {
        indices: Array.from(selectedPartial),
        old_phrase: first.old_phrase,
        new_phrase: first.new_phrase,
        expected_revision: revision,
      });
      setRevision(resp.revision);
      setDirty(resp.dirty);
      // This pushed a draft snapshot on the server, so the client log has to
      // grow with it — otherwise the next `되돌리기` pops this one while
      // claiming to undo the edit before it. No keys: nothing was handled here.
      if (resp.changed_indices.length > 0) {
        dispatch({ type: "resolve", entry: { kind: "edit", keys: [], revision } });
      }
      setPartialCandidates([]);
      setSelectedPartial(new Set());
      await refresh();
    } catch {
      toast.error("부분 일치 문구 적용에 실패했습니다.");
      await refresh();
    } finally {
      setApplyingPartial(false);
    }
  };

  const undo = async () => {
    const entry = lastAction(queueState);
    if (!entry) return;
    setBusy(true);
    try {
      if (entry.kind === "dismiss") {
        if (entry.entries.length > 0) {
          const resp = await apiClient.updateReviewDismissals(jobId, "restore", entry.entries);
          setDirty(resp.dirty);
        }
      } else {
        const resp = await apiClient.undoReview(jobId, revision);
        setRevision(resp.revision);
        setDirty(resp.dirty);
      }
      dispatch({ type: "undo" });
      setProposalFor(null);
      setPartialCandidates([]);
      await refresh();
    } catch {
      toast.error("되돌리기에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  /**
   * One key per action, but only while the queue itself has focus: typing an
   * `s` into the editor must stay an `s`.
   */
  const handleKey = (event: KeyboardEvent) => {
    const target = event.target as HTMLElement | null;
    const typing =
      target instanceof HTMLElement &&
      (target.isContentEditable ||
        ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName));

    if (event.key === "Escape") {
      if (queueState.editor !== "none" || proposalFor) setEditor("none");
      else if (queueState.mode === "list") dispatch({ type: "mode", mode: "queue" });
      else onClose();
      return;
    }
    if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
    if (loading || error || queueState.mode === "list") return;
    if (partialCandidates.length > 0 || (complete && !reopened)) return;
    if (!current) return;
    const handled = current.key in queueState.resolved;

    switch (event.key) {
      case "Enter":
        if (suggestion && !handled && queueState.editor === "none") {
          event.preventDefault();
          void applySuggestion();
        }
        break;
      case "ArrowLeft":
        dispatch({ type: "move", delta: -1 });
        break;
      case "ArrowRight":
        dispatch({ type: "move", delta: 1 });
        break;
      default:
        switch (event.key.toLowerCase()) {
          case "s":
            if (!handled) void skip();
            break;
          case "e":
            setEditor("manual");
            break;
          case "r":
            setEditor("ai");
            break;
        }
    }
  };

  // The listener is registered once; the ref keeps it reading current state
  // instead of the state it was created with.
  const keyHandler = useRef(handleKey);
  keyHandler.current = handleKey;
  useEffect(() => {
    const listener = (event: KeyboardEvent) => keyHandler.current(event);
    window.addEventListener("keydown", listener);
    return () => window.removeEventListener("keydown", listener);
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      if (dirty) {
        const resp = await apiClient.commitReview(jobId, revision);
        setDirty(resp.dirty);
        await refresh();
      }
      await onDownload();
    } catch {
      toast.error("최종 반영에 실패했습니다. 기존 결과 파일은 유지됩니다.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex flex-col bg-background">
      <StepHeader
        filename={outputFilename}
        canUndo={undoable}
        busy={busy || saving}
        onUndo={undo}
        onClose={onClose}
      />

      {loading && (
        <div className="flex flex-1 items-center justify-center gap-2 text-muted-foreground">
          <Loader2 className="size-5 animate-spin" />
          섹션을 불러오는 중...
        </div>
      )}

      {error && !loading && (
        <div className="m-4 flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
          <AlertTriangle className="size-4" />
          {error}
          <Button variant="outline" size="sm" className="ml-auto" onClick={() => void load()}>
            다시 시도
          </Button>
        </div>
      )}

      {!loading && !error && (
        <div className="flex min-h-0 flex-1">
          <SlideRail
            resolved={total - remaining}
            total={total}
            slides={slides}
            doneSlides={doneSlides}
            activeSlide={current?.slide ?? null}
            allCount={allBlocks.length}
            onSelectSlide={selectSlide}
            onShowAll={() => dispatch({ type: "mode", mode: "list" })}
          />

          <div className="flex min-w-0 flex-1 flex-col">
            {queueState.mode === "list" ? (
              <FragmentList
                blocks={allBlocks}
                resolved={queueState.resolved}
                onOpen={(key) => dispatch({ type: "pin", key })}
                onBack={() => dispatch({ type: "mode", mode: "queue" })}
              />
            ) : partialCandidates.length > 0 ? (
              <PartialMatchCard
                candidates={partialCandidates}
                selected={selectedPartial}
                busy={applyingPartial}
                onToggle={(index) =>
                  setSelectedPartial((currentSelection) => {
                    const next = new Set(currentSelection);
                    if (next.has(index)) next.delete(index);
                    else next.add(index);
                    return next;
                  })
                }
                onApply={applySelectedPartial}
                onSkip={() => receivePartials([])}
              />
            ) : complete && !reopened ? (
              <DoneScreen
                edited={outcomes.filter((outcome) => outcome === "applied").length}
                skipped={outcomes.filter((outcome) => outcome === "skipped").length}
                saving={saving}
                onSave={save}
                onReopen={() => setReopened(true)}
              />
            ) : current && subject ? (
              <div key={current.key} className="review-item-enter flex min-h-0 flex-1 flex-col">
                <QueueItem
                  block={current}
                  finding={currentFinding}
                  subject={subject}
                  suggestion={suggestion}
                  proposal={proposalFor?.key === current.key ? proposalFor.proposal : null}
                  proposalPending={pendingKey === current.key}
                  position={queueState.cursor + 1}
                  total={total}
                  handled={current.key in queueState.resolved}
                  busy={busy}
                  editor={queueState.editor}
                  editTexts={editTexts}
                  instruction={instruction}
                  propagate={propagate}
                  onEditTextChange={(index, value) =>
                    setEditTexts((previous) => ({ ...previous, [index]: value }))
                  }
                  onInstructionChange={setInstruction}
                  onPropagateChange={setPropagate}
                  onPrevious={() => dispatch({ type: "move", delta: -1 })}
                  onNext={() => dispatch({ type: "move", delta: 1 })}
                  onEditor={setEditor}
                  onApplySuggestion={applySuggestion}
                  onApplyEdits={applyEdits}
                  onRetranslate={retranslate}
                  onApplyProposal={applyProposal}
                  onCancelProposal={() => setEditor("none")}
                  onSkip={skip}
                />
              </div>
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 px-7 text-center">
                <CheckCircle2 className="size-10 text-success" />
                <p className="text-lg font-bold">확인이 필요한 항목이 없습니다.</p>
                <p className="text-[13px] text-muted-foreground">
                  번역 결과를 그대로 저장하거나 창을 닫으세요.
                </p>
              </div>
            )}

            <FinishBar
              remaining={remaining}
              dirty={dirty}
              saving={saving}
              busy={busy}
              onSave={save}
              onSkipAll={skipAllRemaining}
            />
          </div>

          <GlossaryPane
            jobId={jobId}
            disabled={busy || saving}
            itemSource={current?.items.map((item) => item.source).join(" ") ?? ""}
            onRegistered={refresh}
          />
        </div>
      )}
    </div>
  );
}

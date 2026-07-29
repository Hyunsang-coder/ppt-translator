"use client";

import { useCallback, useEffect, useMemo, useReducer, useState } from "react";
import { Button } from "@/components/ui/button";
import { apiClient } from "@/lib/api-client";
import { FinishBar } from "@/components/translation/review/FinishBar";
import { GlossaryPane } from "@/components/translation/review/GlossaryPane";
import { QueueItem } from "@/components/translation/review/QueueItem";
import { SlideRail, type SlideProgress } from "@/components/translation/review/SlideRail";
import { StepHeader } from "@/components/translation/review/StepHeader";
import { StyledText } from "@/components/translation/review/StyledText";
import { styleStatusLabel } from "@/components/translation/review/finding-labels";
import {
  blockFindings,
  buildQueue,
  initialQueueState,
  lastAction,
  primaryFinding,
  queueReducer,
  remainingCount,
  type EditorMode,
} from "@/lib/review-queue";
import type {
  FragmentItem,
  FragmentProposalResponse,
  PartialCandidate,
} from "@/types/api";
import { AlertTriangle, CheckCircle2, Loader2, X } from "lucide-react";
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
  const [proposal, setProposal] = useState<FragmentProposalResponse | null>(null);
  const [partialCandidates, setPartialCandidates] = useState<PartialCandidate[]>([]);
  const [selectedPartial, setSelectedPartial] = useState<Set<number>>(new Set());
  const [applyingPartial, setApplyingPartial] = useState(false);
  const [editText, setEditText] = useState("");
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
    () => buildQueue(fragments, queueState.resolved),
    [fragments, queueState.resolved]
  );
  const blocksByKey = useMemo(
    () => new Map(queue.map((block) => [block.key, block])),
    [queue]
  );
  const queueKeys = useMemo(() => queue.map((block) => block.key), [queue]);

  useEffect(() => {
    dispatch({ type: "sync", keys: queueKeys });
  }, [queueKeys]);

  const currentKey = queueState.order[queueState.cursor] ?? null;
  const current = currentKey ? blocksByKey.get(currentKey) ?? null : null;
  const currentFinding = current ? primaryFinding(current) : null;
  const subject = current
    ? current.items.find((item) => item.index === currentFinding?.index) ?? current.items[0]
    : null;
  const total = queueState.order.length;
  const remaining = remainingCount(queueState);
  const undoable = lastAction(queueState) !== null;

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
    if (editor === "manual" && subject) setEditText(subject.target);
    if (editor === "ai") setInstruction("");
    dispatch({ type: "editor", editor });
  };

  const skip = async () => {
    if (!current) return;
    const entries = blockFindings(current).map(({ index, finding }) => ({
      index,
      finding_type: finding.type,
    }));
    if (entries.length === 0) return;
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

  const previewEdit = async () => {
    if (!subject) return;
    setBusy(true);
    try {
      setProposal(
        await apiClient.proposeJobFragment(jobId, subject.index, {
          action: "edit",
          target: editText,
          propagate_identical: propagate,
        })
      );
    } catch {
      toast.error("수정 미리보기를 만들지 못했습니다.");
    } finally {
      setBusy(false);
    }
  };

  const retranslate = async () => {
    if (!subject) return;
    const trimmed = instruction.trim();
    const overBudget =
      subject.length_budget !== null &&
      !subject.is_note &&
      subject.target.length > subject.length_budget;
    setBusy(true);
    try {
      setProposal(
        await apiClient.proposeJobFragment(jobId, subject.index, {
          action: "retranslate",
          instruction: trimmed || (overBudget ? "더 짧게" : undefined),
          propagate_identical: propagate,
        })
      );
    } catch {
      toast.error("재번역에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

  const applyProposal = async () => {
    if (!proposal || !current) return;
    setBusy(true);
    try {
      const resp = await apiClient.applyJobFragmentProposal(
        jobId,
        proposal.proposal_id,
        revision
      );
      setRevision(resp.revision);
      setDirty(resp.dirty);
      setPartialCandidates(resp.partial_candidates);
      // 부분 일치는 문맥 검토가 필요한 보조 후보이므로 사용자가 직접 고른다.
      setSelectedPartial(new Set());
      setProposal(null);
      dispatch({
        type: "resolve",
        entry: { kind: "edit", keys: [current.key], revision: resp.revision },
      });
      await refresh();
    } catch {
      toast.error("적용에 실패했습니다. 목록을 새로 확인해주세요.");
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const applySelectedPartial = async () => {
    if (applyingPartial || selectedPartial.size === 0 || partialCandidates.length === 0) {
      return;
    }
    const first = partialCandidates[0];
    setApplyingPartial(true);
    try {
      await apiClient.applyPartialCandidates(jobId, {
        indices: Array.from(selectedPartial),
        old_phrase: first.old_phrase,
        new_phrase: first.new_phrase,
        expected_revision: revision,
      });
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
      setProposal(null);
      setPartialCandidates([]);
      await refresh();
    } catch {
      toast.error("되돌리기에 실패했습니다.");
    } finally {
      setBusy(false);
    }
  };

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

  const proposalFragment = fragments.find((item) => item.index === proposal?.index) ?? null;

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
            onSelectSlide={selectSlide}
          />

          <div className="flex min-w-0 flex-1 flex-col">
            {current && subject ? (
              <QueueItem
                block={current}
                finding={currentFinding}
                subject={subject}
                position={queueState.cursor + 1}
                total={total}
                handled={current.key in queueState.resolved}
                busy={busy}
                editor={queueState.editor}
                editText={editText}
                instruction={instruction}
                propagate={propagate}
                onEditTextChange={setEditText}
                onInstructionChange={setInstruction}
                onPropagateChange={setPropagate}
                onPrevious={() => dispatch({ type: "move", delta: -1 })}
                onNext={() => dispatch({ type: "move", delta: 1 })}
                onEditor={setEditor}
                onPreviewEdit={previewEdit}
                onRetranslate={retranslate}
                onSkip={skip}
              />
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 px-7 text-center">
                <CheckCircle2 className="size-10 text-success" />
                <p className="text-lg font-bold">확인이 필요한 항목이 없습니다.</p>
                <p className="text-[13px] text-muted-foreground">
                  번역 결과를 그대로 저장하거나 창을 닫으세요.
                </p>
              </div>
            )}

            <FinishBar remaining={remaining} dirty={dirty} saving={saving} onSave={save} />
          </div>

          <GlossaryPane jobId={jobId} disabled={busy || saving} onRegistered={refresh} />
        </div>
      )}

      {proposal && (
        <div className="absolute inset-0 z-20 flex items-center justify-center bg-background/75 p-4 backdrop-blur-sm">
          <div className="w-full max-w-2xl space-y-4 rounded-xl border bg-card p-4 shadow-xl">
            <div className="flex items-start justify-between gap-3">
              <div>
                <h3 className="font-bold">수정 후보 비교</h3>
                <p className="text-xs text-muted-foreground">
                  색상과 강조를 확인한 뒤 초안에 적용하세요.
                </p>
              </div>
              <Button variant="ghost" size="icon-sm" onClick={() => setProposal(null)}>
                <X className="size-4" />
              </Button>
            </div>
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="mb-1 text-[11px] font-bold text-muted-foreground">원문</p>
              <p className="text-sm leading-snug">{proposalFragment?.source ?? ""}</p>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <div className="rounded-lg bg-muted/50 p-3">
                <p className="mb-1 text-[11px] font-bold text-muted-foreground">현재 번역</p>
                <p className="text-sm">{proposal.old_target}</p>
              </div>
              <div className="rounded-lg border border-primary/30 bg-primary/5 p-3">
                <p className="mb-1 text-[11px] font-bold text-muted-foreground">
                  수정 후보 · 색상 미리보기
                </p>
                <StyledText segments={proposal.style_segments} fallback={proposal.target} />
              </div>
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              <span className="rounded-full bg-muted px-2 py-1">
                서식 {styleStatusLabel(proposal.style_status)}
              </span>
              {proposal.changed_indices.length > 1 && (
                <span className="rounded-full bg-info/10 px-2 py-1 text-info">
                  동일 문구 {proposal.changed_indices.length}곳
                </span>
              )}
              {proposal.over_budget && (
                <span className="rounded-full bg-destructive/10 px-2 py-1 text-destructive">
                  예상 박스 용량 초과
                </span>
              )}
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="outline" onClick={() => setProposal(null)}>취소</Button>
              <Button onClick={applyProposal} disabled={busy}>
                {busy && <Loader2 className="mr-2 size-4 animate-spin" />}
                적용
              </Button>
            </div>
          </div>
        </div>
      )}

      {partialCandidates.length > 0 && !proposal && (
        <div className="absolute bottom-4 left-1/2 z-10 w-[min(680px,calc(100%-2rem))] -translate-x-1/2 rounded-xl border bg-card p-4 shadow-xl">
          <div className="mb-3 flex items-start justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold">부분 일치 문구도 변경할까요?</h3>
              <p className="text-xs text-muted-foreground">
                문장 구조가 다른 위치는 원문을 확인하고 필요한 항목만 선택하세요.
              </p>
              {partialCandidates[0] && (
                <p className="mt-1 text-xs font-medium">
                  &ldquo;{partialCandidates[0].old_phrase}&rdquo; → &ldquo;
                  {partialCandidates[0].new_phrase || "(삭제)"}&rdquo;
                </p>
              )}
            </div>
            <Button
              variant="ghost"
              size="icon-sm"
              disabled={applyingPartial}
              onClick={() => setPartialCandidates([])}
            >
              <X className="size-4" />
            </Button>
          </div>
          <div className="max-h-48 space-y-2 overflow-y-auto">
            {partialCandidates.map((candidate) => (
              <label key={candidate.index} className="flex gap-2 rounded-lg bg-muted/50 p-2 text-xs">
                <input
                  type="checkbox"
                  checked={selectedPartial.has(candidate.index)}
                  disabled={applyingPartial}
                  onChange={(event) => {
                    setSelectedPartial((currentSelection) => {
                      const next = new Set(currentSelection);
                      if (event.target.checked) next.add(candidate.index);
                      else next.delete(candidate.index);
                      return next;
                    });
                  }}
                />
                <span className="min-w-0 space-y-0.5">
                  <span className="block text-muted-foreground">
                    <b className="text-foreground">S{candidate.slide}</b>
                    {candidate.is_note && " · 발표자 노트"} · 원문: {candidate.source}
                  </span>
                  <span className="block">{candidate.target}</span>
                  <span className="block text-primary">→ {candidate.proposed_target}</span>
                </span>
              </label>
            ))}
          </div>
          <div className="mt-3 flex justify-end gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={applyingPartial}
              onClick={() => setPartialCandidates([])}
            >
              건너뛰기
            </Button>
            <Button
              size="sm"
              onClick={applySelectedPartial}
              disabled={applyingPartial || selectedPartial.size === 0}
            >
              {applyingPartial && <Loader2 className="mr-2 size-4 animate-spin" />}
              {applyingPartial ? "적용 중" : `선택한 ${selectedPartial.size}건 적용`}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

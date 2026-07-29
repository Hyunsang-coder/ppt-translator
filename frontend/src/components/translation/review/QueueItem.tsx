"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { StyledText } from "@/components/translation/review/StyledText";
import {
  findingBadge,
  stylePreviewNote,
} from "@/components/translation/review/finding-labels";
import type { BlockFinding, EditorMode, FixSuggestion, ReviewBlock } from "@/lib/review-queue";
import type { FragmentItem, FragmentProposalResponse } from "@/types/api";
import {
  Ban,
  Check,
  ChevronLeft,
  ChevronRight,
  Loader2,
  Pencil,
  RefreshCw,
} from "lucide-react";

interface QueueItemProps {
  block: ReviewBlock;
  /** The finding this card speaks about, and the fragment carrying it. */
  finding: BlockFinding | null;
  subject: FragmentItem;
  /** Ready-made replacement, when the wrong wording could be located. */
  suggestion: FixSuggestion | null;
  /** A re-translation waiting to be accepted, and whether one is being made. */
  proposal: FragmentProposalResponse | null;
  proposalPending: boolean;
  position: number;
  total: number;
  handled: boolean;
  busy: boolean;
  editor: EditorMode;
  /** Draft text per paragraph of the block, keyed by fragment index. */
  editTexts: Record<number, string>;
  instruction: string;
  propagate: boolean;
  onEditTextChange: (index: number, value: string) => void;
  onInstructionChange: (value: string) => void;
  onPropagateChange: (value: boolean) => void;
  onPrevious: () => void;
  onNext: () => void;
  onEditor: (mode: EditorMode) => void;
  onApplySuggestion: () => void;
  onApplyEdits: () => void;
  onRetranslate: () => void;
  onApplyProposal: () => void;
  onCancelProposal: () => void;
  onSkip: () => void;
}

/**
 * The queue shows short title fragments and full paragraphs in the same slot,
 * so the type scales down instead of pushing the actions off-screen.
 */
function bodyTextClass(length: number, isNote: boolean): string {
  if (isNote) return "text-[15px]";
  if (length <= 60) return "text-[22px]";
  if (length <= 160) return "text-[18px]";
  return "text-[15px]";
}

/**
 * Paragraphs of one block share a text frame, so they share its capacity too —
 * one gauge for the item, not one per line.
 */
function LengthGauge({ used, budget }: { used: number; budget: number | null }) {
  if (budget === null) return null;
  const over = used > budget;
  return (
    <span className="flex items-center gap-2 text-xs text-muted-foreground">
      {used}자 / 박스 권장 {budget}자
      <span className="h-1 w-20 rounded-full bg-muted">
        <span
          className={`block h-full rounded-full ${over ? "bg-destructive" : "bg-success"}`}
          style={{ width: `${Math.min(used / budget, 1) * 100}%` }}
        />
      </span>
    </span>
  );
}

export function QueueItem({
  block,
  finding,
  subject,
  suggestion,
  proposal,
  proposalPending,
  position,
  total,
  handled,
  busy,
  editor,
  editTexts,
  instruction,
  propagate,
  onEditTextChange,
  onInstructionChange,
  onPropagateChange,
  onPrevious,
  onNext,
  onEditor,
  onApplySuggestion,
  onApplyEdits,
  onRetranslate,
  onApplyProposal,
  onCancelProposal,
  onSkip,
}: QueueItemProps) {
  const badge = finding ? findingBadge(finding.finding) : null;
  const longest = Math.max(...block.items.map((item) => item.target.length));
  const textClass = bodyTextClass(longest, subject.is_note);
  const noteClamp = subject.is_note ? "max-h-40 overflow-y-auto" : "";
  const styleNote = stylePreviewNote(subject.style_status);
  const budget = subject.is_note ? null : subject.length_budget;
  const currentLength = block.items.reduce((sum, item) => sum + item.target.length, 0);
  const draftLength = block.items.reduce(
    (sum, item) => sum + (editTexts[item.index] ?? item.target).length,
    0
  );
  const busyHere = busy || proposalPending;
  // Without a located replacement there is nothing to one-click, so the
  // re-translation becomes the obvious move instead of a secondary one.
  const promoteAi = !suggestion && !handled;
  const idle = editor === "none" && !proposal && !proposalPending;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2.5 px-7 pt-3.5">
        {badge ? (
          <span className={`rounded-full px-[11px] py-[5px] text-xs font-bold ${badge.cls}`}>
            {badge.label}
          </span>
        ) : (
          <span className="rounded-full bg-success/10 px-[11px] py-[5px] text-xs font-bold text-success">
            처리됨
          </span>
        )}
        <span className="text-[13px] text-muted-foreground">
          슬라이드 {block.slide} · {subject.is_note ? "발표자 노트" : "본문 텍스트"}
          {subject.repeat_count > 1 && ` · 덱 안 ${subject.repeat_count}곳에 반복`}
        </span>
        <span className="ml-auto flex items-center gap-1.5">
          <Button
            variant="outline"
            size="icon-sm"
            className="size-[30px]"
            disabled={position <= 1}
            onClick={onPrevious}
            aria-label="이전 항목"
          >
            <ChevronLeft className="size-[15px]" />
          </Button>
          <span className="min-w-16 text-center text-[13px] text-muted-foreground">
            {position}번째 / {total}
          </span>
          <Button
            variant="outline"
            size="icon-sm"
            className="size-[30px]"
            disabled={position >= total}
            onClick={onNext}
            aria-label="다음 항목"
          >
            <ChevronRight className="size-[15px]" />
          </Button>
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-7 pb-5 pt-4">
        {finding && (
          <p className="mb-4 text-[13px] leading-relaxed text-muted-foreground">
            {finding.finding.description}
            {finding.finding.suggested_fix && (
              <span className="text-foreground"> · 제안: {finding.finding.suggested_fix}</span>
            )}
          </p>
        )}

        <p className="mb-[7px] text-[11px] font-semibold tracking-[0.04em] text-muted-foreground">
          원문
        </p>
        <div className={`mb-3.5 space-y-1 ${textClass} leading-[1.45] tracking-[-0.01em] text-foreground/60 ${noteClamp}`}>
          {block.items.map((item) => (
            <p key={item.index}>{item.source}</p>
          ))}
        </div>

        <div className="mb-[7px] flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
          <p className="text-[11px] font-semibold tracking-[0.04em] text-muted-foreground">
            현재 번역
            {styleNote && (
              <span className="ml-2 font-normal tracking-normal">· {styleNote}</span>
            )}
          </p>
          <LengthGauge used={currentLength} budget={budget} />
        </div>
        <div className={`space-y-1 ${textClass} leading-[1.45] tracking-[-0.01em] ${noteClamp}`}>
          {block.items.map((item) => (
            <p key={item.index}>
              <StyledText segments={item.style_segments} fallback={item.target} />
            </p>
          ))}
        </div>

        {proposalPending && (
          <div className="mt-4 flex items-center gap-2 rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5 text-[13px] text-muted-foreground">
            <Loader2 className="size-4 animate-spin" />
            AI가 다시 번역하는 중…
          </div>
        )}

        {proposal && !proposalPending && (
          <div className="mt-4 rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5">
            <p className="mb-2 flex items-baseline gap-2">
              <span className="text-[11px] font-bold tracking-[0.04em] text-primary">
                AI 번역 결과
              </span>
              <span className="text-[11px] text-muted-foreground">
                {stylePreviewNote(proposal.style_status) ?? "단일 서식"}
              </span>
            </p>
            <p className={`mb-3.5 ${textClass} leading-[1.45] tracking-[-0.01em]`}>
              <StyledText segments={proposal.style_segments} fallback={proposal.target} />
            </p>
            {proposal.over_budget && (
              <p className="mb-2 text-xs text-destructive">예상 박스 용량을 넘습니다.</p>
            )}
            <div className="flex flex-wrap items-center gap-2.5">
              <Button
                className="h-[38px] gap-2 px-[18px] text-sm font-semibold"
                disabled={busyHere}
                onClick={onApplyProposal}
              >
                <Check className="size-4" />
                적용하고 다음
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-[34px] gap-1.5"
                disabled={busyHere}
                onClick={onRetranslate}
              >
                <RefreshCw className="size-3.5" />
                다시 시도
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="h-[34px]"
                disabled={busyHere}
                onClick={onCancelProposal}
              >
                취소
              </Button>
              {proposal.changed_indices.length > 1 && (
                <span className="text-xs text-muted-foreground">
                  동일 문구 {proposal.changed_indices.length}곳도 함께 바뀝니다
                </span>
              )}
            </div>
          </div>
        )}

        {suggestion && idle && !handled && (
          <div className="mt-4 rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5">
            <p className="mb-2 flex items-baseline gap-2">
              <span className="text-[11px] font-bold tracking-[0.04em] text-primary">
                추천 수정
              </span>
              <span className="text-[11px] text-muted-foreground">{suggestion.basis}</span>
            </p>
            <p className={`mb-3.5 ${textClass} leading-[1.45] tracking-[-0.01em]`}>
              {suggestion.target.slice(0, suggestion.span.start)}
              <mark className="rounded bg-success/[0.16] px-1 text-foreground">
                {suggestion.target.slice(suggestion.span.start, suggestion.span.end)}
              </mark>
              {suggestion.target.slice(suggestion.span.end)}
            </p>
            <div className="flex flex-wrap items-center gap-2.5">
              <Button
                className="h-[38px] gap-2 px-[18px] text-sm font-semibold"
                disabled={busyHere}
                onClick={onApplySuggestion}
              >
                <Check className="size-4" />
                적용하고 다음
              </Button>
              {subject.repeat_count > 1 && (
                <span className="text-xs text-muted-foreground">
                  반복되는 {subject.repeat_count}곳도 함께 바뀝니다
                </span>
              )}
            </div>
          </div>
        )}

        {editor === "manual" && !proposal && (
          <div className="mt-4 rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5">
            <div className="space-y-2">
              {block.items.map((item, order) => (
                <Textarea
                  key={item.index}
                  value={editTexts[item.index] ?? item.target}
                  onChange={(event) => onEditTextChange(item.index, event.target.value)}
                  onKeyDown={(event) => {
                    // ⌘/Ctrl+Enter applies; a bare Enter still breaks the line.
                    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && !busyHere) {
                      event.preventDefault();
                      onApplyEdits();
                    }
                  }}
                  className="min-h-[88px] text-[15px]"
                  disabled={busyHere}
                  autoFocus={order === 0}
                  aria-label={`번역 ${order + 1}`}
                />
              ))}
            </div>
            <div className="mt-2 flex flex-wrap items-center justify-between gap-x-3 gap-y-1">
              <p className="text-xs text-muted-foreground">
                이전: <s className="text-foreground/50">{subject.target}</s>
              </p>
              <LengthGauge used={draftLength} budget={budget} />
            </div>
            {block.items.length === 1 && subject.repeat_count > 1 && (
              <label className="mt-2 flex items-center gap-1.5 text-xs text-muted-foreground">
                <input
                  type="checkbox"
                  checked={propagate}
                  onChange={(event) => onPropagateChange(event.target.checked)}
                  className="accent-primary"
                />
                똑같은 문구 {subject.repeat_count}곳도 함께 바꾸기
              </label>
            )}
            <div className="mt-3 flex items-center gap-2.5">
              <Button
                className="h-[38px] gap-2 px-[18px] text-sm font-semibold"
                disabled={busyHere}
                onClick={onApplyEdits}
              >
                <Check className="size-4" />
                적용하고 다음
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-[34px]"
                disabled={busyHere}
                onClick={() => onEditor("none")}
              >
                취소
              </Button>
            </div>
          </div>
        )}

        {editor === "ai" && !proposal && !proposalPending && (
          <div className="mt-4 rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5">
            <input
              type="text"
              value={instruction}
              onChange={(event) => onInstructionChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !busyHere) onRetranslate();
              }}
              placeholder="추가 요청사항 (선택) · 예: 더 짧게, 더 격식있게"
              disabled={busyHere}
              autoFocus
              className="w-full rounded-md border bg-card px-2.5 py-2 text-sm outline-none focus:border-primary disabled:opacity-50"
            />
            <div className="mt-3 flex items-center gap-2.5">
              <Button size="sm" className="h-[34px] gap-1.5" disabled={busyHere} onClick={onRetranslate}>
                <RefreshCw className="size-3.5" />
                AI 번역
              </Button>
              <Button
                variant="outline"
                size="sm"
                className="h-[34px]"
                disabled={busyHere}
                onClick={() => onEditor("none")}
              >
                취소
              </Button>
              <span className="text-xs text-muted-foreground">
                비우면 원문을 기준으로 다시 번역합니다.
              </span>
            </div>
          </div>
        )}

        {idle && (
          <div className="mt-5 flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-[34px] gap-1.5"
              disabled={busyHere}
              onClick={() => onEditor("manual")}
            >
              <Pencil className="size-3.5" />
              직접 고치기
            </Button>
            <Button
              variant={promoteAi ? "default" : "outline"}
              size="sm"
              className={promoteAi ? "h-[38px] gap-1.5 px-[18px] font-semibold" : "h-[34px] gap-1.5"}
              disabled={busyHere}
              onClick={() => onEditor("ai")}
            >
              <RefreshCw className="size-3.5" />
              AI에게 다시 맡기기
            </Button>
            {!handled && (
              <Button
                variant="ghost"
                size="sm"
                className="h-[34px] gap-1.5 text-muted-foreground"
                disabled={busyHere}
                onClick={onSkip}
              >
                <Ban className="size-3.5" />
                이대로 두기
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

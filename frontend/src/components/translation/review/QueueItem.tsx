"use client";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { StyledText } from "@/components/translation/review/StyledText";
import { findingBadge } from "@/components/translation/review/finding-labels";
import type { BlockFinding, EditorMode, ReviewBlock } from "@/lib/review-queue";
import type { FragmentItem } from "@/types/api";
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
  position: number;
  total: number;
  handled: boolean;
  busy: boolean;
  editor: EditorMode;
  editText: string;
  instruction: string;
  propagate: boolean;
  onEditTextChange: (value: string) => void;
  onInstructionChange: (value: string) => void;
  onPropagateChange: (value: boolean) => void;
  onPrevious: () => void;
  onNext: () => void;
  onEditor: (mode: EditorMode) => void;
  onPreviewEdit: () => void;
  onRetranslate: () => void;
  onSkip: () => void;
}

/**
 * The queue shows short title fragments and full paragraphs in the same slot,
 * so the type scales down instead of pushing the actions off-screen.
 */
function bodyTextClass(text: string, isNote: boolean): string {
  if (isNote) return "text-[15px]";
  if (text.length <= 60) return "text-[22px]";
  if (text.length <= 160) return "text-[18px]";
  return "text-[15px]";
}

function LengthGauge({ fragment }: { fragment: FragmentItem }) {
  if (fragment.length_budget === null || fragment.is_note) return null;
  const used = fragment.target.length;
  const budget = fragment.length_budget;
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
  position,
  total,
  handled,
  busy,
  editor,
  editText,
  instruction,
  propagate,
  onEditTextChange,
  onInstructionChange,
  onPropagateChange,
  onPrevious,
  onNext,
  onEditor,
  onPreviewEdit,
  onRetranslate,
  onSkip,
}: QueueItemProps) {
  const badge = finding ? findingBadge(finding.finding) : null;
  const longest = Math.max(...block.items.map((item) => item.target.length));
  const textClass = bodyTextClass("x".repeat(longest), subject.is_note);
  const noteClamp = subject.is_note ? "max-h-40 overflow-y-auto" : "";

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

        <div className="mb-[7px] flex items-center justify-between gap-3">
          <p className="text-[11px] font-semibold tracking-[0.04em] text-muted-foreground">
            현재 번역
          </p>
          <LengthGauge fragment={subject} />
        </div>
        <div className={`space-y-1 ${textClass} leading-[1.45] tracking-[-0.01em] ${noteClamp}`}>
          {block.items.map((item) => (
            <p key={item.index}>
              <StyledText segments={item.style_segments} fallback={item.target} />
            </p>
          ))}
        </div>

        {editor === "manual" && (
          <div className="mt-4 rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5">
            <Textarea
              value={editText}
              onChange={(event) => onEditTextChange(event.target.value)}
              className="min-h-[88px] text-[15px]"
              disabled={busy}
              autoFocus
            />
            <p className="mt-1.5 text-xs text-muted-foreground">
              이전: <s className="text-foreground/50">{subject.target}</s>
            </p>
            {subject.repeat_count > 1 && (
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
              <Button size="sm" disabled={busy} className="gap-1.5" onClick={onPreviewEdit}>
                {busy ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
                확인
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
                onClick={() => onEditor("none")}
              >
                취소
              </Button>
            </div>
          </div>
        )}

        {editor === "ai" && (
          <div className="mt-4 rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5">
            <input
              type="text"
              value={instruction}
              onChange={(event) => onInstructionChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !busy) onRetranslate();
              }}
              placeholder="추가 요청사항 (선택) · 예: 더 짧게, 더 격식있게"
              disabled={busy}
              autoFocus
              className="w-full rounded-md border bg-card px-2.5 py-2 text-sm outline-none focus:border-primary disabled:opacity-50"
            />
            <div className="mt-3 flex items-center gap-2.5">
              <Button size="sm" disabled={busy} className="gap-1.5" onClick={onRetranslate}>
                {busy ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <RefreshCw className="size-4" />
                )}
                AI 번역
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={busy}
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

        {editor === "none" && (
          <div className="mt-5 flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-[34px] gap-1.5"
              disabled={busy}
              onClick={() => onEditor("manual")}
            >
              <Pencil className="size-3.5" />
              직접 고치기
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-[34px] gap-1.5"
              disabled={busy}
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
                disabled={busy}
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

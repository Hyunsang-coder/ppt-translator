"use client";

import { Button } from "@/components/ui/button";
import type { PartialCandidate } from "@/types/api";
import { Check, Loader2 } from "lucide-react";

interface PartialMatchCardProps {
  candidates: PartialCandidate[];
  selected: Set<number>;
  busy: boolean;
  onToggle: (index: number) => void;
  onApply: () => void;
  onSkip: () => void;
}

/**
 * Where the same phrase was translated inside a differently-shaped sentence.
 * These are not findings — nothing is wrong with them — so they take the item
 * slot once, right after the edit that raised them, instead of floating over
 * the screen until dismissed.
 */
export function PartialMatchCard({
  candidates,
  selected,
  busy,
  onToggle,
  onApply,
  onSkip,
}: PartialMatchCardProps) {
  const single = candidates.length === 1;
  const phrase = candidates[0];

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center gap-2.5 px-7 pt-3.5">
        <span className="rounded-full bg-info/10 px-[11px] py-[5px] text-xs font-bold text-info">
          비슷한 문구도 바꿀까요?
        </span>
        <span className="text-[13px] text-muted-foreground">
          문장 구조가 달라 자동으로 바꾸지 않은 {candidates.length}곳
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-7 pb-5 pt-4">
        <p className="mb-4 text-[13px] leading-relaxed text-muted-foreground">
          방금 고친 문구{" "}
          <b className="text-foreground">&ldquo;{phrase.old_phrase}&rdquo;</b> →{" "}
          <b className="text-foreground">
            &ldquo;{phrase.new_phrase || "(삭제)"}&rdquo;
          </b>
          가 아래 위치에도 있습니다. 원문을 확인하고 바꿀 곳만 고르세요.
        </p>

        <div className="space-y-2">
          {candidates.map((candidate) => {
            const checked = selected.has(candidate.index);
            const Row = single ? "div" : "label";
            return (
              <Row
                key={candidate.index}
                className={`flex gap-2.5 rounded-[10px] border p-3 text-[13px] ${
                  checked && !single ? "border-primary/40 bg-primary/[0.06]" : "border-border"
                }`}
              >
                {!single && (
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={busy}
                    onChange={() => onToggle(candidate.index)}
                    className="mt-0.5 accent-primary"
                  />
                )}
                <span className="min-w-0 flex-1 space-y-1">
                  <span className="block text-xs text-muted-foreground">
                    슬라이드 {candidate.slide}
                    {candidate.is_note && " · 발표자 노트"} · 원문: {candidate.source}
                  </span>
                  <span className="block text-foreground/60">{candidate.target}</span>
                  <span className="block text-primary">→ {candidate.proposed_target}</span>
                </span>
              </Row>
            );
          })}
        </div>

        <div className="mt-5 flex items-center gap-2.5">
          <Button
            className="h-[38px] gap-2 px-[18px] text-sm font-semibold"
            disabled={busy || selected.size === 0}
            onClick={onApply}
          >
            {busy ? <Loader2 className="size-4 animate-spin" /> : <Check className="size-4" />}
            {single ? "적용" : `선택한 ${selected.size}건 적용`}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="h-[34px]"
            disabled={busy}
            onClick={onSkip}
          >
            건너뛰기
          </Button>
        </div>
      </div>
    </div>
  );
}

"use client";

import { Button } from "@/components/ui/button";
import { CheckCircle2, ChevronRight, Undo2, X } from "lucide-react";

interface StepHeaderProps {
  /** Name the deck will be saved as; hidden until the backend reports it. */
  filename: string | null;
  canUndo: boolean;
  busy: boolean;
  onUndo: () => void;
  onClose: () => void;
}

/** Where the user is: translation done, review now, saving next. */
export function StepHeader({ filename, canUndo, busy, onUndo, onClose }: StepHeaderProps) {
  return (
    <div className="flex items-center justify-between gap-4 border-b border-border bg-card px-5 py-3">
      <div className="flex items-center gap-2.5">
        <span className="flex items-center gap-1.5">
          <CheckCircle2 className="size-4 text-success" />
          <span className="text-[13px] text-muted-foreground">번역</span>
        </span>
        <ChevronRight className="size-3.5 stroke-[2.5] text-border" />
        <span className="flex items-center gap-1.5">
          <span className="inline-flex size-[18px] items-center justify-center rounded-full bg-primary text-[11px] text-primary-foreground">
            2
          </span>
          <span className="text-[13px] font-bold text-primary">검토</span>
        </span>
        <ChevronRight className="size-3.5 stroke-[2.5] text-border" />
        <span className="flex items-center gap-1.5">
          <span className="inline-flex size-[18px] items-center justify-center rounded-full border-[1.5px] border-border text-[11px]">
            3
          </span>
          <span className="text-[13px] text-muted-foreground/70">저장</span>
        </span>
      </div>

      <div className="flex items-center gap-3">
        {filename && (
          <span className="max-w-[280px] truncate text-[13px] text-muted-foreground">
            {filename}
          </span>
        )}
        <Button
          variant="outline"
          size="sm"
          className="gap-1.5"
          disabled={!canUndo || busy}
          onClick={onUndo}
        >
          <Undo2 className="size-[15px]" />
          되돌리기
        </Button>
        <Button variant="ghost" size="icon-sm" onClick={onClose} aria-label="닫기">
          <X className="size-4" />
        </Button>
      </div>
    </div>
  );
}

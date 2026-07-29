"use client";

import { Button } from "@/components/ui/button";
import { Download, Loader2 } from "lucide-react";

interface FinishBarProps {
  remaining: number;
  dirty: boolean;
  saving: boolean;
  busy: boolean;
  onSave: () => void;
  onSkipAll: () => void;
}

/** Saving is always available — a half-reviewed deck is still a usable deck. */
export function FinishBar({
  remaining,
  dirty,
  saving,
  busy,
  onSave,
  onSkipAll,
}: FinishBarProps) {
  const message = remaining > 0
    ? `남은 ${remaining}건은 지금 저장해도 됩니다 — 파일을 다시 열어 이어서 검토할 수 있어요.`
    : dirty
    ? "확인이 필요한 항목을 모두 처리했습니다."
    : "아직 고친 곳이 없습니다. 저장하면 번역 결과가 그대로 저장됩니다.";

  return (
    <div className="flex items-center gap-3.5 border-t border-border bg-card px-7 py-3">
      <p className="text-[13px] text-muted-foreground">{message}</p>
      {remaining > 0 && (
        <Button
          variant="outline"
          className="ml-auto h-[38px]"
          disabled={busy || saving}
          onClick={onSkipAll}
        >
          남은 항목 모두 넘기기
        </Button>
      )}
      <Button
        className={`h-[38px] gap-2 px-[18px] text-sm font-semibold ${
          remaining > 0 ? "" : "ml-auto"
        }`}
        disabled={saving}
        onClick={onSave}
      >
        {saving ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}
        PPT 저장
      </Button>
    </div>
  );
}

"use client";

import { Button } from "@/components/ui/button";
import { CheckCircle2, Download, Loader2 } from "lucide-react";

interface DoneScreenProps {
  edited: number;
  skipped: number;
  saving: boolean;
  onSave: () => void;
  onReopen: () => void;
}

/** The queue is empty. Saving is the only thing left to do. */
export function DoneScreen({ edited, skipped, saving, onSave, onReopen }: DoneScreenProps) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-3 px-7 text-center">
      <CheckCircle2 className="size-10 text-success" />
      <p className="text-lg font-bold">확인이 필요한 항목을 모두 처리했습니다.</p>
      <p className="text-[13px] text-muted-foreground">
        고친 곳 {edited}곳 · 그대로 둔 곳 {skipped}곳
      </p>
      <div className="mt-2 flex items-center gap-2.5">
        <Button
          className="h-[38px] gap-2 px-[18px] text-sm font-semibold"
          disabled={saving}
          onClick={onSave}
        >
          {saving ? <Loader2 className="size-4 animate-spin" /> : <Download className="size-4" />}
          PPT 저장
        </Button>
        <Button variant="ghost" size="sm" className="h-[34px]" onClick={onReopen}>
          처리한 항목 다시 보기
        </Button>
      </div>
    </div>
  );
}

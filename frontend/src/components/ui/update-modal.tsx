"use client";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";

interface UpdateModalProps {
  isOpen: boolean;
  version: string;
  releaseNotes: string | undefined;
  downloading: boolean;
  progress: number;
  error: string | null;
  onUpdate: () => void | Promise<void>;
  onCancel: () => void;
  onSkipVersion: () => void;
  onDismiss: () => void;
}

/**
 * Update-available dialog for the desktop app. Self-contained overlay (no shared
 * Modal component in this project) built from the existing UI primitives.
 */
export function UpdateModal({
  isOpen,
  version,
  releaseNotes,
  downloading,
  progress,
  error,
  onUpdate,
  onCancel,
  onSkipVersion,
  onDismiss,
}: UpdateModalProps) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="update-modal-title"
      onClick={downloading ? undefined : onDismiss}
    >
      <div
        className="w-full max-w-[400px] rounded-lg border border-border bg-background p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="update-modal-title" className="mb-3 text-lg font-semibold">
          새로운 버전이 있습니다
        </h2>

        <p className="mb-4 text-sm text-muted-foreground">
          PPT 번역캣 {version} 버전을 설치할 수 있습니다.
        </p>

        {releaseNotes && (
          <div className="mb-4 max-h-32 overflow-y-auto whitespace-pre-wrap rounded-md bg-muted p-3 text-sm text-muted-foreground">
            {releaseNotes}
          </div>
        )}

        {error && (
          <div className="mb-4 rounded-md bg-destructive/10 p-3 text-sm text-destructive">
            <p>다운로드에 실패했습니다. 잠시 후 다시 시도해주세요.</p>
            <p className="mt-1 font-mono text-xs opacity-75">{error}</p>
          </div>
        )}

        {downloading ? (
          <div>
            <div className="mb-1 flex justify-between text-sm">
              <span>다운로드 중…</span>
              <span>{progress}%</span>
            </div>
            <Progress value={progress} className="mb-3" />
            <div className="flex justify-end">
              <Button variant="ghost" onClick={onCancel}>
                취소
              </Button>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between">
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground"
              onClick={onSkipVersion}
            >
              이 버전 건너뛰기
            </Button>
            <div className="flex gap-2">
              <Button variant="outline" onClick={onDismiss}>
                나중에
              </Button>
              <Button onClick={onUpdate}>지금 업데이트</Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

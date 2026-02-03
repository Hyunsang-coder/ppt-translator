"use client";

import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { JobProgress } from "@/types/api";
import type { TranslationStatus } from "@/stores/translation-store";

interface ProgressPanelProps {
  status: TranslationStatus;
  progress: JobProgress | null;
  startTime?: number;
}

const STATUS_LABELS: Record<string, string> = {
  pending: "대기 중",
  parsing: "파일 분석 중",
  detecting_language: "언어 감지 중",
  preparing_batches: "배치 준비 중",
  translating: "번역 중",
  applying_translations: "번역 적용 중",
  completed: "완료",
  failed: "실패",
};

export function ProgressPanel({ status, progress, startTime }: ProgressPanelProps) {
  const getProgressPercentage = (): number => {
    if (!progress) return 0;

    if (progress.total_batches > 0) {
      return Math.round((progress.current_batch / progress.total_batches) * 100);
    }

    if (progress.total_sentences > 0) {
      return Math.round((progress.current_sentence / progress.total_sentences) * 100);
    }

    return 0;
  };

  const getStatusIcon = (): string => {
    switch (status) {
      case "idle":
        return "⏸️";
      case "uploading":
        return "📤";
      case "translating":
        return "🔄";
      case "completed":
        return "✅";
      case "failed":
        return "❌";
      case "cancelled":
        return "🚫";
      default:
        return "⏳";
    }
  };

  const getElapsedTime = (): string => {
    if (!startTime) return "";
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const minutes = Math.floor(elapsed / 60);
    const seconds = elapsed % 60;
    return minutes > 0 ? `${minutes}분 ${seconds}초` : `${seconds}초`;
  };

  const percentage = getProgressPercentage();
  const statusLabel = progress?.status ? STATUS_LABELS[progress.status] || progress.status : "";

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          <span>{getStatusIcon()}</span>
          <span>번역 진행 상황</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Progress Bar */}
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">{statusLabel}</span>
            <span className="font-medium">{percentage}%</span>
          </div>
          <Progress value={percentage} className="h-2" />
        </div>

        {/* Details */}
        {progress && (
          <div className="grid grid-cols-2 gap-4 text-sm">
            {progress.total_batches > 0 && (
              <div>
                <span className="text-muted-foreground">배치: </span>
                <span className="font-medium">
                  {progress.current_batch} / {progress.total_batches}
                </span>
              </div>
            )}
            {progress.total_sentences > 0 && (
              <div>
                <span className="text-muted-foreground">문장: </span>
                <span className="font-medium">
                  {progress.current_sentence} / {progress.total_sentences}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Status Message */}
        {progress?.message && (
          <p className="text-sm text-muted-foreground bg-muted/50 p-2 rounded">
            {progress.message}
          </p>
        )}

        {/* Elapsed Time */}
        {startTime && status === "translating" && (
          <p className="text-xs text-muted-foreground">경과 시간: {getElapsedTime()}</p>
        )}
      </CardContent>
    </Card>
  );
}

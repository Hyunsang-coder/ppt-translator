"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Loader2, CheckCircle2, XCircle, Clock, FileStack, MessageSquare } from "lucide-react";
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
  fixing_colors: "서식 보정 중",
  applying_translations: "번역 적용 중",
  completed: "완료",
  failed: "실패",
};

interface CircularProgressProps {
  percentage: number;
  size?: number;
  strokeWidth?: number;
}

function CircularProgress({ percentage, size = 120, strokeWidth = 8 }: CircularProgressProps) {
  const radius = (size - strokeWidth) / 2;
  const circumference = radius * 2 * Math.PI;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg
        width={size}
        height={size}
        className="transform -rotate-90"
      >
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          className="text-primary/20"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="currentColor"
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="text-primary transition-all duration-500 ease-out progress-ring-animate"
        />
      </svg>
      {/* Center content */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-2xl font-bold text-foreground">{percentage}%</span>
      </div>
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subValue?: string;
}

function StatCard({ icon, label, value, subValue }: StatCardProps) {
  return (
    <div className="stat-card glass-card p-3 flex items-center gap-3">
      <div className="p-2 rounded-lg bg-primary/10 text-primary">
        {icon}
      </div>
      <div className="flex-1 min-w-0">
        <p className="text-xs text-muted-foreground">{label}</p>
        <p className="text-sm font-semibold truncate">{value}</p>
        {subValue && (
          <p className="text-xs text-muted-foreground">{subValue}</p>
        )}
      </div>
    </div>
  );
}

function formatElapsed(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return minutes > 0 ? `${minutes}분 ${secs}초` : `${secs}초`;
}

export function ProgressPanel({ status, progress, startTime }: ProgressPanelProps) {
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (!startTime) return;

    // Calculate initial elapsed time
    setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));

    // Only tick while translating
    if (status !== "translating" && status !== "uploading") return;

    const interval = setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);

    return () => clearInterval(interval);
  }, [startTime, status]);

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

  const getStatusIcon = () => {
    switch (status) {
      case "idle":
        return <Clock className="w-5 h-5 text-muted-foreground" />;
      case "uploading":
      case "translating":
        return <Loader2 className="w-5 h-5 text-primary animate-spin" />;
      case "completed":
        return <CheckCircle2 className="w-5 h-5 text-success" />;
      case "failed":
      case "cancelled":
        return <XCircle className="w-5 h-5 text-destructive" />;
      default:
        return <Clock className="w-5 h-5 text-muted-foreground" />;
    }
  };

  const percentage = getProgressPercentage();
  const statusLabel = progress?.status ? STATUS_LABELS[progress.status] || progress.status : "";

  return (
    <Card className="glass-card overflow-hidden animate-slide-up">
      <CardHeader className="pb-2">
        <CardTitle className="text-lg flex items-center gap-2">
          {getStatusIcon()}
          <span>번역 진행 상황</span>
          {status === "translating" && (
            <span className="ml-auto text-xs font-normal text-muted-foreground animate-pulse">
              진행 중... ({formatElapsed(elapsedSeconds)})
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Circular Progress */}
        <div className="flex flex-col items-center gap-4">
          <CircularProgress percentage={percentage} />
          <div className="text-center">
            <p className="text-sm font-medium">{statusLabel || "준비 중"}</p>
            {progress?.message && (
              <p className="text-xs text-muted-foreground mt-1 max-w-[250px] truncate">
                {progress.message}
              </p>
            )}
          </div>
        </div>

        {/* Linear Progress Bar */}
        <div className="space-y-2">
          <div className="h-2 rounded-full bg-muted overflow-hidden">
            <div
              className="h-full brand-gradient rounded-full transition-all duration-500 ease-out"
              style={{ width: `${percentage}%` }}
            />
          </div>
        </div>

        {/* Stat Cards */}
        {progress && (
          <div className="grid grid-cols-2 gap-3">
            {progress.total_batches > 0 && (
              <StatCard
                icon={<FileStack className="w-4 h-4" />}
                label="배치"
                value={`${progress.current_batch} / ${progress.total_batches}`}
              />
            )}
            {progress.total_sentences > 0 && (
              <StatCard
                icon={<MessageSquare className="w-4 h-4" />}
                label="문장"
                value={`${progress.current_sentence} / ${progress.total_sentences}`}
              />
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

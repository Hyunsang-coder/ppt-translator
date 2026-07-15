"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FileUploader } from "@/components/shared/FileUploader";
import { SettingsPanel, FilenameSettingsSection } from "./SettingsPanel";
import { ProgressPanel } from "./ProgressPanel";
import { LogViewer } from "./LogViewer";
import { ReviewPanel } from "./ReviewPanel";
import { useTranslation } from "@/hooks/useTranslation";
import { useConfig } from "@/hooks/useConfig";
import { Play, XCircle, RefreshCw, AlertCircle, ClipboardCheck, CheckCircle2, Circle } from "lucide-react";

export function TranslationForm() {
  const [startTime, setStartTime] = useState<number | null>(null);
  const [filenameError, setFilenameError] = useState<string | null>(null);
  const [reviewOpen, setReviewOpen] = useState(false);
  const { config, getModelsForProvider } = useConfig();
  const {
    pptFile,
    settings,
    status,
    progress,
    errorMessage,
    logs,
    jobId,
    isIdle,
    isTranslating,
    isCompleted,
    canStart,
    canCancel,
    setPptFile,
    updateSettings,
    startTranslation,
    cancelTranslation,
    downloadResult,
    retranslate,
    reset,
  } = useTranslation();

  // 번역이 완료되면 검토 & 수정 화면으로 자동 진입한다. 사용자가 검토를
  // 닫은 뒤 같은 job이 다시 자동으로 열리지 않도록 job당 한 번만 연다.
  const autoOpenedJob = useRef<string | null>(null);
  useEffect(() => {
    if (isCompleted && jobId && autoOpenedJob.current !== jobId) {
      autoOpenedJob.current = jobId;
      setReviewOpen(true);
    }
  }, [isCompleted, jobId]);

  // 현재 선택한 모델의 표시 이름 가져오기
  const providerModels = getModelsForProvider(settings.provider);
  const currentModelName = providerModels.find((m) => m.id === settings.model)?.name || settings.model;

  // 검증 체크
  const isCustomFilenameEmpty =
    settings.filenameSettings.mode === "custom" &&
    settings.filenameSettings.customName.trim() === "";
  const isTargetLangEmpty = !settings.targetLang || settings.targetLang === "Auto";

  const handleStart = async () => {
    // 대상 언어 미선택 시 경고
    if (isTargetLangEmpty) {
      setFilenameError("대상 언어를 선택해주세요.");
      return;
    }
    // 직접 입력 모드인데 파일명이 비어있으면 경고
    if (isCustomFilenameEmpty) {
      setFilenameError("출력 파일명을 입력해주세요.");
      return;
    }
    setFilenameError(null);
    setStartTime(Date.now());
    await startTranslation();
  };

  const handleRetranslate = async () => {
    if (isTargetLangEmpty) {
      setFilenameError("대상 언어를 선택해주세요.");
      return;
    }
    if (isCustomFilenameEmpty) {
      setFilenameError("출력 파일명을 입력해주세요.");
      return;
    }
    setFilenameError(null);
    setStartTime(Date.now());
    await retranslate();
  };

  const handleReset = () => {
    setStartTime(null);
    setFilenameError(null);
    reset();
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left Column: File Upload & Settings */}
      <div className="space-y-4">
        <FileUploader
          label="PPT 파일"
          required
          description={`PowerPoint 파일 (.pptx, .ppt) - 최대 ${config?.max_upload_size_mb || 1024}MB`}
          accept={{
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
              ".pptx",
            ],
            "application/vnd.ms-powerpoint": [".ppt"],
          }}
          maxSizeMB={config?.max_upload_size_mb || 1024}
          selectedFile={pptFile}
          onFileSelect={setPptFile}
          disabled={isTranslating}
        />

        <SettingsPanel
          settings={settings}
          onSettingsChange={(newSettings) => {
            updateSettings(newSettings);
            // 파일명 설정이 변경되면 에러 클리어
            if (newSettings.filenameSettings) {
              setFilenameError(null);
            }
          }}
          disabled={isTranslating}
        />
      </div>

      {/* Right Column: Actions, Filename Settings, Progress */}
      <div className="space-y-4">
        {/* Action Buttons */}
        <Card className="border-border overflow-hidden">
          <CardContent className="p-4">
            <div className="flex flex-wrap gap-3">
              {isIdle && (
                <Button
                  onClick={handleStart}
                  disabled={!canStart}
                  className="flex-1 btn-gradient gap-2"
                >
                  <Play className="w-4 h-4" />
                  번역 시작
                </Button>
              )}

              {isTranslating && (
                <Button
                  variant="destructive"
                  onClick={cancelTranslation}
                  disabled={!canCancel}
                  className="flex-1 gap-2"
                >
                  <XCircle className="w-4 h-4" />
                  취소
                </Button>
              )}

              {isCompleted && (
                <>
                  {/* 검토가 기본 경로. 다운로드는 검토 화면 안에서 이뤄진다. */}
                  <Button
                    onClick={() => setReviewOpen(true)}
                    disabled={!jobId}
                    className="flex-1 btn-gradient gap-2"
                  >
                    <ClipboardCheck className="w-4 h-4" />
                    검토 &amp; 수정
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleRetranslate}
                    className="flex-1 gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    전체 재번역
                  </Button>
                </>
              )}

              {(status === "failed" || status === "cancelled") && (
                <Button
                  variant="outline"
                  onClick={handleRetranslate}
                  className="flex-1 gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  다시 시도
                </Button>
              )}
            </div>

            {/* 시작 조건 체크리스트: 버튼이 왜 비활성인지 항상 보여준다 */}
            {isIdle && !canStart && (
              <div className="mt-3 space-y-1.5" aria-live="polite">
                <StartCondition met={pptFile !== null} label="PPT 파일 선택" />
                <StartCondition met={!isTargetLangEmpty} label="대상 언어 선택" />
                {settings.filenameSettings.mode === "custom" && (
                  <StartCondition met={!isCustomFilenameEmpty} label="출력 파일명 입력" />
                )}
              </div>
            )}

            {filenameError && (
              <div className="mt-3 p-3 rounded-lg border border-warning/30 bg-warning/10 animate-slide-in">
                <p className="text-sm text-warning flex items-start gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{filenameError}</span>
                </p>
              </div>
            )}

            {errorMessage && (
              <div className="mt-3 p-3 rounded-lg border border-destructive/30 bg-destructive/10 animate-slide-in">
                <p className="text-sm text-destructive flex items-start gap-2">
                  <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{errorMessage}</span>
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Filename Settings */}
        <Card className="border-border overflow-hidden">
          <CardContent className="p-4">
            <FilenameSettingsSection
              settings={settings.filenameSettings}
              onChange={(filenameSettings) => {
                updateSettings({ filenameSettings });
                setFilenameError(null);
              }}
              pptFile={pptFile}
              targetLang={settings.targetLang}
              modelName={currentModelName}
              disabled={isTranslating}
            />
          </CardContent>
        </Card>

        {/* Progress */}
        {(isTranslating || isCompleted || status === "failed" || status === "cancelled") && (
          <ProgressPanel
            status={status}
            progress={progress}
            startTime={startTime ?? undefined}
          />
        )}

        {/* Logs */}
        <LogViewer logs={logs} />
      </div>

      {/* Review & edit overlay (WP-C5) */}
      {reviewOpen && jobId && (
        <ReviewPanel
          jobId={jobId}
          onClose={() => setReviewOpen(false)}
          onDownload={downloadResult}
        />
      )}
    </div>
  );
}

function StartCondition({ met, label }: { met: boolean; label: string }) {
  return (
    <p
      className={`text-xs flex items-center gap-1.5 ${
        met ? "text-success" : "text-muted-foreground"
      }`}
    >
      {met ? (
        <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" />
      ) : (
        <Circle className="w-3.5 h-3.5 flex-shrink-0 opacity-40" />
      )}
      {label}
    </p>
  );
}

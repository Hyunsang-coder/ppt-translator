"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FileUploader } from "@/components/shared/FileUploader";
import { SettingsPanel } from "./SettingsPanel";
import { ProgressPanel } from "./ProgressPanel";
import { LogViewer } from "./LogViewer";
import { useTranslation } from "@/hooks/useTranslation";
import { useConfig } from "@/hooks/useConfig";
import { Play, XCircle, Download, RefreshCw } from "lucide-react";

export function TranslationForm() {
  const [startTime, setStartTime] = useState<number | null>(null);
  const { config } = useConfig();
  const {
    pptFile,
    glossaryFile,
    settings,
    status,
    progress,
    errorMessage,
    logs,
    isIdle,
    isTranslating,
    isCompleted,
    canStart,
    canCancel,
    canDownload,
    setPptFile,
    setGlossaryFile,
    updateSettings,
    startTranslation,
    cancelTranslation,
    downloadResult,
    reset,
  } = useTranslation();

  const handleStart = async () => {
    setStartTime(Date.now());
    await startTranslation();
  };

  const handleReset = () => {
    setStartTime(null);
    reset();
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Left Column: File Upload & Settings */}
      <div className="space-y-4">
        <FileUploader
          label="PPT 파일"
          description={`PowerPoint 파일 (.pptx, .ppt) - 최대 ${config?.max_upload_size_mb || 50}MB`}
          accept={{
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
              ".pptx",
            ],
            "application/vnd.ms-powerpoint": [".ppt"],
          }}
          maxSizeMB={config?.max_upload_size_mb || 50}
          selectedFile={pptFile}
          onFileSelect={setPptFile}
          disabled={isTranslating}
        />

        <SettingsPanel
          settings={settings}
          onSettingsChange={updateSettings}
          glossaryFile={glossaryFile}
          onGlossaryFileChange={setGlossaryFile}
          disabled={isTranslating}
        />
      </div>

      {/* Right Column: Progress & Actions */}
      <div className="space-y-6">
        {/* Action Buttons */}
        <Card className="border-border overflow-hidden">
          <CardContent className="pt-6">
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
                  <Button
                    onClick={downloadResult}
                    disabled={!canDownload}
                    className="flex-1 btn-gradient gap-2"
                  >
                    <Download className="w-4 h-4" />
                    다운로드
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleReset}
                    className="flex-1 gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    새 번역
                  </Button>
                </>
              )}

              {status === "failed" && (
                <Button
                  variant="outline"
                  onClick={handleReset}
                  className="flex-1 gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  다시 시도
                </Button>
              )}
            </div>

            {errorMessage && (
              <div className="mt-4 p-3 rounded-lg border border-destructive/30 bg-destructive/10 animate-slide-in">
                <p className="text-sm text-destructive flex items-start gap-2">
                  <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{errorMessage}</span>
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Progress */}
        {(isTranslating || isCompleted || status === "failed") && (
          <ProgressPanel
            status={status}
            progress={progress}
            startTime={startTime ?? undefined}
          />
        )}

        {/* Logs */}
        <LogViewer logs={logs} />
      </div>
    </div>
  );
}

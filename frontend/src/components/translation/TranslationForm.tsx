"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FileUploader } from "@/components/shared/FileUploader";
import { SettingsPanel } from "./SettingsPanel";
import { ProgressPanel } from "./ProgressPanel";
import { LogViewer } from "./LogViewer";
import { useTranslation } from "@/hooks/useTranslation";
import { useConfig } from "@/hooks/useConfig";

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
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>파일 업로드</CardTitle>
          </CardHeader>
          <CardContent>
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
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>번역 설정</CardTitle>
          </CardHeader>
          <CardContent>
            <SettingsPanel
              settings={settings}
              onSettingsChange={updateSettings}
              glossaryFile={glossaryFile}
              onGlossaryFileChange={setGlossaryFile}
              disabled={isTranslating}
            />
          </CardContent>
        </Card>
      </div>

      {/* Right Column: Progress & Actions */}
      <div className="space-y-6">
        {/* Action Buttons */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-wrap gap-3">
              {isIdle && (
                <Button onClick={handleStart} disabled={!canStart} className="flex-1">
                  번역 시작
                </Button>
              )}

              {isTranslating && (
                <Button
                  variant="destructive"
                  onClick={cancelTranslation}
                  disabled={!canCancel}
                  className="flex-1"
                >
                  취소
                </Button>
              )}

              {isCompleted && (
                <>
                  <Button onClick={downloadResult} disabled={!canDownload} className="flex-1">
                    다운로드
                  </Button>
                  <Button variant="outline" onClick={handleReset} className="flex-1">
                    새 번역
                  </Button>
                </>
              )}

              {status === "failed" && (
                <Button variant="outline" onClick={handleReset} className="flex-1">
                  다시 시도
                </Button>
              )}
            </div>

            {errorMessage && (
              <p className="mt-4 text-sm text-destructive bg-destructive/10 p-3 rounded">
                {errorMessage}
              </p>
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

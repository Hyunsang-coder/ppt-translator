"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { FileUploader } from "@/components/shared/FileUploader";
import { SettingsPanel, FilenameSettingsSection } from "./SettingsPanel";
import { ProgressPanel } from "./ProgressPanel";
import { LogViewer } from "./LogViewer";
import { useTranslation } from "@/hooks/useTranslation";
import { useConfig } from "@/hooks/useConfig";
import { Play, XCircle, Download, RefreshCw, AlertCircle } from "lucide-react";

export function TranslationForm() {
  const [startTime, setStartTime] = useState<number | null>(null);
  const [filenameError, setFilenameError] = useState<string | null>(null);
  const { config } = useConfig();
  const {
    pptFile,
    glossaryFile,
    settings,
    generatedContext,
    isGeneratingContext,
    generatedInstructions,
    isGeneratingInstructions,
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
    setGeneratedContext,
    generateContext,
    setGeneratedInstructions,
    generateInstructions,
    startTranslation,
    cancelTranslation,
    downloadResult,
    reset,
  } = useTranslation();

  // 검증 체크
  const isCustomFilenameEmpty =
    settings.filenameSettings.mode === "custom" &&
    settings.filenameSettings.customName.trim() === "";
  const isTargetLangEmpty = !settings.targetLang || settings.targetLang === "Auto";

  const handleStart = async () => {
    // 타겟 언어 미선택 시 경고
    if (isTargetLangEmpty) {
      setFilenameError("타겟 언어를 선택해주세요.");
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
          onSettingsChange={(newSettings) => {
            updateSettings(newSettings);
            // 파일명 설정이 변경되면 에러 클리어
            if (newSettings.filenameSettings) {
              setFilenameError(null);
            }
          }}
          glossaryFile={glossaryFile}
          onGlossaryFileChange={setGlossaryFile}
          pptFile={pptFile}
          generatedContext={generatedContext}
          generatedInstructions={generatedInstructions}
          isGeneratingContext={isGeneratingContext}
          isGeneratingInstructions={isGeneratingInstructions}
          onGenerateContext={generateContext}
          onGenerateInstructions={generateInstructions}
          onContextChange={setGeneratedContext}
          onInstructionsChange={setGeneratedInstructions}
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
              model={settings.model}
              disabled={isTranslating}
            />
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

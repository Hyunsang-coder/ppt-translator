"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { FileUploader } from "@/components/shared/FileUploader";
import { MarkdownPreview } from "./MarkdownPreview";
import { useExtraction } from "@/hooks/useExtraction";
import { useConfig } from "@/hooks/useConfig";
import { Play, Copy, Download, RefreshCw, XCircle, WifiOff } from "lucide-react";

export function ExtractionForm() {
  const { config, isBackendConnected } = useConfig();
  const {
    pptFile,
    settings,
    status,
    errorMessage,
    markdown,
    slideCount,
    isExtracting,
    isCompleted,
    canStart,
    hasResult,
    setPptFile,
    updateSettings,
    startExtraction,
    copyToClipboard,
    downloadMarkdown,
    reset,
  } = useExtraction();

  const handleCopy = async () => {
    const success = await copyToClipboard();
    if (success) {
      // Could show a toast notification here
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Backend Connection Warning */}
      {!isBackendConnected && (
        <div className="col-span-full">
          <div className="p-3 rounded-lg border border-warning/30 bg-warning/10 flex items-center gap-3">
            <WifiOff className="w-5 h-5 text-warning flex-shrink-0" />
            <div className="text-sm">
              <span className="font-medium text-warning">백엔드 미연결</span>
              <span className="text-muted-foreground ml-2">
                서버가 연결되면 추출 기능을 사용할 수 있습니다.
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Left Column: File Upload & Settings */}
      <div className="space-y-4">
        <FileUploader
          label="PPT 파일"
          description={`PowerPoint 파일 (.pptx) - 최대 ${config?.max_upload_size_mb || 200}MB`}
          accept={{
            "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
              ".pptx",
            ],
          }}
          maxSizeMB={config?.max_upload_size_mb || 200}
          selectedFile={pptFile}
          onFileSelect={setPptFile}
          disabled={isExtracting}
        />

        {/* Extraction Settings */}
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3">
            {/* Figures handling */}
            <div className="space-y-1.5">
              <Label htmlFor="figures">그림 처리</Label>
              <Select
                value={settings.figures}
                onValueChange={(value: "omit" | "placeholder") =>
                  updateSettings({ figures: value })
                }
                disabled={isExtracting}
              >
                <SelectTrigger id="figures">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="omit">제외</SelectItem>
                  <SelectItem value="placeholder">플레이스홀더</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Charts handling */}
            <div className="space-y-1.5">
              <Label htmlFor="charts">차트 처리</Label>
              <Select
                value={settings.charts}
                onValueChange={(value: "labels" | "placeholder" | "omit") =>
                  updateSettings({ charts: value })
                }
                disabled={isExtracting}
              >
                <SelectTrigger id="charts">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="labels">레이블 포함</SelectItem>
                  <SelectItem value="placeholder">플레이스홀더</SelectItem>
                  <SelectItem value="omit">제외</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* With Notes */}
          <div className="flex items-center space-x-2">
            <Checkbox
              id="with-notes"
              checked={settings.withNotes}
              onCheckedChange={(checked) => updateSettings({ withNotes: checked === true })}
              disabled={isExtracting}
            />
            <Label htmlFor="with-notes" className="text-sm font-normal cursor-pointer">
              발표자 노트 포함
            </Label>
          </div>

          {/* Table Header */}
          <div className="flex items-center space-x-2">
            <Checkbox
              id="table-header"
              checked={settings.tableHeader}
              onCheckedChange={(checked) => updateSettings({ tableHeader: checked === true })}
              disabled={isExtracting}
            />
            <Label htmlFor="table-header" className="text-sm font-normal cursor-pointer">
              테이블 첫 행을 헤더로 처리
            </Label>
          </div>
        </div>
      </div>

      {/* Right Column: Actions & Preview */}
      <div className="space-y-4">
        {/* Action Buttons */}
        <Card className="border-border overflow-hidden">
          <CardContent className="p-4">
            <div className="flex flex-wrap gap-3">
              {!hasResult ? (
                <Button
                  onClick={startExtraction}
                  disabled={!canStart}
                  className="flex-1 btn-gradient gap-2"
                >
                  <Play className="w-4 h-4" />
                  {isExtracting ? "추출 중..." : "Markdown 변환"}
                </Button>
              ) : (
                <>
                  <Button
                    variant="outline"
                    onClick={handleCopy}
                    className="flex-1 gap-2"
                  >
                    <Copy className="w-4 h-4" />
                    복사
                  </Button>
                  <Button
                    onClick={downloadMarkdown}
                    className="flex-1 btn-gradient gap-2"
                  >
                    <Download className="w-4 h-4" />
                    다운로드
                  </Button>
                  <Button
                    variant="outline"
                    onClick={reset}
                    className="flex-1 gap-2"
                  >
                    <RefreshCw className="w-4 h-4" />
                    새 추출
                  </Button>
                </>
              )}
            </div>

            {errorMessage && (
              <div className="mt-3 p-3 rounded-lg border border-destructive/30 bg-destructive/10">
                <p className="text-sm text-destructive flex items-start gap-2">
                  <XCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                  <span>{errorMessage}</span>
                </p>
              </div>
            )}

            {slideCount !== null && (
              <p className="mt-3 text-sm text-muted-foreground">
                {slideCount}개 슬라이드에서 텍스트를 추출했습니다.
              </p>
            )}
          </CardContent>
        </Card>

        {/* Preview */}
        <MarkdownPreview markdown={markdown} />
      </div>
    </div>
  );
}

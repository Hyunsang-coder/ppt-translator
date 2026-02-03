"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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

export function ExtractionForm() {
  const { config } = useConfig();
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
      {/* Left Column: File Upload & Settings */}
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>파일 업로드</CardTitle>
          </CardHeader>
          <CardContent>
            <FileUploader
              label="PPT 파일"
              description={`PowerPoint 파일 (.pptx) - 최대 ${config?.max_upload_size_mb || 50}MB`}
              accept={{
                "application/vnd.openxmlformats-officedocument.presentationml.presentation": [
                  ".pptx",
                ],
              }}
              maxSizeMB={config?.max_upload_size_mb || 50}
              selectedFile={pptFile}
              onFileSelect={setPptFile}
              disabled={isExtracting}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>추출 설정</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Figures handling */}
            <div className="space-y-2">
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
            <div className="space-y-2">
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
          </CardContent>
        </Card>

        {/* Action Buttons */}
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-wrap gap-3">
              <Button onClick={startExtraction} disabled={!canStart} className="flex-1">
                {isExtracting ? "추출 중..." : "Markdown 변환"}
              </Button>

              {hasResult && (
                <>
                  <Button variant="outline" onClick={handleCopy} className="flex-1">
                    복사
                  </Button>
                  <Button variant="outline" onClick={downloadMarkdown} className="flex-1">
                    다운로드
                  </Button>
                </>
              )}

              {isCompleted && (
                <Button variant="ghost" onClick={reset}>
                  초기화
                </Button>
              )}
            </div>

            {errorMessage && (
              <p className="mt-4 text-sm text-destructive bg-destructive/10 p-3 rounded">
                {errorMessage}
              </p>
            )}

            {slideCount !== null && (
              <p className="mt-4 text-sm text-muted-foreground">
                {slideCount}개 슬라이드에서 텍스트를 추출했습니다.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Right Column: Preview */}
      <div>
        <MarkdownPreview markdown={markdown} />
      </div>
    </div>
  );
}

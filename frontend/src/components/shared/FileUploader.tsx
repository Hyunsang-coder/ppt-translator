"use client";

import { useCallback, useEffect, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { Upload, FileText, X, CloudUpload, Plus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { isTauri } from "@/lib/api-base";

interface FileUploaderProps {
  accept: Record<string, string[]>;
  maxSizeMB?: number;
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  label: string;
  description?: string;
  disabled?: boolean;
  /** 필수 항목 표시 (*) */
  required?: boolean;
  /** compact: 드롭존 대신 작은 버튼형 업로더 (보조 파일용) */
  variant?: "dropzone" | "compact";
}

export function FileUploader({
  accept,
  maxSizeMB = 1024,
  onFileSelect,
  selectedFile,
  label,
  description,
  disabled = false,
  required = false,
  variant = "dropzone",
}: FileUploaderProps) {
  const [error, setError] = useState<string | null>(null);
  // Tauri webview는 네이티브 핸들러가 drop 이벤트를 가로채므로 HTML5 드래그는
  // 브라우저(웹/개발) 환경에서만 활성화한다. mount 후 판별해 hydration 불일치 방지.
  const [dragEnabled, setDragEnabled] = useState(false);
  useEffect(() => {
    setDragEnabled(!isTauri());
  }, []);

  const onDrop = useCallback(
    (acceptedFiles: File[], rejectedFiles: FileRejection[]) => {
      setError(null);

      if (rejectedFiles.length > 0) {
        const rejection = rejectedFiles[0];
        const errorMessages = rejection.errors.map((e) => e.message).join(", ");
        setError(errorMessages);
        return;
      }

      if (acceptedFiles.length > 0) {
        const file = acceptedFiles[0];
        if (file.size > maxSizeMB * 1024 * 1024) {
          setError(`파일 크기가 ${maxSizeMB}MB를 초과합니다.`);
          return;
        }
        onFileSelect(file);
      }
    },
    [maxSizeMB, onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept,
    maxFiles: 1,
    noDrag: !dragEnabled,
    disabled,
  });

  const handleRemove = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      onFileSelect(null);
      setError(null);
    },
    [onFileSelect]
  );

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  // 보조 파일(용어집 등)용 컴팩트 버튼형: 메인 드롭존과 시각적으로 구분한다
  if (variant === "compact") {
    return (
      <div className="space-y-2">
        {selectedFile ? (
          <div className="inline-flex items-center gap-2 rounded-lg border border-primary/40 bg-primary/5 px-3 py-1.5 text-sm animate-scale-in">
            <FileText className="w-4 h-4 text-primary flex-shrink-0" />
            <span className="max-w-[240px] truncate font-medium">{selectedFile.name}</span>
            <span className="text-xs text-muted-foreground">
              {formatFileSize(selectedFile.size)}
            </span>
            {!disabled && (
              <Button
                variant="ghost"
                size="sm"
                onClick={handleRemove}
                className="h-6 w-6 p-0 hover:bg-destructive/10 hover:text-destructive flex-shrink-0"
                aria-label={`${label} 제거`}
              >
                <X className="w-3.5 h-3.5" />
              </Button>
            )}
          </div>
        ) : (
          <div {...getRootProps()} className="inline-block">
            <input {...getInputProps()} />
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={disabled}
              className="gap-1.5"
            >
              <Plus className="w-4 h-4" />
              {label}
              {description && (
                <span className="text-xs font-normal text-muted-foreground">
                  {description}
                </span>
              )}
            </Button>
          </div>
        )}
        {error && (
          <p className="text-sm text-destructive flex items-center gap-2 animate-slide-in">
            <span className="w-1 h-1 rounded-full bg-destructive" />
            {error}
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">
        {label}
        {required && <span className="text-destructive"> *</span>}
      </label>
      <Card
        {...getRootProps()}
        className={`
          relative p-4 border-2 border-dashed cursor-pointer transition-all duration-300
          overflow-hidden
          ${isDragActive ? "border-primary bg-primary/5 scale-[1.01]" : "border-border"}
          ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-primary"}
          ${error ? "border-destructive" : ""}
          ${selectedFile ? "border-primary" : ""}
        `}
      >
        <input {...getInputProps()} />

        {/* Background decoration */}
        {isDragActive && (
          <div className="absolute inset-0 bg-primary/5 animate-pulse" />
        )}

        <div className="relative z-10">
          {selectedFile ? (
            <div className="flex items-center justify-between gap-3 animate-scale-in">
              <div className="flex items-center gap-2 min-w-0">
                <div className="p-1.5 rounded brand-gradient flex-shrink-0">
                  <FileText className="w-4 h-4 text-primary-foreground" />
                </div>
                <div className="text-left min-w-0">
                  <p className="font-medium text-sm truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>
              {!disabled && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleRemove}
                  className="h-8 w-8 p-0 hover:bg-destructive/10 hover:text-destructive flex-shrink-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-3">
              <div className={`
                p-2 rounded-lg border border-border
                ${isDragActive ? "animate-bounce-subtle border-primary bg-primary/10" : ""}
              `}>
                {isDragActive ? (
                  <CloudUpload className="w-5 h-5 text-primary animate-pulse" />
                ) : (
                  <Upload className="w-5 h-5 text-primary" />
                )}
              </div>
              <div className="text-left">
                <p className={`text-sm font-medium transition-colors ${isDragActive ? "text-primary" : ""}`}>
                  {isDragActive
                    ? "여기에 파일을 놓으세요"
                    : dragEnabled
                      ? "클릭하거나 파일을 끌어다 놓으세요"
                      : "클릭해서 파일 선택"}
                </p>
                {description && (
                  <p className="text-xs text-muted-foreground">{description}</p>
                )}
              </div>
            </div>
          )}
        </div>
      </Card>
      {error && (
        <p className="text-sm text-destructive flex items-center gap-2 animate-slide-in">
          <span className="w-1 h-1 rounded-full bg-destructive" />
          {error}
        </p>
      )}
    </div>
  );
}

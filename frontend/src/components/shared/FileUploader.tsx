"use client";

import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
import { Upload, FileText, X, CloudUpload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

interface FileUploaderProps {
  accept: Record<string, string[]>;
  maxSizeMB?: number;
  onFileSelect: (file: File | null) => void;
  selectedFile: File | null;
  label: string;
  description?: string;
  disabled?: boolean;
}

export function FileUploader({
  accept,
  maxSizeMB = 200,
  onFileSelect,
  selectedFile,
  label,
  description,
  disabled = false,
}: FileUploaderProps) {
  const [error, setError] = useState<string | null>(null);

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

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">{label}</label>
      <Card
        {...getRootProps()}
        className={`
          relative p-4 border-2 border-dashed cursor-pointer transition-all duration-300
          overflow-hidden
          ${isDragActive ? "border-foreground scale-[1.01]" : "border-border"}
          ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-foreground"}
          ${error ? "border-destructive" : ""}
          ${selectedFile ? "border-foreground" : ""}
        `}
      >
        <input {...getInputProps()} />

        {/* Background decoration */}
        {isDragActive && (
          <div className="absolute inset-0 brand-gradient opacity-5 animate-pulse" />
        )}

        <div className="relative z-10">
          {selectedFile ? (
            <div className="flex items-center justify-between gap-3 animate-scale-in">
              <div className="flex items-center gap-2 min-w-0">
                <div className="p-1.5 rounded brand-gradient flex-shrink-0">
                  <FileText className="w-4 h-4 text-background" />
                </div>
                <div className="text-left min-w-0">
                  <p className="font-medium text-sm truncate">
                    {selectedFile.name}
                  </p>
                  <p className="text-xs text-foreground/60">
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
                ${isDragActive ? "animate-bounce-subtle border-foreground" : ""}
              `}>
                {isDragActive ? (
                  <CloudUpload className="w-5 h-5 text-foreground animate-pulse" />
                ) : (
                  <Upload className="w-5 h-5 text-foreground" />
                )}
              </div>
              <div className="text-left">
                <p className={`text-sm font-medium transition-colors ${isDragActive ? "text-foreground" : ""}`}>
                  {isDragActive ? "파일을 여기에 놓으세요" : "클릭 또는 드래그"}
                </p>
                {description && (
                  <p className="text-xs text-foreground/60">{description}</p>
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

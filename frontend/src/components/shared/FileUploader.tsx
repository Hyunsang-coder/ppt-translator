"use client";

import { useCallback, useState } from "react";
import { useDropzone, type FileRejection } from "react-dropzone";
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
  maxSizeMB = 50,
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
          p-6 border-2 border-dashed cursor-pointer transition-colors
          ${isDragActive ? "border-primary bg-primary/5" : "border-muted-foreground/25"}
          ${disabled ? "opacity-50 cursor-not-allowed" : "hover:border-primary/50"}
          ${error ? "border-destructive" : ""}
        `}
      >
        <input {...getInputProps()} />
        <div className="flex flex-col items-center justify-center text-center">
          {selectedFile ? (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <svg
                  className="w-8 h-8 text-primary"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                <div className="text-left">
                  <p className="font-medium text-sm truncate max-w-[200px]">{selectedFile.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(selectedFile.size)}
                  </p>
                </div>
              </div>
              {!disabled && (
                <Button variant="outline" size="sm" onClick={handleRemove}>
                  파일 제거
                </Button>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              <svg
                className="w-10 h-10 mx-auto text-muted-foreground"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                />
              </svg>
              <div>
                <p className="text-sm font-medium">
                  {isDragActive ? "파일을 여기에 놓으세요" : "파일을 드래그하거나 클릭하세요"}
                </p>
                {description && <p className="text-xs text-muted-foreground mt-1">{description}</p>}
              </div>
            </div>
          )}
        </div>
      </Card>
      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}

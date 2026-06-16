/**
 * Hook for managing text extraction workflow
 */

import { useCallback, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api-client";
import { saveBlob } from "@/lib/save-file";
import { useExtractionStore } from "@/stores/extraction-store";
import { useTranslationStore } from "@/stores/translation-store";
import { getFileKey, useSharedFileStore } from "@/stores/shared-file-store";

const EXTRACTION_TIMEOUT_MS = 5 * 60 * 1000;

export function useExtraction() {
  const { pptFile, setPptFile: setSharedPptFile } = useSharedFileStore();
  const {
    settings,
    status,
    errorMessage,
    markdown,
    slideCount,
    updateSettings,
    setStatus,
    setErrorMessage,
    setResult,
    resetForPptFileChange,
    reset,
  } = useExtractionStore();
  const abortControllerRef = useRef<AbortController | null>(null);

  useEffect(() => {
    return () => {
      abortControllerRef.current?.abort();
    };
  }, []);

  const startExtraction = useCallback(async () => {
    if (!pptFile) {
      setErrorMessage("파일을 선택해주세요.");
      return;
    }

    let timeout: number | undefined;

    try {
      setStatus("extracting");
      setErrorMessage(null);

      abortControllerRef.current?.abort();
      const abortController = new AbortController();
      abortControllerRef.current = abortController;
      timeout = window.setTimeout(() => abortController.abort(), EXTRACTION_TIMEOUT_MS);

      const response = await apiClient.extractText(pptFile, settings, abortController.signal);
      setResult(response.markdown, response.slide_count);
    } catch (err) {
      const message =
        err instanceof DOMException && err.name === "AbortError"
          ? "마크다운 추출 시간이 초과되었습니다. 파일 크기를 줄이거나 다시 시도해주세요."
          : err instanceof Error
            ? err.message
            : "추출 중 오류가 발생했습니다.";
      setErrorMessage(message);
      setStatus("failed");
    } finally {
      if (timeout !== undefined) {
        window.clearTimeout(timeout);
      }
      abortControllerRef.current = null;
    }
  }, [pptFile, settings, setStatus, setErrorMessage, setResult]);

  const copyToClipboard = useCallback(async () => {
    if (!markdown) return;

    try {
      await navigator.clipboard.writeText(markdown);
      return true;
    } catch {
      return false;
    }
  }, [markdown]);

  const downloadMarkdown = useCallback(() => {
    if (!markdown || !pptFile) return;

    const blob = new Blob([markdown], { type: "text/markdown" });
    const filename = `${pptFile.name.replace(/\.[^/.]+$/, "")}.md`;
    void saveBlob(blob, filename);
  }, [markdown, pptFile]);

  const setPptFile = useCallback((file: File | null) => {
    const currentFileKey = getFileKey(useSharedFileStore.getState().pptFile);
    const nextFileKey = getFileKey(file);
    if (currentFileKey === nextFileKey) {
      return;
    }

    setSharedPptFile(file);
    resetForPptFileChange();
    useTranslationStore.getState().resetForPptFileChange();
  }, [resetForPptFileChange, setSharedPptFile]);

  return {
    // State
    pptFile,
    settings,
    status,
    errorMessage,
    markdown,
    slideCount,

    // Computed
    isIdle: status === "idle",
    isExtracting: status === "extracting",
    isCompleted: status === "completed",
    isFailed: status === "failed",
    canStart: pptFile !== null && (status === "idle" || status === "completed" || status === "failed"),
    hasResult: markdown !== null,

    // Actions
    setPptFile,
    updateSettings,
    startExtraction,
    copyToClipboard,
    downloadMarkdown,
    reset,
  };
}

/**
 * Hook for managing text extraction workflow
 */

import { useCallback } from "react";
import { apiClient } from "@/lib/api-client";
import { useExtractionStore } from "@/stores/extraction-store";

export function useExtraction() {
  const {
    pptFile,
    settings,
    status,
    errorMessage,
    markdown,
    slideCount,
    setPptFile,
    updateSettings,
    setStatus,
    setErrorMessage,
    setResult,
    reset,
  } = useExtractionStore();

  const startExtraction = useCallback(async () => {
    if (!pptFile) {
      setErrorMessage("파일을 선택해주세요.");
      return;
    }

    try {
      setStatus("extracting");
      setErrorMessage(null);

      const response = await apiClient.extractText(pptFile, settings);
      setResult(response.markdown, response.slide_count);
    } catch (err) {
      const message = err instanceof Error ? err.message : "추출 중 오류가 발생했습니다.";
      setErrorMessage(message);
      setStatus("failed");
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
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${pptFile.name.replace(/\.[^/.]+$/, "")}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [markdown, pptFile]);

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

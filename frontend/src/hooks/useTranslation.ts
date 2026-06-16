/**
 * Hook for managing translation workflow
 */

import { useCallback, useEffect, useRef } from "react";
import { apiClient } from "@/lib/api-client";
import { saveBlob } from "@/lib/save-file";
import { createSSEClient, SSEClient } from "@/lib/sse-client";
import { useTranslationStore } from "@/stores/translation-store";
import { useExtractionStore } from "@/stores/extraction-store";
import { getFileKey, useSharedFileStore } from "@/stores/shared-file-store";
import type { JobProgress, SSEEvent } from "@/types/api";

export function useTranslation() {
  const { pptFile, setPptFile: setSharedPptFile } = useSharedFileStore();
  const {
    glossaryFile,
    settings,
    jobId,
    status,
    progress,
    errorMessage,
    resultFilename,
    logs,
    setGlossaryFile,
    updateSettings,
    setJobId,
    setStatus,
    setProgress,
    setErrorMessage,
    setResultFilename,
    addLog,
    clearLogs,
    resetForPptFileChange,
    resetJobState,
    reset,
  } = useTranslationStore();

  const sseClientRef = useRef<SSEClient | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      sseClientRef.current?.close();
      abortControllerRef.current?.abort();
    };
  }, []);

  const startTranslation = useCallback(async () => {
    if (!pptFile) {
      setErrorMessage("파일을 선택해주세요.");
      return;
    }

    try {
      setStatus("uploading");
      setErrorMessage(null);
      clearLogs();
      addLog("번역 작업을 시작합니다...", "info");

      // Create abort controller for upload cancellation
      const abortController = new AbortController();
      abortControllerRef.current = abortController;

      // Create job
      const response = await apiClient.createJob(pptFile, settings, glossaryFile ?? undefined, abortController.signal);
      abortControllerRef.current = null;
      setJobId(response.job_id);
      addLog(`작업 생성 완료 (ID: ${response.job_id.slice(0, 8)}...)`, "info");

      // Start polling for progress
      setStatus("translating");

      sseClientRef.current = createSSEClient("", {
        jobId: response.job_id,
        getJobStatus: (id) => apiClient.getJobStatus(id),
        onStarted: () => {
          addLog("번역이 시작되었습니다.", "info");
        },
        onProgress: (event: SSEEvent) => {
          const data = event.data as unknown as JobProgress;
          setProgress(data);
          if (data.message) {
            addLog(data.message, "info");
          }
        },
        onComplete: async (event: SSEEvent) => {
          const data = event.data as unknown as { source_language?: string; target_language?: string };
          addLog("번역이 완료되었습니다!", "success");
          if (data.source_language) {
            addLog(`감지된 소스 언어: ${data.source_language}`, "info");
          }
          if (data.target_language) {
            addLog(`번역된 언어: ${data.target_language}`, "info");
          }
          setStatus("completed");

          // Fetch result filename
          try {
            const jobStatus = await apiClient.getJobStatus(response.job_id);
            if (jobStatus.state === "completed") {
              // Filename will be set when downloading
              setResultFilename("ready");
            }
          } catch {
            // Ignore, filename will be fetched on download
          }
        },
        onError: (event: SSEEvent) => {
          const data = event.data as { message?: string };
          const message = data.message || "번역 중 오류가 발생했습니다.";
          setErrorMessage(message);
          setStatus("failed");
          addLog(message, "error");
        },
        onCancelled: () => {
          setStatus("cancelled");
          addLog("번역이 취소되었습니다.", "warning");
        },
      });
    } catch (err) {
      // Ignore abort errors (handled by cancelTranslation)
      if (err instanceof DOMException && err.name === "AbortError") return;

      // Server busy (429) - reset to idle so user can retry easily
      const is429 =
        err instanceof Error &&
        "status" in err &&
        (err as { status: number }).status === 429;

      const message = err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.";
      setErrorMessage(message);
      setStatus(is429 ? "idle" : "failed");
      addLog(message, is429 ? "warning" : "error");
    }
  }, [
    pptFile,
    glossaryFile,
    settings,
    setStatus,
    setErrorMessage,
    clearLogs,
    addLog,
    setJobId,
    setProgress,
    setResultFilename,
  ]);

  const cancelTranslation = useCallback(async () => {
    // Abort upload if still in progress
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;

    // Close SSE connection
    sseClientRef.current?.close();

    const currentJobId = useTranslationStore.getState().jobId;
    if (currentJobId) {
      try {
        await apiClient.cancelJob(currentJobId);
      } catch (err) {
        const message = err instanceof Error ? err.message : "취소 실패";
        addLog(`취소 실패: ${message}`, "error");
      }
    }

    setStatus("cancelled");
    addLog("번역이 취소되었습니다.", "warning");
  }, [setStatus, addLog]);

  const downloadResult = useCallback(async () => {
    const currentJobId = useTranslationStore.getState().jobId;
    if (!currentJobId) return;

    try {
      addLog("파일 다운로드 중...", "info");
      const { blob, filename } = await apiClient.downloadJobResult(currentJobId);

      await saveBlob(blob, filename);

      setResultFilename(filename);
      addLog(`다운로드 완료: ${filename}`, "success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "다운로드 실패";
      addLog(`다운로드 실패: ${message}`, "error");
    }
  }, [addLog, setResultFilename]);

  const retranslate = useCallback(async () => {
    sseClientRef.current?.close();
    resetJobState();
    // startTranslation은 현재 store 상태(pptFile, settings 등)를 그대로 사용
    await startTranslation();
  }, [resetJobState, startTranslation]);

  // Wrap setPptFile so translation/extraction state follows the shared file.
  const handleSetPptFile = useCallback((file: File | null) => {
    const currentFileKey = getFileKey(useSharedFileStore.getState().pptFile);
    const nextFileKey = getFileKey(file);
    if (currentFileKey === nextFileKey) {
      return;
    }

    setSharedPptFile(file);
    resetForPptFileChange();
    useExtractionStore.getState().resetForPptFileChange();
  }, [resetForPptFileChange, setSharedPptFile]);

  const resetTranslation = useCallback(() => {
    sseClientRef.current?.close();
    setSharedPptFile(null);
    useExtractionStore.getState().resetForPptFileChange();
    reset();
  }, [reset, setSharedPptFile]);

  return {
    // State
    pptFile,
    glossaryFile,
    settings,
    jobId,
    status,
    progress,
    errorMessage,
    resultFilename,
    logs,

    // Computed
    isIdle: status === "idle",
    isTranslating: status === "uploading" || status === "translating",
    isCompleted: status === "completed",
    isFailed: status === "failed",
    canStart:
      pptFile !== null &&
      status === "idle" &&
      // 타겟 언어 필수 선택
      settings.targetLang !== "" &&
      settings.targetLang !== "Auto" &&
      // 직접 입력 모드일 때 파일명 필수
      (settings.filenameSettings.mode !== "custom" ||
        settings.filenameSettings.customName.trim() !== ""),
    canCancel: status === "uploading" || status === "translating",
    canDownload: status === "completed" && jobId !== null,

    // Actions
    setPptFile: handleSetPptFile,
    setGlossaryFile,
    updateSettings,
    startTranslation,
    cancelTranslation,
    downloadResult,
    retranslate,
    reset: resetTranslation,
    addLog,
  };
}

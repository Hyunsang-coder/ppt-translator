/**
 * Hook for managing translation workflow
 */

import { useCallback, useRef } from "react";
import { apiClient } from "@/lib/api-client";
import { createSSEClient, SSEClient } from "@/lib/sse-client";
import { useTranslationStore } from "@/stores/translation-store";
import type { JobProgress, SSEEvent } from "@/types/api";

export function useTranslation() {
  const {
    pptFile,
    glossaryFile,
    settings,
    jobId,
    status,
    progress,
    errorMessage,
    resultFilename,
    logs,
    setPptFile,
    setGlossaryFile,
    updateSettings,
    setJobId,
    setStatus,
    setProgress,
    setErrorMessage,
    setResultFilename,
    addLog,
    clearLogs,
    reset,
  } = useTranslationStore();

  const sseClientRef = useRef<SSEClient | null>(null);

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

      // Create job
      const response = await apiClient.createJob(pptFile, settings, glossaryFile ?? undefined);
      setJobId(response.job_id);
      addLog(`작업 생성 완료 (ID: ${response.job_id.slice(0, 8)}...)`, "info");

      // Start SSE connection
      setStatus("translating");
      const eventsUrl = apiClient.getJobEventsUrl(response.job_id);

      sseClientRef.current = createSSEClient(eventsUrl, {
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
      const message = err instanceof Error ? err.message : "알 수 없는 오류가 발생했습니다.";
      setErrorMessage(message);
      setStatus("failed");
      addLog(message, "error");
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
    if (!jobId) return;

    try {
      sseClientRef.current?.close();
      await apiClient.cancelJob(jobId);
      setStatus("cancelled");
      addLog("번역이 취소되었습니다.", "warning");
    } catch (err) {
      const message = err instanceof Error ? err.message : "취소 실패";
      addLog(`취소 실패: ${message}`, "error");
    }
  }, [jobId, setStatus, addLog]);

  const downloadResult = useCallback(async () => {
    if (!jobId) return;

    try {
      addLog("파일 다운로드 중...", "info");
      const { blob, filename } = await apiClient.downloadJobResult(jobId);

      // Create download link
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      setResultFilename(filename);
      addLog(`다운로드 완료: ${filename}`, "success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "다운로드 실패";
      addLog(`다운로드 실패: ${message}`, "error");
    }
  }, [jobId, addLog, setResultFilename]);

  const resetTranslation = useCallback(() => {
    sseClientRef.current?.close();
    reset();
  }, [reset]);

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
    canStart: pptFile !== null && status === "idle",
    canCancel: status === "uploading" || status === "translating",
    canDownload: status === "completed" && jobId !== null,

    // Actions
    setPptFile,
    setGlossaryFile,
    updateSettings,
    startTranslation,
    cancelTranslation,
    downloadResult,
    reset: resetTranslation,
    addLog,
  };
}

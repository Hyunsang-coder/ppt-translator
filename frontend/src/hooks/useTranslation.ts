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
    generatedContext,
    isGeneratingContext,
    generatedInstructions,
    isGeneratingInstructions,
    jobId,
    status,
    progress,
    errorMessage,
    resultFilename,
    logs,
    setPptFile,
    setGlossaryFile,
    updateSettings,
    setGeneratedContext,
    setIsGeneratingContext,
    setGeneratedInstructions,
    setIsGeneratingInstructions,
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

  const generateContext = useCallback(async () => {
    if (!pptFile) return;

    try {
      setIsGeneratingContext(true);
      addLog("컨텍스트 생성을 시작합니다...", "info");

      // Step 1: Extract text from PPT
      addLog("PPT에서 텍스트를 추출하는 중...", "info");
      const extractionResult = await apiClient.extractText(pptFile, {
        figures: "omit",
        charts: "labels",
        withNotes: false,
        tableHeader: true,
      });
      addLog(`${extractionResult.slide_count}개 슬라이드에서 텍스트 추출 완료`, "info");

      // Step 2: Summarize the extracted text
      addLog("AI가 프레젠테이션 내용을 요약하는 중...", "info");
      const summaryResult = await apiClient.summarizeText(
        extractionResult.markdown,
        settings.provider,
        "gpt-4o-mini"  // Use fast model for summarization
      );

      setGeneratedContext(summaryResult.summary);
      addLog("컨텍스트 생성 완료!", "success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "컨텍스트 생성 실패";
      addLog(`컨텍스트 생성 실패: ${message}`, "error");
    } finally {
      setIsGeneratingContext(false);
    }
  }, [pptFile, settings.provider, addLog, setGeneratedContext, setIsGeneratingContext]);

  const generateInstructions = useCallback(async () => {
    if (!settings.targetLang) return;

    try {
      setIsGeneratingInstructions(true);
      addLog("번역 지침을 생성합니다...", "info");

      // Call API to generate instructions based on target language
      const response = await apiClient.generateInstructions(
        settings.targetLang,
        settings.provider,
        "gpt-4o-mini"
      );

      setGeneratedInstructions(response.instructions);
      addLog("번역 지침 생성 완료!", "success");
    } catch (err) {
      const message = err instanceof Error ? err.message : "지침 생성 실패";
      addLog(`지침 생성 실패: ${message}`, "error");
    } finally {
      setIsGeneratingInstructions(false);
    }
  }, [settings.targetLang, settings.provider, addLog, setGeneratedInstructions, setIsGeneratingInstructions]);

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

      // Merge generated context and instructions into settings if available
      const effectiveSettings = {
        ...settings,
        context: generatedContext || settings.context,
        instructions: generatedInstructions || settings.instructions,
      };

      // Create job
      const response = await apiClient.createJob(pptFile, effectiveSettings, glossaryFile ?? undefined);
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
    generatedContext,
    generatedInstructions,
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
    generatedContext,
    isGeneratingContext,
    generatedInstructions,
    isGeneratingInstructions,
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
      !isGeneratingContext &&
      !isGeneratingInstructions &&
      // 타겟 언어 필수 선택
      settings.targetLang !== "" &&
      settings.targetLang !== "Auto" &&
      // 직접 입력 모드일 때 파일명 필수
      (settings.filenameSettings.mode !== "custom" ||
        settings.filenameSettings.customName.trim() !== ""),
    canCancel: status === "uploading" || status === "translating",
    canDownload: status === "completed" && jobId !== null,

    // Actions
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
    reset: resetTranslation,
    addLog,
  };
}

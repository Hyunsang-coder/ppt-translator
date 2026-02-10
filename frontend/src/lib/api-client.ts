/**
 * API Client for PPT Translator Backend
 */

import type {
  ConfigResponse,
  ExtractionResponse,
  ExtractionSettings,
  JobCreateResponse,
  JobStatusResponse,
  LanguageInfo,
  ModelInfo,
  SummarizeResponse,
  TranslationSettings,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public detail?: string
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined;
    try {
      const data = await response.json();
      detail = data.detail;
    } catch {
      // ignore JSON parse errors
    }

    // User-friendly message for 429 Too Many Requests
    if (response.status === 429) {
      throw new ApiError(
        detail || "서버가 바쁩니다. 잠시 후 다시 시도해주세요.",
        response.status,
        detail
      );
    }

    throw new ApiError(
      detail || `Request failed with status ${response.status}`,
      response.status,
      detail
    );
  }
  return response.json();
}

export const apiClient = {
  /**
   * Get available models
   */
  async getModels(provider?: string): Promise<ModelInfo[]> {
    const url = new URL(`${API_BASE}/api/v1/models`);
    if (provider) {
      url.searchParams.set("provider", provider);
    }
    try {
      const response = await fetch(url.toString());
      const data = await handleResponse<{ models: ModelInfo[] }>(response);
      return data.models;
    } catch {
      // Return empty array when backend is unavailable
      return [];
    }
  },

  /**
   * Get supported languages
   */
  async getLanguages(): Promise<LanguageInfo[]> {
    try {
      const response = await fetch(`${API_BASE}/api/v1/languages`);
      const data = await handleResponse<{ languages: LanguageInfo[] }>(response);
      return data.languages;
    } catch {
      // Return empty array when backend is unavailable
      return [];
    }
  },

  /**
   * Get application config
   */
  async getConfig(): Promise<ConfigResponse | null> {
    try {
      const response = await fetch(`${API_BASE}/api/v1/config`);
      return handleResponse<ConfigResponse>(response);
    } catch {
      // Return null when backend is unavailable
      return null;
    }
  },

  /**
   * Create a translation job
   */
  async createJob(
    pptFile: File,
    settings: TranslationSettings,
    glossaryFile?: File,
    signal?: AbortSignal
  ): Promise<JobCreateResponse> {
    const formData = new FormData();
    formData.append("ppt_file", pptFile);
    formData.append("source_lang", settings.sourceLang);
    formData.append("target_lang", settings.targetLang);
    formData.append("provider", settings.provider);
    formData.append("model", settings.model);
    formData.append("preprocess_repetitions", String(settings.preprocessRepetitions));
    formData.append("translate_notes", String(settings.translateNotes));
    if (settings.context) {
      formData.append("context", settings.context);
    }
    if (settings.instructions) {
      formData.append("instructions", settings.instructions);
    }
    if (glossaryFile) {
      formData.append("glossary_file", glossaryFile);
    }
    // Filename settings
    formData.append("filename_settings", JSON.stringify(settings.filenameSettings));
    // Text fitting settings
    formData.append("text_fit_mode", settings.textFitMode);
    formData.append("min_font_ratio", String(settings.minFontRatio));
    // Image compression
    formData.append("compress_images", settings.imageCompression);
    // Length limit
    if (settings.lengthLimit !== null) {
      formData.append("length_limit", String(settings.lengthLimit));
    }

    const response = await fetch(`${API_BASE}/api/v1/jobs`, {
      method: "POST",
      body: formData,
      signal,
    });
    return handleResponse<JobCreateResponse>(response);
  },

  /**
   * Get job status
   */
  async getJobStatus(jobId: string): Promise<JobStatusResponse> {
    const response = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`);
    return handleResponse<JobStatusResponse>(response);
  },

  /**
   * Cancel a job
   */
  async cancelJob(jobId: string): Promise<void> {
    const response = await fetch(`${API_BASE}/api/v1/jobs/${jobId}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      throw new ApiError("Failed to cancel job", response.status);
    }
  },

  /**
   * Download job result
   */
  async downloadJobResult(jobId: string): Promise<{ blob: Blob; filename: string }> {
    const response = await fetch(`${API_BASE}/api/v1/jobs/${jobId}/result`);
    if (!response.ok) {
      throw new ApiError("Failed to download result", response.status);
    }

    const blob = await response.blob();
    const contentDisposition = response.headers.get("Content-Disposition");
    let filename = "translated.pptx";

    if (contentDisposition) {
      // Parse filename from Content-Disposition header
      const match = contentDisposition.match(/filename\*=UTF-8''(.+)/);
      if (match) {
        filename = decodeURIComponent(match[1]);
      }
    }

    return { blob, filename };
  },

  /**
   * Get SSE event stream URL
   */
  getJobEventsUrl(jobId: string): string {
    return `${API_BASE}/api/v1/jobs/${jobId}/events`;
  },

  /**
   * Extract text from PPT
   */
  async extractText(pptFile: File, settings: ExtractionSettings, signal?: AbortSignal): Promise<ExtractionResponse> {
    const formData = new FormData();
    formData.append("ppt_file", pptFile);
    formData.append("figures", settings.figures);
    formData.append("charts", settings.charts);
    formData.append("with_notes", String(settings.withNotes));
    formData.append("table_header", String(settings.tableHeader));

    const response = await fetch(`${API_BASE}/api/v1/extract`, {
      method: "POST",
      body: formData,
      signal,
    });
    return handleResponse<ExtractionResponse>(response);
  },

  /**
   * Summarize presentation content for translation context
   */
  async summarizeText(
    markdown: string,
    provider?: string,
    model?: string,
    signal?: AbortSignal
  ): Promise<SummarizeResponse> {
    const response = await fetch(`${API_BASE}/api/v1/summarize`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        markdown,
        ...(provider && { provider }),
        ...(model && { model }),
      }),
      signal,
    });
    return handleResponse<SummarizeResponse>(response);
  },

  /**
   * Generate translation instructions based on target language and document content
   */
  async generateInstructions(
    targetLang: string,
    markdown: string,
    provider?: string,
    model?: string,
    signal?: AbortSignal
  ): Promise<{ instructions: string }> {
    const response = await fetch(`${API_BASE}/api/v1/generate-instructions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target_lang: targetLang,
        markdown,
        ...(provider && { provider }),
        ...(model && { model }),
      }),
      signal,
    });
    return handleResponse<{ instructions: string }>(response);
  },

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${API_BASE}/health`);
      return response.ok;
    } catch {
      return false;
    }
  },
};

export { ApiError };

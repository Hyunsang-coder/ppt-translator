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
  TranslationSettings,
} from "@/types/api";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
    const response = await fetch(url.toString());
    const data = await handleResponse<{ models: ModelInfo[] }>(response);
    return data.models;
  },

  /**
   * Get supported languages
   */
  async getLanguages(): Promise<LanguageInfo[]> {
    const response = await fetch(`${API_BASE}/api/v1/languages`);
    const data = await handleResponse<{ languages: LanguageInfo[] }>(response);
    return data.languages;
  },

  /**
   * Get application config
   */
  async getConfig(): Promise<ConfigResponse> {
    const response = await fetch(`${API_BASE}/api/v1/config`);
    return handleResponse<ConfigResponse>(response);
  },

  /**
   * Create a translation job
   */
  async createJob(
    pptFile: File,
    settings: TranslationSettings,
    glossaryFile?: File
  ): Promise<JobCreateResponse> {
    const formData = new FormData();
    formData.append("ppt_file", pptFile);
    formData.append("source_lang", settings.sourceLang);
    formData.append("target_lang", settings.targetLang);
    formData.append("provider", settings.provider);
    formData.append("model", settings.model);
    formData.append("preprocess_repetitions", String(settings.preprocessRepetitions));
    if (settings.userPrompt) {
      formData.append("user_prompt", settings.userPrompt);
    }
    if (glossaryFile) {
      formData.append("glossary_file", glossaryFile);
    }

    const response = await fetch(`${API_BASE}/api/v1/jobs`, {
      method: "POST",
      body: formData,
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
  async extractText(pptFile: File, settings: ExtractionSettings): Promise<ExtractionResponse> {
    const formData = new FormData();
    formData.append("ppt_file", pptFile);
    formData.append("figures", settings.figures);
    formData.append("charts", settings.charts);
    formData.append("with_notes", String(settings.withNotes));
    formData.append("table_header", String(settings.tableHeader));

    const response = await fetch(`${API_BASE}/api/v1/extract`, {
      method: "POST",
      body: formData,
    });
    return handleResponse<ExtractionResponse>(response);
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

/**
 * API Client for PPT Translator Backend
 */

import type {
  ConfigResponse,
  ExtractionResponse,
  ExtractionSettings,
  FragmentEditRequest,
  FragmentEditResponse,
  FragmentProposalRequest,
  FragmentProposalResponse,
  ApplyProposalResponse,
  FragmentsResponse,
  JobCreateResponse,
  JobStatusResponse,
  LanguageInfo,
  ModelInfo,
  ReviewMutationResponse,
  TranslationSettings,
} from "@/types/api";
import { ensureApiBase } from "@/lib/api-base";

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

async function apiUrl(path: string): Promise<string> {
  const base = await ensureApiBase();
  return `${base}${path}`;
}

export const apiClient = {
  /**
   * Get available models
   */
  async getModels(provider?: string): Promise<ModelInfo[]> {
    try {
      const url = new URL(await apiUrl("/api/v1/models"), window.location.origin);
      if (provider) {
        url.searchParams.set("provider", provider);
      }
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
      const response = await fetch(await apiUrl("/api/v1/languages"));
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
      const response = await fetch(await apiUrl("/api/v1/config"));
      return handleResponse<ConfigResponse>(response);
    } catch {
      // Return null when backend is unavailable
      return null;
    }
  },

  /**
   * Parse an Excel glossary into structured entries (for the in-app editor import).
   */
  async parseGlossaryFile(
    glossaryFile: File
  ): Promise<{ entries: { source: string; target: string }[]; count: number }> {
    const formData = new FormData();
    formData.append("glossary_file", glossaryFile);
    const response = await fetch(await apiUrl("/api/v1/glossary/parse"), {
      method: "POST",
      body: formData,
    });
    return handleResponse(response);
  },

  /**
   * Create a translation job
   */
  async createJob(
    pptFile: File,
    settings: TranslationSettings,
    glossaryJson?: string | null,
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
    if (glossaryJson) {
      formData.append("glossary_json", glossaryJson);
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

    const response = await fetch(await apiUrl("/api/v1/jobs"), {
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
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}`));
    return handleResponse<JobStatusResponse>(response);
  },

  /**
   * Cancel a job
   */
  async cancelJob(jobId: string): Promise<void> {
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}`), {
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
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}/result`));
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
   * List reviewable fragments (source/target + detection badges) for a job.
   */
  async getJobFragments(jobId: string): Promise<FragmentsResponse> {
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}/fragments`));
    return handleResponse<FragmentsResponse>(response);
  },

  /**
   * Merge glossary terms into a completed job's review session (resweeps findings).
   */
  async updateJobGlossary(
    jobId: string,
    entries: Record<string, string>
  ): Promise<{ count: number; revision: number; dirty: boolean }> {
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}/glossary`), {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ entries }),
    });
    return handleResponse(response);
  },

  /**
   * Edit, re-translate, or ignore a single fragment (WP-C5).
   */
  async editJobFragment(
    jobId: string,
    index: number,
    body: FragmentEditRequest
  ): Promise<FragmentEditResponse> {
    const response = await fetch(
      await apiUrl(`/api/v1/jobs/${jobId}/fragments/${index}`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );
    return handleResponse<FragmentEditResponse>(response);
  },

  async proposeJobFragment(
    jobId: string,
    index: number,
    body: FragmentProposalRequest
  ): Promise<FragmentProposalResponse> {
    const response = await fetch(
      await apiUrl(`/api/v1/jobs/${jobId}/fragments/${index}/proposals`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      }
    );
    return handleResponse<FragmentProposalResponse>(response);
  },

  async applyJobFragmentProposal(
    jobId: string,
    proposalId: string,
    expectedRevision: number
  ): Promise<ApplyProposalResponse> {
    const response = await fetch(
      await apiUrl(`/api/v1/jobs/${jobId}/proposals/${proposalId}/apply`),
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_revision: expectedRevision }),
      }
    );
    return handleResponse<ApplyProposalResponse>(response);
  },

  async applyPartialCandidates(
    jobId: string,
    body: {
      indices: number[];
      old_phrase: string;
      new_phrase: string;
      expected_revision: number;
    }
  ): Promise<ReviewMutationResponse> {
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}/review/partial`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return handleResponse<ReviewMutationResponse>(response);
  },

  async undoReview(jobId: string, expectedRevision: number): Promise<ReviewMutationResponse> {
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}/review/undo`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_revision: expectedRevision }),
    });
    return handleResponse<ReviewMutationResponse>(response);
  },

  async commitReview(jobId: string, expectedRevision: number): Promise<ReviewMutationResponse> {
    const response = await fetch(await apiUrl(`/api/v1/jobs/${jobId}/review/commit`), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ expected_revision: expectedRevision }),
    });
    return handleResponse<ReviewMutationResponse>(response);
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

    const response = await fetch(await apiUrl("/api/v1/extract"), {
      method: "POST",
      body: formData,
      signal,
    });
    return handleResponse<ExtractionResponse>(response);
  },

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(await apiUrl("/health"));
      return response.ok;
    } catch {
      return false;
    }
  },
};

export { ApiError };

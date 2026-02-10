/**
 * Hook for fetching and caching API configuration
 */

import { useEffect, useState } from "react";
import { apiClient } from "@/lib/api-client";
import type { ConfigResponse, LanguageInfo, ModelInfo } from "@/types/api";

// Fallback data when backend is unavailable
const FALLBACK_MODELS: ModelInfo[] = [
  { id: "gpt-5.2", name: "GPT-5.2", provider: "openai" },
  { id: "gpt-5-mini", name: "GPT-5 Mini", provider: "openai" },
  { id: "claude-opus-4-6", name: "Claude Opus 4.6", provider: "anthropic" },
  { id: "claude-sonnet-4-5-20250929", name: "Claude Sonnet 4.5", provider: "anthropic" },
  { id: "claude-haiku-4-5-20251001", name: "Claude Haiku 4.5", provider: "anthropic" },
];

const FALLBACK_LANGUAGES: LanguageInfo[] = [
  { code: "Auto", name: "Auto (자동 감지)" },
  { code: "한국어", name: "한국어" },
  { code: "영어", name: "English" },
  { code: "일본어", name: "日本語" },
  { code: "중국어", name: "中文" },
  { code: "스페인어", name: "Español" },
  { code: "프랑스어", name: "Français" },
  { code: "독일어", name: "Deutsch" },
];

const FALLBACK_CONFIG: ConfigResponse = {
  max_upload_size_mb: 200,
  providers: ["openai", "anthropic"],
  default_provider: "anthropic",
  default_model: "claude-sonnet-4-5-20250929",
};

interface ConfigState {
  models: ModelInfo[];
  languages: LanguageInfo[];
  config: ConfigResponse | null;
  isLoading: boolean;
  error: string | null;
  isBackendConnected: boolean;
}

export function useConfig() {
  const [state, setState] = useState<ConfigState>({
    models: [],
    languages: [],
    config: null,
    isLoading: true,
    error: null,
    isBackendConnected: false,
  });

  useEffect(() => {
    let mounted = true;

    async function fetchConfig() {
      try {
        const [models, languages, config] = await Promise.all([
          apiClient.getModels(),
          apiClient.getLanguages(),
          apiClient.getConfig(),
        ]);

        // Check if backend returned data
        const hasBackendData = models.length > 0 && languages.length > 0 && config !== null;

        if (mounted) {
          setState({
            models: hasBackendData ? models : FALLBACK_MODELS,
            languages: hasBackendData ? languages : FALLBACK_LANGUAGES,
            config: hasBackendData ? config : FALLBACK_CONFIG,
            isLoading: false,
            error: null,
            isBackendConnected: hasBackendData,
          });
        }
      } catch (err) {
        // Use fallback data on error
        if (mounted) {
          setState({
            models: FALLBACK_MODELS,
            languages: FALLBACK_LANGUAGES,
            config: FALLBACK_CONFIG,
            isLoading: false,
            error: null, // Don't show error, just use fallbacks
            isBackendConnected: false,
          });
        }
      }
    }

    fetchConfig();

    return () => {
      mounted = false;
    };
  }, []);

  const getModelsForProvider = (provider: string): ModelInfo[] => {
    return state.models.filter((m) => m.provider === provider);
  };

  return {
    ...state,
    getModelsForProvider,
  };
}

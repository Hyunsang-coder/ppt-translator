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
  { id: "claude-opus-4-5-20251101", name: "Claude Opus 4.5", provider: "anthropic" },
  { id: "claude-sonnet-4-5-20250929", name: "Claude Sonnet 4.5", provider: "anthropic" },
  { id: "claude-haiku-4-5-20251001", name: "Claude Haiku 4.5", provider: "anthropic" },
];

const FALLBACK_LANGUAGES: LanguageInfo[] = [
  { code: "한국어", name: "한국어" },
  { code: "English", name: "English" },
  { code: "日本語", name: "日本語" },
  { code: "中文(简体)", name: "中文(简体)" },
  { code: "中文(繁體)", name: "中文(繁體)" },
];

const FALLBACK_CONFIG: ConfigResponse = {
  max_upload_size_mb: 50,
  providers: ["openai", "anthropic"],
  default_provider: "openai",
  default_model: "gpt-5.2",
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

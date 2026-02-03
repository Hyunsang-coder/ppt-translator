/**
 * Hook for fetching and caching API configuration
 */

import { useEffect, useState } from "react";
import { apiClient } from "@/lib/api-client";
import type { ConfigResponse, LanguageInfo, ModelInfo } from "@/types/api";

interface ConfigState {
  models: ModelInfo[];
  languages: LanguageInfo[];
  config: ConfigResponse | null;
  isLoading: boolean;
  error: string | null;
}

export function useConfig() {
  const [state, setState] = useState<ConfigState>({
    models: [],
    languages: [],
    config: null,
    isLoading: true,
    error: null,
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

        if (mounted) {
          setState({
            models,
            languages,
            config,
            isLoading: false,
            error: null,
          });
        }
      } catch (err) {
        if (mounted) {
          setState((prev) => ({
            ...prev,
            isLoading: false,
            error: err instanceof Error ? err.message : "설정을 불러오는데 실패했습니다.",
          }));
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

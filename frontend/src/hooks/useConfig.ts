/**
 * Hook for fetching and caching API configuration
 */

import { useEffect, useState } from "react";
import { apiClient } from "@/lib/api-client";
import type { ConfigResponse, LanguageInfo, ModelInfo } from "@/types/api";

const DEFAULT_MAX_UPLOAD_SIZE_MB = 1024;

// A-2: the backend (`/api/v1/config`, `/api/v1/models`, `/api/v1/languages`) is
// the single source of truth for the model/language lists. The desktop shell
// (SidecarProvider) already gates the whole UI until the sidecar is reachable,
// so by the time any config consumer mounts the fetch below will resolve — no
// hard-coded copy of the model registry is kept here to drift out of sync.

function normalizeConfig(config: ConfigResponse): ConfigResponse {
  return {
    ...config,
    max_upload_size_mb: Math.max(
      config.max_upload_size_mb || DEFAULT_MAX_UPLOAD_SIZE_MB,
      DEFAULT_MAX_UPLOAD_SIZE_MB
    ),
  };
}

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

        const hasBackendData = models.length > 0 && languages.length > 0 && config !== null;

        if (mounted) {
          setState({
            models,
            languages,
            config: config ? normalizeConfig(config) : null,
            isLoading: false,
            error: null,
            isBackendConnected: hasBackendData,
          });
        }
      } catch (err) {
        // Backend unreachable after the sidecar reported ready: surface empty
        // lists (dropdowns render disabled) rather than a stale hard-coded copy.
        if (mounted) {
          setState({
            models: [],
            languages: [],
            config: null,
            isLoading: false,
            error: null,
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

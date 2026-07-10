"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import type { Update } from "@tauri-apps/plugin-updater";
import { isTauri } from "@/lib/api-base";

const SKIPPED_VERSION_KEY = "ppt_translator_skipped_update_version";

interface UpdateState {
  checking: boolean;
  available: boolean;
  downloading: boolean;
  progress: number;
  error: string | null;
  update: Update | null;
}

/**
 * Auto-update lifecycle for the Tauri desktop app.
 *
 * - Checks GitHub Releases (`latest.json`) ~3s after launch (production only).
 * - Surfaces the available `Update` so the UI can show a modal.
 * - Downloads + installs with progress, then relaunches.
 * - Remembers "skip this version" in localStorage.
 *
 * No-ops outside Tauri.
 */
export function useAutoUpdate() {
  const [state, setState] = useState<UpdateState>({
    checking: false,
    available: false,
    downloading: false,
    progress: 0,
    error: null,
    update: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);

  const checkForUpdate = useCallback(async (): Promise<{
    update: Update | null;
    error: string | null;
    skipped: boolean;
  }> => {
    if (!isTauri()) return { update: null, error: null, skipped: false };

    setState((prev) => ({ ...prev, checking: true, error: null }));

    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const update = await check();

      if (update) {
        // 이미 "건너뛰기" 처리된 버전이면 무시
        const skippedVersion = localStorage.getItem(SKIPPED_VERSION_KEY);
        if (skippedVersion === update.version) {
          setState((prev) => ({ ...prev, checking: false, available: false }));
          return { update: null, error: null, skipped: true };
        }

        setState((prev) => ({
          ...prev,
          checking: false,
          available: true,
          update,
        }));
        return { update, error: null, skipped: false };
      } else {
        setState((prev) => ({ ...prev, checking: false, available: false }));
        return { update: null, error: null, skipped: false };
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : "업데이트 확인 실패";
      setState((prev) => ({ ...prev, checking: false, error: msg }));
      return { update: null, error: msg, skipped: false };
    }
  }, []);

  const downloadAndInstall = useCallback(async () => {
    if (!state.update) return;

    const controller = new AbortController();
    abortControllerRef.current = controller;
    let sidecarStopped = false;

    setState((prev) => ({ ...prev, downloading: true, progress: 0, error: null }));

    try {
      let contentLength = 0;
      let downloaded = 0;

      // Keep the sidecar available while the package downloads. It is stopped
      // only for the short install window, after all bytes are on disk.
      await state.update.download((event) => {
        if (controller.signal.aborted) {
          throw new Error("Download cancelled");
        }

        switch (event.event) {
          case "Started":
            contentLength = event.data.contentLength ?? 0;
            break;
          case "Progress": {
            downloaded += event.data.chunkLength;
            const progress =
              contentLength > 0
                ? Math.round((downloaded / contentLength) * 100)
                : 0;
            setState((prev) => ({ ...prev, progress }));
            break;
          }
          case "Finished":
            setState((prev) => ({ ...prev, progress: 100 }));
            break;
        }
      });

      if (controller.signal.aborted) {
        throw new Error("Download cancelled");
      }

      // Windows cannot overwrite native modules loaded by the Python sidecar.
      // The Rust command kills it and waits for the process handle to signal,
      // guaranteeing that files such as PIL/_imaging*.pyd are unlocked.
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke("prepare_for_update");
      sidecarStopped = true;

      await state.update.install();

      const { relaunch } = await import("@tauri-apps/plugin-process");
      await relaunch();
    } catch (error) {
      // If download or installation fails after the sidecar was stopped, make
      // the current app usable again without requiring a manual restart.
      if (sidecarStopped) {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          await invoke("restart_sidecar");
        } catch (restartError) {
          console.error("[updater] failed to restart sidecar", restartError);
        }
      }

      if (controller.signal.aborted) {
        setState((prev) => ({
          ...prev,
          downloading: false,
          progress: 0,
          error: null,
        }));
      } else {
        setState((prev) => ({
          ...prev,
          downloading: false,
          error: error instanceof Error ? error.message : "업데이트 실패",
        }));
      }
    } finally {
      abortControllerRef.current = null;
    }
  }, [state.update]);

  const cancelDownload = useCallback(() => {
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
    setState((prev) => ({ ...prev, downloading: false, progress: 0 }));
  }, []);

  const skipVersion = useCallback((version: string) => {
    localStorage.setItem(SKIPPED_VERSION_KEY, version);
    setState((prev) => ({ ...prev, available: false, update: null }));
  }, []);

  const dismissUpdate = useCallback(() => {
    setState((prev) => ({ ...prev, available: false }));
  }, []);

  const setManualUpdate = useCallback((manualUpdate: Update) => {
    setState((prev) => ({ ...prev, available: true, update: manualUpdate }));
  }, []);

  // 앱 시작 시 자동 체크 (프로덕션 데스크톱 앱만)
  useEffect(() => {
    if (process.env.NODE_ENV !== "production") return;
    if (!isTauri()) return;

    const timer = setTimeout(() => {
      void checkForUpdate();
    }, 3000);

    return () => clearTimeout(timer);
  }, [checkForUpdate]);

  return {
    ...state,
    checkForUpdate,
    downloadAndInstall,
    cancelDownload,
    skipVersion,
    dismissUpdate,
    setManualUpdate,
  };
}

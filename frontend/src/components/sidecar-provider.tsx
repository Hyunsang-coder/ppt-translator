"use client";

/**
 * Bootstraps the Tauri sidecar connection.
 *
 * On the desktop app the Python API server binds a random port; the Rust shell
 * emits a `sidecar-ready` event with that port and also answers a
 * `get_sidecar_port` command (for the case where the event fired before this
 * component mounted). We cache the resulting base URL via setSidecarPort() and
 * gate the UI until the API is reachable.
 *
 * Outside Tauri, local browser development can render the app screens with the
 * NEXT_PUBLIC_API_URL fallback.
 */

import { useEffect, useState } from "react";
import { ensureApiBase, isTauri } from "@/lib/api-base";

export function SidecarProvider({ children }: { children: React.ReactNode }) {
  const runningInTauri = isTauri();
  const [engineState, setEngineState] = useState<"ready" | "starting" | "failed">("ready");
  const [engineError, setEngineError] = useState<string | null>(null);

  // `mounted` stays false during SSR and the first client render, so the first
  // client render matches the server HTML. Tauri-only loading can start after
  // hydration without causing a mismatch.
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted || !runningInTauri) return;

    let cancelled = false;
    setEngineState("starting");
    setEngineError(null);

    (async () => {
      try {
        await ensureApiBase();
        if (!cancelled) {
          setEngineState("ready");
        }
      } catch (error) {
        if (!cancelled) {
          console.error("[sidecar] failed to initialize", error);
          setEngineError(error instanceof Error ? error.message : "번역 엔진을 시작하지 못했습니다.");
          setEngineState("failed");
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [mounted, runningInTauri]);

  if (mounted && runningInTauri && engineState !== "ready") {
    return (
      <main className="min-h-screen animated-gradient-bg flex items-center justify-center px-6">
        <div className="w-full max-w-md rounded-lg border border-border bg-card p-6 text-center shadow-sm">
          <h1 className="text-lg font-semibold">
            {engineState === "starting" ? "번역 엔진 준비 중" : "번역 엔진 시작 실패"}
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            {engineState === "starting"
              ? "파일 처리 기능을 준비하고 있습니다. 잠시만 기다려주세요."
              : engineError || "앱을 다시 실행해 주세요."}
          </p>
        </div>
      </main>
    );
  }

  return <>{children}</>;
}

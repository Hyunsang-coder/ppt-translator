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
 * NEXT_PUBLIC_API_URL fallback. Hosted web builds redirect app routes back to
 * the desktop download 안내 page.
 */

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { isTauri, setSidecarPort, getApiBase } from "@/lib/api-base";

const LOCAL_WEB_HOSTS = new Set(["localhost", "127.0.0.1", "0.0.0.0", "::1"]);

function isHostedWebBuild(): boolean {
  if (typeof window === "undefined" || isTauri()) return false;
  return !LOCAL_WEB_HOSTS.has(window.location.hostname);
}

/** Poll the sidecar /health until it responds 200 (heavy import can take ~20s). */
async function waitForHealth(signal: () => boolean): Promise<boolean> {
  for (let i = 0; i < 120; i++) {
    if (signal()) return false;
    try {
      const res = await fetch(`${getApiBase()}/health`);
      if (res.ok) return true;
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 500));
  }
  return false;
}

export function SidecarProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();

  // `mounted` stays false during SSR and the first client render, so the very
  // first render always matches the server (children) — no hydration mismatch.
  // Tauri-only gating only kicks in after mount.
  const [mounted, setMounted] = useState(false);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // In the desktop app the web "/" is a "service ended" notice; send the user
  // straight to the translate screen instead.
  useEffect(() => {
    if (!mounted) return;

    if (isTauri() && pathname === "/") {
      router.replace("/translate");
      return;
    }

    if (isHostedWebBuild() && pathname !== "/") {
      router.replace("/");
    }
  }, [mounted, pathname, router]);

  useEffect(() => {
    if (!mounted || !isTauri()) return;

    let unlisten: (() => void) | undefined;
    let cancelled = false;

    const onPort = async (port: number) => {
      if (cancelled) return;
      setSidecarPort(port);
      // The port is known, but the FastAPI app may still be importing
      // (~20s cold start). Wait until /health answers before showing the UI.
      const ok = await waitForHealth(() => cancelled);
      if (!cancelled && ok) setReady(true);
    };

    (async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      const { listen } = await import("@tauri-apps/api/event");

      // Subscribe first to avoid missing the event.
      unlisten = await listen<number>("sidecar-ready", (e) => {
        void onPort(e.payload);
      });

      // In case the sidecar was already up before we subscribed.
      const port = await invoke<number | null>("get_sidecar_port");
      if (!cancelled && port) {
        void onPort(port);
      }
    })();

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [mounted]);

  // Show the loading screen only in Tauri, only after mount, and only until the
  // sidecar reports its port. Web build and SSR render children directly.
  if (mounted && isTauri() && !ready) {
    return (
      <div
        style={{
          display: "flex",
          height: "100vh",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: "1rem",
          color: "var(--muted-foreground, #888)",
        }}
      >
        <div>번역 엔진을 시작하는 중…</div>
        <div style={{ fontSize: "0.8rem", opacity: 0.7 }}>
          처음 실행 시 20초 정도 걸릴 수 있습니다.
        </div>
      </div>
    );
  }

  return <>{children}</>;
}

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
 * Outside Tauri (plain web build) this renders children immediately and the API
 * base falls back to NEXT_PUBLIC_API_URL.
 */

import { useEffect, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { isTauri, setSidecarPort } from "@/lib/api-base";

type Status = "connecting" | "ready" | "web";

export function SidecarProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [status, setStatus] = useState<Status>(() =>
    isTauri() ? "connecting" : "web"
  );

  // In the desktop app the web "/" is a "service ended" notice; send the user
  // straight to the translate screen instead.
  useEffect(() => {
    if (isTauri() && pathname === "/") {
      router.replace("/translate");
    }
  }, [pathname, router]);

  useEffect(() => {
    if (!isTauri()) return;

    let unlisten: (() => void) | undefined;
    let cancelled = false;

    (async () => {
      const { invoke } = await import("@tauri-apps/api/core");
      const { listen } = await import("@tauri-apps/api/event");

      // Subscribe first to avoid missing the event.
      unlisten = await listen<number>("sidecar-ready", (e) => {
        if (cancelled) return;
        setSidecarPort(e.payload);
        setStatus("ready");
      });

      // In case the sidecar was already up before we subscribed.
      const port = await invoke<number | null>("get_sidecar_port");
      if (!cancelled && port) {
        setSidecarPort(port);
        setStatus("ready");
      }
    })();

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  if (status === "connecting") {
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
      </div>
    );
  }

  return <>{children}</>;
}

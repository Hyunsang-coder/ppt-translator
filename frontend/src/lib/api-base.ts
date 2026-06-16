/**
 * Resolves the API base URL at runtime.
 *
 * - In the Tauri desktop app the Python sidecar binds a random free port that
 *   is only known at runtime. The Rust shell emits a `sidecar-ready` event and
 *   exposes a `get_sidecar_port` command; we cache the resulting base URL on
 *   `window.__API_BASE__`.
 * - In a plain browser build we fall back to the build-time
 *   `NEXT_PUBLIC_API_URL` for local development. The Vercel deployment is only
 *   a desktop download page and does not call the API.
 */

const BUILD_TIME_BASE = process.env.NEXT_PUBLIC_API_URL ?? "";
const SIDECAR_TIMEOUT_MS = 120_000;

declare global {
  interface Window {
    __API_BASE__?: string;
    isTauri?: boolean;
    __TAURI_INTERNALS__?: unknown;
  }
}

let sidecarBasePromise: Promise<string> | null = null;

export function isTauri(): boolean {
  if (typeof window === "undefined") return false;
  return window.isTauri === true || window.__TAURI_INTERNALS__ !== undefined;
}

/** The current API base URL (no trailing slash). */
export function getApiBase(): string {
  if (typeof window !== "undefined" && window.__API_BASE__) {
    return window.__API_BASE__;
  }
  return BUILD_TIME_BASE;
}

export function isWaitingForSidecarBase(): boolean {
  return isTauri() && getApiBase() === "";
}

/** Cache the sidecar base URL once the port is known. */
export function setSidecarPort(port: number): void {
  if (typeof window !== "undefined") {
    window.__API_BASE__ = `http://127.0.0.1:${port}`;
  }
}

async function waitForHealth(baseUrl: string): Promise<void> {
  const deadline = Date.now() + SIDECAR_TIMEOUT_MS;

  while (Date.now() < deadline) {
    try {
      const res = await fetch(`${baseUrl}/health`);
      if (res.ok) return;
    } catch {
      // Sidecar is still importing or binding.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  throw new Error("번역 엔진 시작 시간이 초과되었습니다.");
}

async function resolveSidecarBase(): Promise<string> {
  const existingBase = getApiBase();
  if (existingBase) return existingBase;

  if (!isTauri()) return BUILD_TIME_BASE;

  const { invoke } = await import("@tauri-apps/api/core");
  const { listen } = await import("@tauri-apps/api/event");

  const unlisteners: Array<() => void> = [];
  let timeout: number | undefined;
  let poll: number | undefined;
  let cancelled = false;
  const cleanup = () => {
    cancelled = true;
    if (timeout !== undefined) {
      window.clearTimeout(timeout);
      timeout = undefined;
    }
    if (poll !== undefined) {
      window.clearInterval(poll);
      poll = undefined;
    }
    for (const unlisten of unlisteners.splice(0)) {
      unlisten();
    }
  };

  const portPromise = new Promise<number>(async (resolve, reject) => {
    const resolvePort = (port: number) => {
      if (cancelled) return;
      cleanup();
      resolve(port);
    };
    const rejectPort = (error: unknown) => {
      if (cancelled) return;
      cleanup();
      reject(error);
    };

    timeout = window.setTimeout(() => {
      rejectPort(new Error("번역 엔진 포트를 받지 못했습니다."));
    }, SIDECAR_TIMEOUT_MS);

    try {
      // Register listeners before the first command call so the ready event
      // cannot be missed while the command bridge is round-tripping.
      unlisteners.push(
        await listen<number>("sidecar-ready", (event) => {
          resolvePort(event.payload);
        })
      );
      unlisteners.push(
        await listen<string>("sidecar-error", (event) => {
          rejectPort(new Error(event.payload || "번역 엔진 시작에 실패했습니다."));
        })
      );
      unlisteners.push(
        await listen("sidecar-terminated", () => {
          rejectPort(new Error("번역 엔진이 시작 중 종료되었습니다."));
        })
      );

      const checkPort = async () => {
        try {
          const currentPort = await invoke<number | null>("get_sidecar_port");
          if (currentPort) {
            resolvePort(currentPort);
          }
        } catch (error) {
          rejectPort(error);
        }
      };

      await checkPort();
      poll = window.setInterval(() => {
        void checkPort();
      }, 500);
    } catch (error) {
      rejectPort(error);
    }
  });

  const port = await portPromise;
  setSidecarPort(port);
  const baseUrl = getApiBase();
  await waitForHealth(baseUrl);
  return baseUrl;
}

export async function ensureApiBase(): Promise<string> {
  const existingBase = getApiBase();
  if (existingBase || !isTauri()) return existingBase;

  sidecarBasePromise ??= resolveSidecarBase().catch((error) => {
    sidecarBasePromise = null;
    throw error;
  });
  return sidecarBasePromise;
}

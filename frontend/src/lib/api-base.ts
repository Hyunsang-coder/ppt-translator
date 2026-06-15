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

declare global {
  interface Window {
    __API_BASE__?: string;
    __TAURI_INTERNALS__?: unknown;
  }
}

export function isTauri(): boolean {
  return typeof window !== "undefined" && window.__TAURI_INTERNALS__ !== undefined;
}

/** The current API base URL (no trailing slash). */
export function getApiBase(): string {
  if (typeof window !== "undefined" && window.__API_BASE__) {
    return window.__API_BASE__;
  }
  return BUILD_TIME_BASE;
}

/** Cache the sidecar base URL once the port is known. */
export function setSidecarPort(port: number): void {
  if (typeof window !== "undefined") {
    window.__API_BASE__ = `http://127.0.0.1:${port}`;
  }
}

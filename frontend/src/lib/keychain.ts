/**
 * Thin wrappers over the Rust keychain commands. No-ops outside Tauri.
 */

import { isTauri } from "@/lib/api-base";

export type Provider = "openai" | "anthropic";

export async function saveApiKey(provider: Provider, key: string): Promise<void> {
  if (!isTauri()) throw new Error("keychain is only available in the desktop app");
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("save_api_key", { provider, key });
}

export async function deleteApiKey(provider: Provider): Promise<void> {
  if (!isTauri()) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("delete_api_key", { provider });
}

export async function hasApiKey(provider: Provider): Promise<boolean> {
  if (!isTauri()) return false;
  const { invoke } = await import("@tauri-apps/api/core");
  return invoke<boolean>("has_api_key", { provider });
}

/** Restart the sidecar so newly-saved keys take effect (no app restart). */
export async function restartSidecar(): Promise<void> {
  if (!isTauri()) return;
  const { invoke } = await import("@tauri-apps/api/core");
  await invoke("restart_sidecar");
}

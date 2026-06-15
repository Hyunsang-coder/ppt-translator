/**
 * Saves a Blob to disk.
 *
 * - In Tauri the browser's `<a download>` trick does nothing (no download
 *   handler), so we use the native save dialog + filesystem plugin.
 * - On the web we fall back to the standard object-URL + anchor click.
 */

import { isTauri } from "@/lib/api-base";

export async function saveBlob(blob: Blob, filename: string): Promise<void> {
  if (isTauri()) {
    const { save } = await import("@tauri-apps/plugin-dialog");
    const { writeFile } = await import("@tauri-apps/plugin-fs");

    // Build a save-dialog filter from the filename's extension.
    const ext = filename.split(".").pop() || "";
    const filters = ext
      ? [{ name: ext.toUpperCase(), extensions: [ext] }]
      : undefined;

    const path = await save({ defaultPath: filename, filters });
    if (!path) return; // user cancelled

    const bytes = new Uint8Array(await blob.arrayBuffer());
    await writeFile(path, bytes);
    return;
  }

  // Web fallback
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

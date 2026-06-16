#!/usr/bin/env node
// Ensure the Python sidecar is staged for Tauri.
//
// Cross-platform (Windows / macOS / Linux) replacement for the old
// ensure-sidecar.sh. By default this only rebuilds when the staged executable
// is missing or when backend/sidecar inputs are newer than it. Pass --force to
// rebuild unconditionally (used by production builds).
import { spawnSync } from "node:child_process";
import { existsSync, statSync, readdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const isWindows = process.platform === "win32";
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");
const force = process.argv.includes("--force");

const EXE_NAME = isWindows
  ? "ppt-translator-sidecar.exe"
  : "ppt-translator-sidecar";
const STAGED_EXE = join(
  REPO_ROOT,
  "src-tauri",
  "resources",
  "sidecar",
  EXE_NAME,
);

// Newest modification time (ms) under a file or directory tree, or 0 if absent.
function newestMtime(path) {
  const stat = statSync(path, { throwIfNoEntry: false });
  if (!stat) return 0;
  if (!stat.isDirectory()) return stat.mtimeMs;
  let newest = 0;
  for (const entry of readdirSync(path)) {
    newest = Math.max(newest, newestMtime(join(path, entry)));
  }
  return newest;
}

function needsBuild() {
  if (force) return true;
  if (!existsSync(STAGED_EXE)) return true;

  const stagedMtime = statSync(STAGED_EXE).mtimeMs;
  const inputs = [
    "api.py",
    "src",
    join("desktop", "sidecar.py"),
    join("desktop", "sidecar.spec"),
    join("desktop", "requirements-desktop.txt"),
  ];
  return inputs.some(
    (input) => newestMtime(join(REPO_ROOT, input)) > stagedMtime,
  );
}

if (needsBuild()) {
  const res = spawnSync(
    process.execPath,
    [join(SCRIPT_DIR, "build-sidecar.mjs")],
    { stdio: "inherit", cwd: REPO_ROOT },
  );
  if (res.status !== 0) process.exit(res.status ?? 1);
} else {
  console.log("==> Sidecar is already staged and up to date");
}

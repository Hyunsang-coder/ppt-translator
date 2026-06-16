#!/usr/bin/env node
// Build the Python sidecar (onedir) and stage it into the Tauri resources dir.
//
// Cross-platform (Windows / macOS / Linux) replacement for the old
// build-sidecar.sh. Tauri runs beforeBuildCommand through the OS default shell
// (cmd.exe on Windows, sh on Unix), so a bash-only `.sh` hook fails on Windows.
// Node is already required for the frontend build and behaves identically on
// every platform, so the sidecar build logic lives here instead.
//
// Run from anywhere:  node desktop/build-sidecar.mjs
// The desktop venv (desktop/.venv-desktop) is created automatically on first run.
import { spawnSync } from "node:child_process";
import { existsSync, rmSync, mkdirSync, cpSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const isWindows = process.platform === "win32";
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(SCRIPT_DIR, "..");

function run(cmd, args) {
  const res = spawnSync(cmd, args, { stdio: "inherit", cwd: REPO_ROOT });
  if (res.error) {
    console.error(`error: failed to run ${cmd}: ${res.error.message}`);
    process.exit(1);
  }
  if (res.status !== 0) {
    process.exit(res.status ?? 1);
  }
}

const VENV = process.env.DESKTOP_VENV || join("desktop", ".venv-desktop");
const venvBin = (name) =>
  isWindows
    ? join(REPO_ROOT, VENV, "Scripts", `${name}.exe`)
    : join(REPO_ROOT, VENV, "bin", name);

const pyinstaller = venvBin("pyinstaller");

if (!existsSync(pyinstaller)) {
  // Provision the desktop venv on first build so `cargo tauri build` works out
  // of the box. The pip install only runs once; later builds reuse the venv.
  console.log("==> Desktop venv not found; creating it (one-time setup)");
  const basePython = isWindows ? "python" : "python3";
  run(basePython, ["-m", "venv", VENV]);
  run(venvBin("python"), ["-m", "pip", "install", "--upgrade", "pip"]);
  run(venvBin("python"), [
    "-m",
    "pip",
    "install",
    "-r",
    join("desktop", "requirements-desktop.txt"),
  ]);
}

if (!existsSync(pyinstaller)) {
  console.error(
    `error: ${pyinstaller} not found even after provisioning the venv.`,
  );
  process.exit(1);
}

console.log("==> Building sidecar (onedir) with PyInstaller");
rmSync(join(REPO_ROOT, "desktop", "dist"), { recursive: true, force: true });
rmSync(join(REPO_ROOT, "desktop", "build"), { recursive: true, force: true });
run(pyinstaller, [
  join("desktop", "sidecar.spec"),
  "--noconfirm",
  "--distpath",
  join("desktop", "dist"),
  "--workpath",
  join("desktop", "build"),
]);

const SRC = join(REPO_ROOT, "desktop", "dist", "ppt-translator-sidecar");
const DEST = join(REPO_ROOT, "src-tauri", "resources", "sidecar");
if (!existsSync(SRC)) {
  console.error(`error: expected build output at ${SRC}`);
  process.exit(1);
}

console.log(`==> Staging into ${DEST}`);
rmSync(DEST, { recursive: true, force: true });
mkdirSync(dirname(DEST), { recursive: true });
cpSync(SRC, DEST, { recursive: true });

console.log(`==> Done. Sidecar staged at ${DEST}`);

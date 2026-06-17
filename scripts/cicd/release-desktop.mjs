#!/usr/bin/env node

/**
 * Bump desktop version files, commit, tag, and push to trigger Desktop Release.
 *
 * Usage:
 *   node scripts/cicd/release-desktop.mjs            # patch bump
 *   node scripts/cicd/release-desktop.mjs 0.1.6      # explicit version
 */

import { readFileSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const tauriConfigPath = path.join(root, "src-tauri/tauri.conf.json");
const cargoTomlPath = path.join(root, "src-tauri/Cargo.toml");
const packageJsonPath = path.join(root, "frontend/package.json");

function readJson(filePath) {
  return JSON.parse(readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

function bumpPatch(version) {
  const parts = version.split(".").map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) {
    throw new Error(`Unsupported version format: ${version}`);
  }
  parts[2] += 1;
  return parts.join(".");
}

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: "utf8", stdio: "inherit" });
  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

const currentVersion = readJson(tauriConfigPath).version;
const nextVersion = process.argv[2] ?? bumpPatch(currentVersion);
const tag = `v${nextVersion}`;

console.log(`Preparing desktop release ${tag}...`);

const tauriConfig = readJson(tauriConfigPath);
tauriConfig.version = nextVersion;
writeJson(tauriConfigPath, tauriConfig);

const cargoToml = readFileSync(cargoTomlPath, "utf8").replace(
  /^version = ".+"$/m,
  `version = "${nextVersion}"`,
);
writeFileSync(cargoTomlPath, cargoToml);

const packageJson = readJson(packageJsonPath);
packageJson.version = nextVersion;
writeJson(packageJsonPath, packageJson);

run("git", ["add", tauriConfigPath, cargoTomlPath, packageJsonPath]);
run("git", ["commit", "-m", `chore: release desktop ${nextVersion}`]);
run("git", ["tag", tag]);
run("git", ["push", "origin", "HEAD", tag]);

console.log(`Release ${tag} pushed. Desktop Release workflow will build installers and deploy the web page.`);

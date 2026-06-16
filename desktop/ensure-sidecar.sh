#!/usr/bin/env bash
# Ensure the Python sidecar is staged for Tauri.
#
# By default this only rebuilds when the staged executable is missing or when
# backend/sidecar inputs are newer than the staged executable. Pass --force to
# rebuild unconditionally.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

EXE_NAME="ppt-translator-sidecar"
if [[ "${OS:-}" == "Windows_NT" ]]; then
  EXE_NAME="ppt-translator-sidecar.exe"
fi

STAGED_EXE="src-tauri/resources/sidecar/$EXE_NAME"

needs_build() {
  if [[ "$FORCE" == "1" ]]; then
    return 0
  fi

  if [[ ! -x "$STAGED_EXE" ]]; then
    return 0
  fi

  if find \
    api.py \
    src \
    desktop/sidecar.py \
    desktop/sidecar.spec \
    desktop/requirements-desktop.txt \
    -newer "$STAGED_EXE" \
    -print -quit | grep -q .; then
    return 0
  fi

  return 1
}

if needs_build; then
  ./desktop/build-sidecar.sh
else
  echo "==> Sidecar is already staged and up to date"
fi

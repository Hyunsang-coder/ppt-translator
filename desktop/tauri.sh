#!/usr/bin/env bash
# Run the Tauri CLI with the Defender-DLP workaround env applied.
#
# On this managed Mac, Microsoft Defender DLP blocks the linker from opening
# object files unless both the cargo target dir AND TMPDIR live under the
# Defender-excluded folder ~/_NOAV/XcodeDerivedData. (proc-macro crates link a
# .dylib whose rmeta.o is written to $TMPDIR, so TMPDIR must be redirected too.)
#
# Usage (from anywhere):
#   ./desktop/tauri.sh dev          # hot-reload dev run
#   TAURI_BUILD=1 ./desktop/tauri.sh build   # production bundle
#
# Any args are passed straight through to `cargo tauri`.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXCLUDED="$HOME/_NOAV/XcodeDerivedData"
mkdir -p "$EXCLUDED/cargo-target" "$EXCLUDED/tmp"

cd "$REPO_ROOT/src-tauri"
exec env \
  TMPDIR="$EXCLUDED/tmp" \
  CARGO_TARGET_DIR="$EXCLUDED/cargo-target" \
  PATH="$HOME/.cargo/bin:$PATH" \
  cargo tauri "$@"

#!/usr/bin/env bash
# Build the Python sidecar (onedir) and stage it into the Tauri resources dir.
#
# Run from the repo root:  ./desktop/build-sidecar.sh
# Requires the desktop venv at desktop/.venv-desktop (see desktop/README.md).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV="desktop/.venv-desktop"
if [ ! -x "$VENV/bin/pyinstaller" ]; then
  echo "error: $VENV/bin/pyinstaller not found. Create the venv first:" >&2
  echo "  python3 -m venv $VENV && $VENV/bin/pip install -r desktop/requirements-desktop.txt" >&2
  exit 1
fi

echo "==> Building sidecar (onedir) with PyInstaller"
rm -rf desktop/dist desktop/build
"$VENV/bin/pyinstaller" desktop/sidecar.spec --noconfirm \
  --distpath desktop/dist --workpath desktop/build

SRC="desktop/dist/ppt-translator-sidecar"
DEST="src-tauri/resources/sidecar"
if [ ! -d "$SRC" ]; then
  echo "error: expected build output at $SRC" >&2
  exit 1
fi

echo "==> Staging into $DEST"
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -R "$SRC" "$DEST"

echo "==> Done. Sidecar staged at $DEST"
du -sh "$DEST"

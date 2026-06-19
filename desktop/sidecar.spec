# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the local sidecar binary.

Build (from repo root, inside the desktop venv):
    pyinstaller desktop/sidecar.spec --noconfirm --distpath desktop/dist --workpath desktop/build

Produces a single-folder bundle at desktop/dist/ppt-translator-sidecar/
(executable + _internal/). We use onedir (NOT onefile) because:
  - No per-launch self-extraction to a temp dir, so no cold start.
  - onefile re-execs itself out of a temp _MEI dir, which DLP/endpoint-security
    software (e.g. Microsoft Defender on managed Macs) can block. onedir runs
    in place and avoids that failure mode entirely.
The whole folder is shipped via Tauri `bundle.resources` and spawned directly
by the Rust shell (see src-tauri/src/lib.rs).
"""

import os
import sys

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# PyInstaller exposes SPECPATH as the directory containing this spec file
# (i.e. .../desktop). Repo root is its parent. Add the root to the search path
# so `import api` and `import src.*` resolve during analysis.
SPEC_DIR = os.path.abspath(SPECPATH)
REPO_ROOT = os.path.dirname(SPEC_DIR)

# collect_submodules imports by walking sys.path; make the repo root importable
# during analysis so `src` and `api` are discoverable.
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Our own top-level module (api.py at the repo root) plus the src package.
hidden = ["api"]
hidden += collect_submodules("src", on_error="ignore")

# langchain and pydantic resolve a lot of providers dynamically, so PyInstaller's
# static analysis misses them. Pull their submodules in explicitly.
for pkg in (
    "langchain",
    "langchain_core",
    "langchain_openai",
    "langchain_anthropic",
    "langchain_community",
    "openai",
    "anthropic",
    "pptx",
    "tenacity",
    "langdetect",
    "sse_starlette",
    "uvicorn",
):
    try:
        hidden += collect_submodules(pkg)
    except Exception:
        # Optional package (e.g. langchain_community) may be absent; skip.
        pass

datas = []
for pkg in ("langchain", "langchain_core", "pptx", "langdetect"):
    try:
        datas += collect_data_files(pkg)
    except Exception:
        pass

# Large native packages that are pulled in transitively but never imported by
# our source. Excluding them keeps the bundle small and the build fast.
excludes = [
    "torch",
    "transformers",
    "tensorflow",
    "scipy",
    "sklearn",
    "matplotlib",
    "cv2",
    "fitz",          # PyMuPDF
    "pytesseract",
    "boto3",         # optional Bedrock path from Anthropic, unused here
    "botocore",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
]

a = Analysis(
    [os.path.join(SPEC_DIR, "sidecar.py")],
    pathex=[REPO_ROOT],     # so `from api import app` and `import src.*` resolve
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

# macOS code signing is delegated to PyInstaller: when codesign_identity is set,
# PyInstaller signs the executable AND every collected binary/framework in the
# COLLECT step, in the correct order and with the framework file placement Apple's
# codesign requires (per PyInstaller docs). This is what makes notarization pass —
# hand-signing the staged tree afterwards can't seal Python.framework's root
# binary slot. Unset (local / Windows / no secrets) → ad-hoc signing, unchanged.
CODESIGN_IDENTITY = os.environ.get("APPLE_SIGNING_IDENTITY") or None
ENTITLEMENTS_FILE = (
    os.path.join(REPO_ROOT, "src-tauri", "entitlements.plist")
    if CODESIGN_IDENTITY
    else None
)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ppt-translator-sidecar",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Keep console=True: a windowed (console=False) build can leave sys.stdout
    # as None on Windows, which would crash the SIDECAR_READY print() and stall
    # the handshake. The terminal *window* is suppressed instead by the parent
    # spawning us with CREATE_NO_WINDOW (see spawn_sidecar in src-tauri/src/lib.rs),
    # which keeps the stdout/stderr pipes intact while showing no console.
    console=True,
    codesign_identity=CODESIGN_IDENTITY,
    entitlements_file=ENTITLEMENTS_FILE,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="ppt-translator-sidecar",
)

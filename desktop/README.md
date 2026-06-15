# Desktop build (Tauri)

Local desktop packaging of the PPT translator. The Python translation core runs
as a bundled sidecar; the user supplies their own API keys, stored in the OS
keychain. No AWS / SSM involved (that path stays in the legacy server build).

## Layout

- `requirements-desktop.txt` — slimmed Python deps (no boto3/OCR/CV/mangum).
- `sidecar.py` — sidecar entrypoint; binds a free port, prints
  `SIDECAR_READY port=N`, then serves the FastAPI app from `api.py`.
- `sidecar.spec` — PyInstaller spec (onedir; excludes unused native deps).
- `build-sidecar.sh` — builds the sidecar and stages it into
  `../src-tauri/resources/sidecar/`.

The Tauri shell lives in `../src-tauri/` (Rust). It reads keys from the keychain,
spawns the sidecar, and points the WebView at the sidecar port.

## One-time setup

```bash
# from repo root
python3 -m venv desktop/.venv-desktop
desktop/.venv-desktop/bin/pip install -r desktop/requirements-desktop.txt

# Tauri CLI. On this managed Mac, Defender DLP blocks linking from the default
# temp dir, so build into a Defender-excluded folder:
CARGO_TARGET_DIR=~/_NOAV/cargo-target cargo install tauri-cli --version "^2.0.0"
```

## Build

```bash
# 1. Build + stage the Python sidecar
./desktop/build-sidecar.sh

# 2. Dev run (hot-reload frontend, spawns sidecar)
cd src-tauri && CARGO_TARGET_DIR=~/_NOAV/cargo-target cargo tauri dev

# 3. Production bundle (.app/.dmg on macOS, .msi/.exe on Windows)
TAURI_BUILD=1 CARGO_TARGET_DIR=~/_NOAV/cargo-target cargo tauri build
```

`TAURI_BUILD=1` switches the Next.js build to static export (`output: 'export'`)
into `frontend/out`, which Tauri bundles as the frontend.

## Cross-platform note

The sidecar is platform-specific (PyInstaller bundles a native CPython). Build
the sidecar **on each target OS**: run `build-sidecar.sh` on macOS for the mac
build, and an equivalent step on Windows for the Windows build. You cannot
cross-compile the sidecar from one OS to another.

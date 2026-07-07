# Start Desktop App (Tauri)

Build and launch the Tauri desktop app in hot-reload dev mode. This is the
single command for running the app — it stages the Python sidecar, starts the
Next.js frontend, and opens the native window in one step.

## Instructions

1. Confirm the app isn't already running: `ps aux | grep -iE "cargo tauri|ppt-translator" | grep -v grep`
   - If a dev session is already up, tell the user instead of starting a second one.
2. Launch the app (run in background so hot-reload keeps working):
   ```bash
   ./desktop/tauri.sh dev
   ```
3. The first run compiles Rust and may take several minutes; later runs are fast.
   Watch the output for the window opening and for `sidecar-ready` / a bound port.
4. Report the result to the user (window opened, or the error if the build failed).

## Notes

- **One command does everything.** `tauri.sh dev` runs `beforeDevCommand`, which
  calls `desktop/ensure-sidecar.mjs` (rebuilds the sidecar only when `api.py`,
  `src/`, or the sidecar spec changed) and then `npm run dev` for the frontend.
  You do NOT need to run `build-sidecar.mjs` or start a backend separately.
- **DLP workaround is baked in.** `tauri.sh` redirects `CARGO_TARGET_DIR` and
  `TMPDIR` under `~/_NOAV/XcodeDerivedData` so Microsoft Defender doesn't block
  the linker. Never run `cargo tauri` directly — always go through `tauri.sh`.
- The sidecar binds a random free port at runtime; the frontend discovers it via
  the Rust `get_sidecar_port` command, so there is no fixed `:8000` backend here.
- API keys come from the macOS Keychain via the app's settings UI, not `.env`.
- For a production bundle instead of dev: `TAURI_BUILD=1 ./desktop/tauri.sh build`.

# Deploy (Local Build → Install → Web)

Three phases: (1) local Tauri production build, (2) replace the local
install and smoke-test it, (3) commit/push + web deploy so the public
download page updates.

## Phase 1 — Local production build

1. Preconditions: `gh auth status` must succeed. Confirm no dev session is
   running:
   `ps aux | grep -iE "cargo tauri|ppt-translator" | grep -v grep`
   - If one is up, stop it first (or tell the user) instead of building over it.
2. Run the production bundle (foreground; release compile + PyInstaller
   sidecar rebuild can take a long time on first run):
   ```bash
   TAURI_BUILD=1 ./desktop/tauri.sh build
   ```
3. Verify fresh bundles and report their paths:
   ```bash
   ls -la ~/_NOAV/XcodeDerivedData/cargo-target/release/bundle/macos/ \
          ~/_NOAV/XcodeDerivedData/cargo-target/release/bundle/dmg/
   ```
   Expect `ppt-translation-cat.app`, `ppt-translation-cat.app.tar.gz`
   (updater), and the versioned `.dmg` — all with fresh timestamps.
   - A trailing `Error: ... no private key` (`TAURI_SIGNING_PRIVATE_KEY`) is
     EXPECTED on local builds, not a failure: signing happens in
     `desktop-release.yml`. As long as the bundles above exist, proceed.

## Phase 2 — Replace local install + smoke test

1. Quit the installed app if running, replace it, smoke-test, quit:
   ```bash
   osascript -e 'quit app "ppt-translation-cat"' 2>/dev/null; sleep 2
   rm -rf /Applications/ppt-translation-cat.app && ditto \
     "$HOME/_NOAV/XcodeDerivedData/cargo-target/release/bundle/macos/ppt-translation-cat.app" \
     /Applications/ppt-translation-cat.app
   open /Applications/ppt-translation-cat.app && sleep 8 && \
     ps aux | grep -E "ppt-translat" | grep -v grep
   osascript -e 'quit app "ppt-translation-cat"'
   ```
   The smoke test must show both `ppt-translator-desktop` and its
   `ppt-translator-sidecar` child before quitting. If either is missing,
   stop — do not proceed to Phase 3.

## Phase 3 — Commit, push, web deploy

1. `git status` to see what will be committed. If the tree is clean and in
   sync with `origin`, skip to step 3 (nothing to commit/push).
2. Commit and push the current branch:
   - Commit message follows repo convention (`fix:` / `feat:` / `chore:`
     prefix, describing the included changes).
   - Push with `git push` (or `git push -u origin <branch>` if no upstream).
3. Update the web download page (`https://ppt-translator.vercel.app`):
   - If step 2 pushed to `main`: `Full Validation` + `Deploy Web` run
     automatically. Do NOT fire a manual duplicate — watch them:
     ```bash
     gh run list --workflow=predeploy.yml --branch main --limit 3
     gh run list --workflow=deploy-web.yml --limit 3
     ```
   - If there was nothing to push but a web deploy was explicitly
     requested, trigger it manually (no auto-run to duplicate):
     ```bash
     gh workflow run deploy-web.yml --ref main
     gh run watch <run-id>
     ```
   - On any other branch: trigger explicitly for the pushed ref, then watch:
     ```bash
     gh workflow run deploy-web.yml --ref <branch>
     gh run watch <run-id>
     ```
4. Report: fresh bundle paths, install + smoke-test result, pushed commits,
   and the Deploy Web run outcome (+ deployment URL if shown).

## Notes

- **One command builds everything.** `beforeBuildCommand` in
  `src-tauri/tauri.conf.json` runs `ensure-sidecar.mjs --force` (rebuilds the
  Python sidecar) and `npm run build` (static export to `frontend/out`) before
  `cargo tauri build` packages the app. You do NOT need to build the sidecar or
  frontend separately.
- **DLP workaround is baked in.** `tauri.sh` redirects `CARGO_TARGET_DIR` and
  `TMPDIR` under `~/_NOAV/XcodeDerivedData`. Never run `cargo tauri` directly.
  `src-tauri/target/` is stale by design (pre-`tauri.sh` leftovers) — do not
  recreate it; if it reappears from a stray direct run, delete it.
- **Keep the active caches.** These live outside git and make the next build
  fast — do NOT delete during deploy: `~/_NOAV/XcodeDerivedData/cargo-target`
  (Rust), `desktop/.venv-desktop` (sidecar venv), `frontend/.next`,
  `desktop/build` + `desktop/dist` (sidecar staging).
- **Build outputs are gitignored** (`src-tauri/resources/sidecar/`,
  `desktop/dist/`, `frontend/out`) — only source changes are committed.
- API keys come from the macOS Keychain via the app's settings UI, not `.env`.
- This is NOT a versioned installer release (no tag, no GitHub Release). For
  that, use `node scripts/cicd/release-desktop.mjs` (see `docs/CICD.md`).

## Safety Rules

- Never use `--force` / `--force-with-lease` unless explicitly requested.
- Never trigger a production web deploy with uncommitted changes left behind —
  commit them first (Phase 3 step 2) or stop and tell the user.
- If Phase 1 fails, stop. Do not install, commit, push, or trigger anything.
- If the Phase 2 smoke test fails, stop. Do not push or trigger anything.

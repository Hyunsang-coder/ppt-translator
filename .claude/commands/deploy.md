# Deploy (Local Build → Web)

Two-phase deploy: (1) local Tauri production build, (2) commit/push + trigger
the web deploy so the public download page updates.

## Phase 1 — Local production build

1. Confirm no dev session is running:
   `ps aux | grep -iE "cargo tauri|ppt-translator" | grep -v grep`
   - If one is up, stop it first (or tell the user) instead of building over it.
2. Run the production bundle (run in foreground; release compile + PyInstaller
   sidecar rebuild can take a long time on first run):
   ```bash
   TAURI_BUILD=1 ./desktop/tauri.sh build
   ```
3. Verify the bundles exist under `src-tauri/target/*/release/bundle/`
   (`.dmg` / `.app` on macOS) and report their paths.

## Phase 2 — Commit, push, trigger web deploy

4. Check prerequisites: `gh auth status` must succeed; `git status` to see what
   will be committed.
5. Commit and push the current branch:
   - Commit message follows repo convention (`fix:` / `feat:` / `chore:` prefix,
     describing the included changes).
   - Push with `git push` (or `git push -u origin <branch>` if no upstream).
6. Update the web download page (`https://ppt-translator.vercel.app`):
   - On `main`: the push itself triggers `Full Validation`, and `Deploy Web`
     runs automatically on success. Do NOT fire a manual duplicate — watch the
     auto-triggered runs instead:
     ```bash
     gh run list --workflow=predeploy.yml --branch main --limit 3
     gh run list --workflow=deploy-web.yml --limit 3
     ```
   - On any other branch: trigger explicitly for the pushed ref, then watch:
     ```bash
     gh workflow run deploy-web.yml --ref <branch>
     gh run watch <run-id>
     ```
7. Report: bundle paths from Phase 1, pushed commits, and the Deploy Web run
   outcome (+ deployment URL if shown).

## Notes

- **One command builds everything.** `beforeBuildCommand` in
  `src-tauri/tauri.conf.json` runs `ensure-sidecar.mjs --force` (rebuilds the
  Python sidecar) and `npm run build` (static export to `frontend/out`) before
  `cargo tauri build` packages the app. You do NOT need to build the sidecar or
  frontend separately.
- **DLP workaround is baked in.** `tauri.sh` redirects `CARGO_TARGET_DIR` and
  `TMPDIR` under `~/_NOAV/XcodeDerivedData`. Never run `cargo tauri` directly.
- **Build outputs are gitignored** (`src-tauri/target/`,
  `src-tauri/resources/sidecar/`, `desktop/dist/`) — "local app file changes"
  stay local; only source changes are committed.
- API keys come from the macOS Keychain via the app's settings UI, not `.env`.
- This is NOT a versioned installer release (no tag, no GitHub Release). For
  that, use `node scripts/cicd/release-desktop.mjs` (see `docs/CICD.md`).

## Safety Rules

- Never use `--force` / `--force-with-lease` unless explicitly requested.
- Never trigger a production web deploy with uncommitted changes left behind —
  commit them first (step 5) or stop and tell the user.
- If Phase 1 fails, stop. Do not commit, push, or trigger anything.

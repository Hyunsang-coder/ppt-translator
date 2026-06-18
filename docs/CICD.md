# CI/CD

Automated validation, desktop release, and public web deployment for PPT 번역캣.

## Pipelines

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | PR, manual | Fast backend tests + frontend typecheck |
| `predeploy.yml` (`Full Validation`) | Push to `main`, manual | Full backend pytest + frontend production build |
| `deploy-web.yml` | After successful `Full Validation` on `main`, desktop release, manual | Deploy download page to Vercel production |
| `desktop-release.yml` | Tag `v*`, manual | Build + sign macOS/Windows installers, generate auto-update manifest, publish GitHub Release, deploy web |

## End-to-end release flow

1. Merge changes to `main`.
2. `Full Validation` runs automatically.
3. On success, `Deploy Web` updates `https://ppt-translator.vercel.app`.
4. When a desktop build is ready:
   ```bash
   node scripts/cicd/release-desktop.mjs
   ```
   This bumps `0.1.x`, commits, tags `v0.1.x`, and pushes.
5. `Desktop Release` builds + signs installers, generates the `latest.json` auto-update manifest, publishes GitHub Release assets, then calls `Deploy Web`.

The public download page reads GitHub `releases/latest` metadata, so version/date text stays in sync after each release.

## Auto-update

The desktop app checks GitHub Releases for a newer version on launch (and from **설정 → 앱 업데이트**), then downloads and installs in-app via `tauri-plugin-updater`. This needs three things in each release, all produced automatically by `tauri-apps/tauri-action` in `desktop-release.yml`:

- updater bundles (`*.app.tar.gz`, `*-setup.nsis.zip`) — generated because `bundle.createUpdaterArtifacts: true` in `src-tauri/tauri.conf.json`
- a `*.sig` minisign signature per bundle — requires the `TAURI_SIGNING_*` secrets
- `latest.json` — the manifest the app polls, at `releases/latest/download/latest.json`

The app verifies updates against the public key in `tauri.conf.json` → `plugins.updater.pubkey`.

> Auto-update only activates from **0.1.6 onward** (the first build containing the updater plugin). A 0.1.5 install cannot self-detect 0.1.6; users must download 0.1.6 manually once, after which 0.1.6 → 0.1.7 updates in-app.

### Updater signing key (one-time)

```bash
npx @tauri-apps/cli@^2 signer generate -w ~/.tauri/ppt-translator-updater.key
```

- Public key → already set in `tauri.conf.json` `plugins.updater.pubkey`. Regenerating the key requires updating that field.
- Private key + password → GitHub secrets `TAURI_SIGNING_PRIVATE_KEY` (file contents) and `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.
- **Keep the private key safe.** Losing it means existing installs can no longer verify updates.

### macOS code-signing & notarization (one-time)

Needs an Apple Developer ID. With these secrets set, `tauri-action` signs and notarizes automatically — no Gatekeeper warning. If they are absent the build still succeeds but stays unsigned (Gatekeeper warning remains; auto-update still works).

1. Export the **Developer ID Application** certificate as `.p12`, then base64-encode it:
   ```bash
   base64 -i certificate.p12 | pbcopy   # → APPLE_CERTIFICATE
   ```
2. Create an app-specific password at https://appleid.apple.com → APPLE_PASSWORD.
3. Add the secrets below.

## One-time Vercel setup

GitHub Actions needs access to your existing Vercel project (`hyunsang-coders-projects/ppt-translator`).

### Option A — bootstrap script (recommended)

1. Create a Vercel token: https://vercel.com/account/tokens
2. Run:

```bash
VERCEL_TOKEN=your_token node scripts/cicd/bootstrap-vercel.mjs
```

This configures all GitHub secrets, sets Vercel `rootDirectory=frontend` and `productionBranch=main`, creates a production deploy hook, and triggers the first production deploy.

### Option B — manual secret only

Add one GitHub repository secret:

- `VERCEL_TOKEN`

`Deploy Web` will trigger Vercel production deployments through the GitHub-connected project API. For faster redeploys, run the bootstrap script later to also set `VERCEL_DEPLOY_HOOK_URL`.

## Required GitHub secrets

| Secret | Required | Set by |
|--------|----------|--------|
| `VERCEL_TOKEN` | Yes | Bootstrap or manual |
| `VERCEL_DEPLOY_HOOK_URL` | Optional | Bootstrap |
| `VERCEL_ORG_ID` | Optional | Bootstrap |
| `VERCEL_PROJECT_ID` | Optional | Bootstrap |
| `TAURI_SIGNING_PRIVATE_KEY` | Yes for auto-update | `signer generate` (see Auto-update) |
| `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` | Yes for auto-update | `signer generate` (see Auto-update) |
| `APPLE_CERTIFICATE` | For macOS notarization | Developer ID `.p12`, base64 |
| `APPLE_CERTIFICATE_PASSWORD` | For macOS notarization | `.p12` export password |
| `APPLE_SIGNING_IDENTITY` | For macOS notarization | e.g. `Developer ID Application: Name (TEAMID)` |
| `APPLE_ID` | For macOS notarization | Apple account email |
| `APPLE_PASSWORD` | For macOS notarization | App-specific password |
| `APPLE_TEAM_ID` | For macOS notarization | Apple Developer Team ID |

Without the `TAURI_SIGNING_*` secrets the release builds succeed but produce no auto-update manifest. Without the `APPLE_*` secrets, macOS installers are unsigned (Gatekeeper warning) but auto-update still works.

## Manual commands

```bash
# Deploy web now
gh workflow run deploy-web.yml --ref main

# Release next desktop patch version
node scripts/cicd/release-desktop.mjs

# Release explicit version
node scripts/cicd/release-desktop.mjs 0.1.6
```

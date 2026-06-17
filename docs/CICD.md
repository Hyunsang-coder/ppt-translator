# CI/CD

Automated validation, desktop release, and public web deployment for PPT 번역캣.

## Pipelines

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | PR, manual | Fast backend tests + frontend typecheck |
| `predeploy.yml` (`Full Validation`) | Push to `main`, manual | Full backend pytest + frontend production build |
| `deploy-web.yml` | After successful `Full Validation` on `main`, desktop release, manual | Deploy download page to Vercel production |
| `desktop-release.yml` | Tag `v*`, manual | Build macOS/Windows installers, publish GitHub Release, deploy web |

## End-to-end release flow

1. Merge changes to `main`.
2. `Full Validation` runs automatically.
3. On success, `Deploy Web` updates `https://ppt-translator.vercel.app`.
4. When a desktop build is ready:
   ```bash
   node scripts/cicd/release-desktop.mjs
   ```
   This bumps `0.1.x`, commits, tags `v0.1.x`, and pushes.
5. `Desktop Release` builds installers, publishes GitHub Release assets, then calls `Deploy Web`.

The public download page reads GitHub `releases/latest` metadata, so version/date text stays in sync after each release.

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

Desktop release builds do not require extra secrets today. Unsigned installers use the default `GITHUB_TOKEN`.

## Manual commands

```bash
# Deploy web now
gh workflow run deploy-web.yml --ref main

# Release next desktop patch version
node scripts/cicd/release-desktop.mjs

# Release explicit version
node scripts/cicd/release-desktop.mjs 0.1.6
```

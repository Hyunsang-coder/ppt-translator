# Desktop Releases

GitHub Actions builds macOS and Windows installers from `.github/workflows/desktop-release.yml`.
The public download page links to the latest GitHub Release assets through stable names:

- `ppt-translator-macos-arm64.dmg`
- `ppt-translator-macos-x64.dmg`
- `ppt-translator-windows-x64-setup.exe`

The Vercel download page also reads GitHub's latest release metadata (tag and
published date) so users can see which version they are downloading.

## Release Flow

1. Bump `version` in `src-tauri/tauri.conf.json` and `frontend/package.json`.
2. Commit and push to `main`.
3. Create a version tag such as `v0.1.0` and push it, or run the `Desktop Release`
   workflow manually with `version=0.1.0`.

The workflow then:

1. Builds these installers:
   - macOS Apple Silicon on `macos-15`
   - macOS Intel on `macos-15-intel`
   - Windows x64 on `windows-latest`
2. Verifies the release tag matches `src-tauri/tauri.conf.json`.
3. Creates or updates the GitHub Release and uploads the installers as `latest`.
4. Optionally triggers a Vercel redeploy so the public download page refreshes.

See [`docs/CICD.md`](CICD.md) for one-time Vercel bootstrap and release commands.

No repository secrets are required for the current unsigned release flow. GitHub's
built-in `GITHUB_TOKEN` is used to create or update the Release.

### Vercel production deploy

Run once:

```bash
VERCEL_TOKEN=your_token node scripts/cicd/bootstrap-vercel.mjs
```

Or add only `VERCEL_TOKEN` to GitHub repository secrets. After that, `Deploy Web`
runs automatically after successful `main` validation and after each desktop release.

## Unsigned Build Limits

Unsigned installers are enough for early distribution, but they are not a
production trust setup:

- macOS DMGs are not notarized, so Gatekeeper can warn that the app cannot be
  verified.
- Windows installers are not code-signed, so Microsoft SmartScreen can warn
  users before installation.
- Tauri updater support should not be enabled for production until update
  artifacts are signed with `TAURI_SIGNING_PRIVATE_KEY`.

## macOS Signing And Notarization

Required Apple Developer prerequisites:

- Apple Developer Program membership.
- Developer ID Application certificate exported as a password-protected `.p12`.
- Notarization credentials, either Apple ID app-specific password or App Store
  Connect API key credentials.

Recommended GitHub Secrets:

- `APPLE_CERTIFICATE`: base64-encoded `.p12` certificate.
- `APPLE_CERTIFICATE_PASSWORD`: export password for the `.p12`.
- `APPLE_SIGNING_IDENTITY`: Developer ID Application signing identity.
- `APPLE_ID`: Apple account email for notarization.
- `APPLE_PASSWORD`: app-specific password for `APPLE_ID`.
- `APPLE_TEAM_ID`: Apple Developer Team ID.

Alternative notarization secrets if using App Store Connect API authentication:

- `APPLE_API_ISSUER`
- `APPLE_API_KEY`
- `APPLE_API_KEY_PATH` or a workflow step that writes the private key file.

## Windows Code Signing

Required Windows prerequisite:

- A code-signing certificate. EV certificates usually avoid SmartScreen warnings
  faster than OV certificates, but the exact setup depends on the certificate
  issuer and signing backend.

Common GitHub Secrets and config inputs:

- Certificate material or signing-service credentials, depending on the issuer.
- `TAURI_WINDOWS_SIGNTOOL_PATH` only when a custom `signtool.exe` path is needed.
- Tauri `bundle.windows` signing configuration, such as a custom `signCommand`
  for Azure Trusted Signing, EV token tooling, or another signing service.

Do not commit private keys, certificate files, provisioning profiles, or API key
files to this repository.

## Updater Signing

If automatic in-app updates are added later, generate a Tauri updater key pair:

```bash
cargo tauri signer generate -w ~/.tauri/ppt-translator.key
```

Store the private key outside the repository and add these GitHub Secrets:

- `TAURI_SIGNING_PRIVATE_KEY`
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`, if the key is encrypted.

The public key belongs in `tauri.conf.json` under the updater plugin
configuration, and every updater artifact must be published with its `.sig`
signature file.

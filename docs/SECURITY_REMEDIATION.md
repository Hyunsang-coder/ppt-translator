# Security Remediation Memory

Last reviewed: 2026-09-04

This project is a Tauri desktop application with a local FastAPI sidecar. Keep
the following security invariants when changing the runtime or release flow:

- The API binds to loopback by default. Any non-loopback `API_HOST` requires a
  non-empty `SIDECAR_AUTH_TOKEN`.
- Tauri generates a fresh UUID capability token per sidecar process. The token
  is kept in Rust/frontend memory only and is sent as `X-Sidecar-Token`.
- CORS origins are explicit. Never reintroduce `allow_origins=["*"]` or the
  removed `CORS_ALLOW_ALL` bypass.
- The default PPT/PPTX upload limit is 1024 MiB (desktop-only sidecar; bomb
  protection lives in the ZIP-content caps, not this number). The request-body
  middleware runs before FastAPI multipart/JSON parsing, and per-upload limits
  remain in place.
- All OOXML ZIP input must pass `validate_zip_archive()` before
  `python-pptx`, Pillow, or image compression reads entries. Preserve the
  entry-count, uncompressed-size, compression-ratio, path, and image-pixel
  checks.
- LLM instances must be created through `src.chains.llm_factory.create_llm()`
  so provider rate limiters remain shared.
- Desktop dependencies are installed only from
  `desktop/requirements-desktop.lock` with `--require-hashes`. Regenerate the
  lockfile intentionally when changing `requirements-desktop.txt`.
- Do not log generated translations, instructions, uploaded text, or API keys.

Required verification after security-sensitive changes:

```bash
pytest -q tests/
cd frontend && npx tsc --noEmit && npm run build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo test --manifest-path src-tauri/Cargo.toml
python -m pip install --dry-run --ignore-installed --require-hashes -r desktop/requirements-desktop.lock
```

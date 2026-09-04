# 보안 검토 및 조치 보고서

검토·조치일: 2026-09-04
범위: FastAPI 사이드카, Tauri 셸, Next.js/React UI, 업로드·압축 처리, LLM 호출, 빌드·의존성 설정.

## 결과

초기 보안 검토에서 확인된 SEC-001~SEC-008을 모두 코드와 배포 설정에 반영했다. 기본 API는 loopback에만 바인딩되고, Tauri 실행 시 매번 새 capability token을 사용한다. 업로드·multipart·JSON·OOXML ZIP에는 계층별 크기와 구조 제한을 적용했으며, 데스크톱 Python 의존성은 hash-pinned lockfile로 설치하도록 변경했다.

| 우선순위 | ID | 조치 상태 | 핵심 조치 |
|---|---|---|---|
| 즉시 | SEC-001 | 완료 | 기본 `127.0.0.1` 바인딩, 비-loopback은 token 없이는 실행 거부 |
| 즉시 | SEC-002 | 완료 | ZIP metadata/media 검증, 압축비·전개 크기·엔트리·픽셀 제한 |
| 즉시 | SEC-003 | 완료 | Next.js `16.1.6` → `16.3.4`, lockfile 재생성 |
| 높음 | SEC-004 | 완료 | ASGI 전체 본문 제한, 청크 업로드 제한, 입력 필드 상한 |
| 높음 | SEC-005 | 완료 | wildcard CORS 제거, Tauri per-run token을 모든 API 요청에 적용 |
| 계획 | SEC-006 | 완료 | provider별 process-shared LLM rate limiter 및 우회 경로 제거 |
| 계획 | SEC-007 | 완료 | 데스크톱 lockfile + `--require-hashes` 설치 |
| 낮음 | SEC-008 | 완료 | 모델 응답 본문을 INFO 로그에서 제거 |

## 조치 내역

### SEC-001 — 인증 없는 API의 LAN 노출

- `api.py:2252-2259`의 직접 실행 기본값을 `127.0.0.1`로 변경했다.
- `API_HOST`가 loopback이 아니면 `SIDECAR_AUTH_TOKEN` 없이는 실행하지 않는다.
- token이 설정된 프로세스는 `/health`와 `/api/*`에 `X-Sidecar-Token`을 요구한다(`api.py:213-222`).
- 비-Tauri 로컬 개발은 기존처럼 loopback에서 token 없이 사용할 수 있다.

### SEC-002 — ZIP 폭탄·압축 해제·이미지 자원 고갈

- `src/utils/security.py:26-130`에 공유 ZIP 검증기를 추가했다.
- PPTX/XLSX 필수 엔트리, 최대 10,000개 엔트리, 엔트리당 64 MiB, 전체 256 MiB, 최대 압축비 200, 중복·탈출 경로·심볼릭 링크·암호화 ZIP을 검사한다.
- PPTX의 raster media는 Pillow로 헤더 크기를 확인하고 40,000,000 픽셀을 초과하면 거부한다.
- 번역·추출·이미지 압축 전에 같은 검증기를 사용한다(`api.py`, `src/utils/image_compressor.py`).

### SEC-003 — Next.js 보안 패치

- `frontend/package.json`과 `frontend/package-lock.json`을 Next.js `16.3.4`로 갱신했다.
- `frontend`에서 TypeScript 검사와 `next build`를 통과했다.

### SEC-004 — 요청·파일 크기 제한

- `api.py:77-145`의 ASGI middleware가 `Content-Length`와 chunked body를 FastAPI 파싱 전에 제한한다.
- 기본 PPT/PPTX 업로드 한도는 256 MiB이며 `MAX_UPLOAD_SIZE_MB`는 512 MiB까지로 제한한다.
- 전체 HTTP body 기본 한도는 272 MiB이며 `MAX_REQUEST_BODY_MB`는 544 MiB까지다.
- glossary/rules 업로드, glossary JSON, markdown, context/instructions, provider/model 및 기타 문자열 필드에 독립 상한을 적용했다.
- 프런트엔드 기본 표시·dropzone 한도도 256 MiB로 맞췄다.

### SEC-005 — wildcard CORS·sidecar 인증

- `api.py:167-202`에서 wildcard CORS와 `allow_methods/headers=["*"]`를 제거했다.
- Tauri 및 로컬 개발에 필요한 명시적 origin만 기본 허용하며, 사용자 설정에 `*`가 들어오면 무시한다.
- `src-tauri/src/lib.rs:217-265`가 실행마다 UUID token을 만들고 sidecar 환경 변수로 전달한다.
- token은 Rust 메모리에만 보관하고 ready event, 파일, 로그에는 넣지 않는다. 프런트는 IPC로 받아 메모리 내에서만 모든 API 요청 헤더에 주입한다.

### SEC-006 — LLM 비용·속도 제한 우회

- `src/chains/llm_factory.py:23-40`에서 provider별 limiter를 `lru_cache`로 공유한다.
- `/generate-instructions`의 Anthropic 경로도 `create_llm()`을 사용해 동일한 limiter를 거친다.

### SEC-007 — Python 공급망 재현성

- `desktop/requirements-desktop.lock`을 버전·hash 고정으로 생성했다.
- `desktop/build-sidecar.mjs`, `desktop/README.md`, GitHub desktop release workflow가 모두 `pip install --require-hashes -r desktop/requirements-desktop.lock`을 사용한다.
- sidecar stale 판정에도 lockfile을 포함했다.
- lockfile은 `pip install --dry-run --ignore-installed --require-hashes`로 검증했다.

### SEC-008 — 민감한 모델 응답 로그

- `api.py:2230-2235`의 INFO 로그는 provider, model, target language, 결과 길이만 남긴다.
- 생성된 instruction 본문은 로그에 기록하지 않는다.

## 영구 운영 메모

- 운영 설정과 sidecar token 동작은 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)에 기록했다.
- 변경 이유·유지보수 규칙·검증 명령은 [`docs/SECURITY_REMEDIATION.md`](docs/SECURITY_REMEDIATION.md)에 기록했다.
- Tauri updater 공개키/HTTPS, OS keychain, 파일명 정제, formula injection 방어, model/provider allowlist는 기존 방어를 유지한다.

## 검증 결과

- `pytest -q tests/`: 358 passed, 8 existing openpyxl deprecation warnings.
- `pytest -q tests/test_security_fix.py tests/test_api.py`: 58 passed.
- `python -m compileall -q api.py src`: passed.
- `frontend`: `npx tsc --noEmit` 및 `npm run build` passed on Next.js 16.3.4.
- `cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check`: passed.
- `cargo test --manifest-path src-tauri/Cargo.toml`: 2 passed.
- `python -m pip install --dry-run --ignore-installed --require-hashes -r desktop/requirements-desktop.lock`: passed.

참고: 로컬 `npm ci`는 npm registry 응답 지연으로 완료되지 않아 중단했지만, 기존 설치 의존성으로 TypeScript와 production build를 통과했고 lockfile은 별도로 생성·검토했다.

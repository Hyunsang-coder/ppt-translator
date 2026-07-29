# ADR-0005: 배포는 Tauri 데스크톱 전용, 웹은 다운로드 페이지만

- **상태**: 채택됨
- **날짜**: 2026-06-15

## 맥락

이 앱은 원래 EC2에 올린 웹 서비스로 돌았다. 그런데 사용자가 다루는 자산이 회사 PPTX 원본이고, 서버가 API 키와 파일을 모두 떠안는 구조였다. 서버 운영·키 보관·업로드 용량이 전부 유지비였고, 정작 얻는 건 "설치 없이 쓴다" 하나였다.

## 결정

저장소를 데스크톱 앱 중심으로 재편한다. Tauri 셸이 API 키를 OS 키체인에 두고, 번들된 Python 사이드카를 띄우고, 그 포트를 WebView에 넘긴다. 번역과 파일은 사용자 기기를 벗어나지 않는다(LLM 호출 제외).

Vercel에 남는 웹은 공개 다운로드 안내 페이지 하나다. GitHub `releases/latest` 메타데이터를 읽어 버전·날짜를 스스로 맞춘다. 호스팅된 웹에서 앱 라우트로 들어오면 `desktop-shell`이 공개 루트로 돌려보낸다.

배포는 GitHub Releases + `tauri-plugin-updater` 인앱 자동 업데이트로 한다.

## 결과

- 서버 운영 비용과 키 보관 책임이 사라졌다. 사용자가 각자 자기 키를 넣는다.
- 대신 설치·서명·공증·업데이트 파이프라인이 새 유지비로 생겼다. 자동 업데이트는 **0.1.6부터** 동작한다 — 그 이전 설치본은 스스로 갱신을 감지하지 못해 한 번은 수동 설치가 필요하다.
- macOS Intel 빌드는 제공하지 않는다.
- 사용자 기기의 Python 사이드카 수명 관리가 우리 문제가 됐다. Windows 업데이트 시 NSIS가 파일을 덮기 전에 사이드카를 멈춰야 하고(`installer-hooks.nsh`), 이 훅이 빠지면 Pillow의 `_imaging*.pyd` 같은 네이티브 모듈이 잠긴 채 남는다.

## 참고

- 커밋: `1097504`, `5b8f714`, `754b8c5`
- 코드: `src-tauri/`, `desktop/sidecar.py`, `frontend/src/components/desktop-shell.tsx`
- 문서: [`docs/CICD.md`](../CICD.md) "Auto-update", [`docs/DESKTOP_RELEASES.md`](../DESKTOP_RELEASES.md)

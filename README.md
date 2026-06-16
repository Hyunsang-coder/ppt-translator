# PPT 번역캣

**https://ppt-translator.vercel.app**

PowerPoint 번역 데스크톱 앱입니다. Tauri 셸이 Next.js UI를 띄우고,
번역 엔진은 로컬 FastAPI sidecar로 실행됩니다. OpenAI / Anthropic API
키는 OS 보안 저장소(macOS Keychain, Windows Credential Manager)에 저장됩니다.

## 아키텍처

- **Desktop shell**: Tauri 2 (`src-tauri/`)
- **UI**: Next.js 16 + React 19 + TypeScript (`frontend/`)
- **Sidecar**: FastAPI + LangChain Python server (`desktop/`, `api.py`, `src/`)
- **Public web**: Vercel root page only, for desktop download 안내

## 주요 기능

- PPT/PPTX 업로드 후 원본 서식을 보존하면서 번역된 PPTX 생성
- 반복 문구 사전 처리로 중복 문장은 한 번만 번역하고 결과를 재사용
- 소스/타겟 언어 자동 감지 및 추론 (`langdetect`)
- Excel 용어집 업로드 + 번역 전/후 용어 치환
- AI 기반 컨텍스트 요약 및 번역 스타일 가이드 자동 생성
- 다색 문단 서식 보정 (LLM 기반 색상 분배)
- 텍스트 피팅 (자동 축소, 박스 확장) 지원
- PPT → Markdown 텍스트 추출 기능

## 설치 방법

### 요구 사항
- Python >= 3.12
- Node.js >= 18
- Rust + Cargo
- Tauri prerequisites for your OS

### Python sidecar
```bash
python3 -m venv desktop/.venv-desktop
desktop/.venv-desktop/bin/pip install -r desktop/requirements-desktop.txt
```

### Frontend
```bash
cd frontend
npm install
```

## 실행 방법

```bash
# 데스크톱 앱 개발 실행
cd src-tauri
cargo tauri dev
```

Tauri 개발 실행은 Python sidecar가 없거나 오래된 경우 자동으로 빌드하고,
`frontend` 개발 서버도 자동으로 띄웁니다. Rust 셸이 sidecar 포트를 WebView에
전달합니다.

## 용어집 파일 형식

- `glossary_template.xlsx` 템플릿을 다운로드하여 사용합니다.
- A열: 원문, B열: 번역 (예: PUBG → 배틀그라운드)
- 헤더를 포함한 단일 시트 구조만 지원합니다.

## 테스트

```bash
pytest tests/ -v

cd frontend && npx tsc --noEmit && npm run build
```

## 지원 모델

| Provider | Model ID | Display Name |
|----------|----------|--------------|
| OpenAI | `gpt-5.5-2026-04-23` | GPT-5.5 |
| OpenAI | `gpt-5.4-mini-2026-03-17` | GPT-5.4 Mini |
| Anthropic | `claude-opus-4-8` | Claude Opus 4.8 |
| Anthropic | `claude-sonnet-4-6` | Claude Sonnet 4.6 |
| Anthropic | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 |

## 배포

- **Desktop**: `cd src-tauri && TAURI_BUILD=1 cargo tauri build`
- **Vercel**: 데스크톱 앱 다운로드 안내용 루트 페이지만 배포

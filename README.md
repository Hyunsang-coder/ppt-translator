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
- 앱 내 용어집 라이브러리: 수동 편집, 다중 활성화/우선순위, CSV·Excel 가져오기/내보내기
- 수동 컨텍스트 및 번역 스타일 지침 입력
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

## 용어집 관리

- 번역 화면에서 여러 용어집을 만들고 원문, 번역, 메모를 직접 추가·수정·삭제할 수 있습니다.
- 사용할 용어집을 복수 선택하고 우선순위를 정할 수 있습니다. 같은 원문이 여러 활성 용어집에 있으면 우선순위가 높은 항목을 사용합니다.
- 용어집은 앱의 브라우저 저장소(`localStorage`)에 저장되므로 같은 앱 출처에서 다시 열면 유지됩니다. 중요한 용어집은 CSV 또는 Excel로 내보내 백업하세요.
- CSV, TSV, 세미콜론 구분 파일과 Excel(`.xlsx`, `.xls`)을 가져올 수 있습니다. 첫 세 열은 원문, 번역, 메모이며 메모는 선택 사항입니다.
- `glossary_template.xlsx`도 계속 사용할 수 있습니다. A열은 원문, B열은 번역, C열은 선택 메모입니다.

## 테스트

```bash
pytest tests/ -v

cd frontend && npm test && npx tsc --noEmit && npm run build
```

## 지원 모델

| Provider | Model ID | Display Name |
|----------|----------|--------------|
| OpenAI | `gpt-5.6-sol` | GPT-5.6 Sol (High) |
| OpenAI | `gpt-5.6-luna` | GPT-5.6 Luna (High) |
| Anthropic | `claude-opus-4-8` | Claude Opus 4.8 |
| Anthropic | `claude-sonnet-5` | Claude Sonnet 5 |
| Anthropic | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 |

## 배포

- **Desktop**: `cd src-tauri && TAURI_BUILD=1 cargo tauri build`
- **Vercel**: 데스크톱 앱 다운로드 안내용 루트 페이지만 배포

# PPT 번역캣

**https://ppt-translator.vercel.app**

PowerPoint 번역 웹 애플리케이션으로, LangChain과 OpenAI GPT / Anthropic Claude 모델을 활용해 슬라이드 텍스트를 고품질로 번역합니다. 원본 서식(볼드, 이탤릭, 폰트, 색상)을 최대한 유지하면서 용어집, 자동 언어 감지, 실시간 진행률 표시를 지원합니다.

## 아키텍처

- **Frontend**: Next.js 16 + React 19 + TypeScript (Vercel 배포)
- **Backend**: FastAPI + LangChain (EC2 Docker 배포)
- **통신**: REST API + Server-Sent Events (SSE) 실시간 진행률

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

### Backend
```bash
# 가상 환경 생성 및 활성화
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 설정
```

### Frontend
```bash
cd frontend
npm install

# 로컬 개발용 환경 변수
echo 'NEXT_PUBLIC_API_URL=http://localhost:8000' > .env.local
```

## 실행 방법

```bash
# Backend (FastAPI)
uvicorn api:app --reload --port 8000

# Frontend (Next.js) - 별도 터미널
cd frontend && npm run dev
```

브라우저에서 `http://localhost:3000`으로 접속합니다.

## 용어집 파일 형식

- `glossary_template.xlsx` 템플릿을 다운로드하여 사용합니다.
- A열: 원문, B열: 번역 (예: PUBG → 배틀그라운드)
- 헤더를 포함한 단일 시트 구조만 지원합니다.

## 테스트

```bash
# Backend 유닛 테스트
pytest tests/ -v

# Frontend 타입 체크 및 빌드
cd frontend && npx tsc --noEmit && npm run build
```

## 지원 모델

| Provider | Model ID | Display Name |
|----------|----------|--------------|
| OpenAI | `gpt-5.2` | GPT-5.2 |
| OpenAI | `gpt-5-mini` | GPT-5 Mini |
| Anthropic | `claude-opus-4-6` | Claude Opus 4.6 |
| Anthropic | `claude-sonnet-4-6` | Claude Sonnet 4.6 |
| Anthropic | `claude-haiku-4-5-20251001` | Claude Haiku 4.5 |

## 배포

- **Frontend**: Vercel (자동 배포, `vercel.json`으로 API 프록시)
- **Backend**: EC2 + Docker (`docker compose up -d --build`)

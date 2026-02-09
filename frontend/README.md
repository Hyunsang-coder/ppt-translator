# PPT 번역캣 — Frontend

Next.js 16 기반의 PPT 번역 웹 애플리케이션 프론트엔드입니다.

## Tech Stack

- **Next.js 16** (App Router) + **React 19**
- **TypeScript 5**
- **Tailwind CSS 4**
- **Zustand 5** (상태 관리)
- **Radix UI** (접근성 컴포넌트)
- **Lucide React** (아이콘)

## Getting Started

```bash
# 의존성 설치
npm install

# 개발 서버 실행 (http://localhost:3000)
npm run dev

# 프로덕션 빌드
npm run build

# 타입 체크
npx tsc --noEmit
```

## Environment

`.env.local` 파일 설정:

```env
# 백엔드 API URL (로컬 개발 시)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Vercel 배포 시에는 비워두면 rewrites를 통해 프록시됨
# NEXT_PUBLIC_API_URL=
```

## Project Structure

```
src/
├── app/                  # Pages (App Router)
│   ├── page.tsx          # Home → /translate 리다이렉트
│   ├── translate/        # 번역 페이지
│   ├── extract/          # 텍스트 추출 페이지
│   ├── layout.tsx        # Root layout + ThemeProvider
│   └── globals.css       # CSS variables, OKLch colors
├── components/
│   ├── shared/           # Header, FileUploader
│   ├── translation/      # TranslationForm, SettingsPanel, ProgressPanel, LogViewer
│   ├── extraction/       # ExtractionForm, MarkdownPreview
│   └── ui/               # Shadcn/Radix 컴포넌트
├── hooks/                # useTranslation, useExtraction, useConfig
├── stores/               # Zustand stores (translation, extraction)
├── lib/                  # API client, SSE client, utils
└── types/                # TypeScript 타입 정의
```

## Deployment

Vercel에 배포되며, `vercel.json`의 rewrites를 통해 `/api/*` 요청을 EC2 백엔드로 프록시합니다.

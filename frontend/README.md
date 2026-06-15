# PPT 번역캣 — Frontend

Next.js 16 기반의 Tauri WebView UI입니다. 공개 Vercel 배포에서는 루트
페이지가 데스크톱 앱 다운로드 안내만 표시합니다. 실제 번역/추출 화면은
Tauri 데스크톱 앱에서 sidecar API와 연결되어 동작합니다.

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

Tauri 앱에서는 Rust 셸이 sidecar 포트를 런타임에 전달하므로
`.env.local`이 필요하지 않습니다. 브라우저에서 UI만 단독 개발할 때만
로컬 API URL을 지정합니다.

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

```
src/
├── app/                  # Pages (App Router)
│   ├── page.tsx          # Vercel 다운로드 안내
│   ├── translate/        # 데스크톱 번역 페이지
│   ├── extract/          # 데스크톱 텍스트 추출 페이지
│   ├── settings/         # 데스크톱 API 키 설정
│   ├── layout.tsx        # Root layout + ThemeProvider
│   └── globals.css       # CSS variables, OKLch colors
├── components/
│   ├── shared/           # Header, FileUploader
│   ├── translation/      # TranslationForm, SettingsPanel, ProgressPanel, LogViewer
│   ├── extraction/       # ExtractionForm, MarkdownPreview
│   ├── sidecar-provider.tsx
│   └── ui/               # Shadcn/Radix 컴포넌트
├── hooks/                # useTranslation, useExtraction, useConfig
├── stores/               # Zustand stores (translation, extraction)
├── lib/                  # API client, sidecar base URL, keychain, save-file helpers
└── types/                # TypeScript 타입 정의
```

## Build

`TAURI_BUILD=1`일 때 `next.config.ts`가 static export를 켜고 `frontend/out`
을 생성합니다. 이 출력물이 Tauri 번들에 포함됩니다.

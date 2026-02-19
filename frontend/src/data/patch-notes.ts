import { Sparkles, Bug, Wrench, type LucideIcon } from "lucide-react";

export type ChangeType = "feature" | "fix" | "improvement";

export interface Change {
  type: ChangeType;
  description: string;
}

export interface PatchNote {
  version: string;
  date: string;
  title: string;
  commitHash: string;
  changes: Change[];
}

export interface ChangeTypeConfig {
  label: string;
  icon: LucideIcon;
  colorClass: string;
  bgClass: string;
}

export const changeTypeConfig: Record<ChangeType, ChangeTypeConfig> = {
  feature: {
    label: "새 기능",
    icon: Sparkles,
    colorClass: "text-success",
    bgClass: "bg-success/10",
  },
  fix: {
    label: "버그 수정",
    icon: Bug,
    colorClass: "text-destructive",
    bgClass: "bg-destructive/10",
  },
  improvement: {
    label: "개선",
    icon: Wrench,
    colorClass: "text-info",
    bgClass: "bg-info/10",
  },
};

export const patchNotes: PatchNote[] = [
  {
    version: "2026.02.19",
    date: "2026-02-19",
    title: "Claude Sonnet 4.6 모델 업그레이드",
    commitHash: "959ebf8",
    changes: [
      { type: "feature", description: "기본 번역 모델을 Claude Sonnet 4.6으로 업그레이드" },
      { type: "improvement", description: "GitHub Actions CI/CD 파이프라인 추가 (테스트 및 타입 체크 자동화)" },
    ],
  },
  {
    version: "2026.02.13",
    date: "2026-02-13",
    title: "파일명 순서 지정, 길이 제한, 동시성 관리",
    commitHash: "6592001",
    changes: [
      { type: "feature", description: "출력 파일명 suffix를 체크박스 선택 순서대로 정렬" },
      { type: "feature", description: "번역 길이 제한 옵션 추가 (110/130/150%)" },
      { type: "improvement", description: "서버 동시 번역 작업 수 제한 및 대기열 관리" },
    ],
  },
  {
    version: "2026.02.10",
    date: "2026-02-10",
    title: "이미지 압축, 취소 개선, 안정성 강화",
    commitHash: "37149e8",
    changes: [
      { type: "feature", description: "이미지 압축 옵션 추가 (대용량 PPTX 최적화)" },
      { type: "feature", description: "패치 노트 페이지 추가" },
      { type: "fix", description: "파일 변경 시 job 상태 리셋 및 컨텍스트/지침 생성 중단" },
      { type: "fix", description: "업로드 중 취소 버튼 동작 (AbortController)" },
      { type: "fix", description: "SSE 레이스 컨디션, 취소 UI, 폴백 언어 처리" },
      { type: "fix", description: "스레드 안전 이벤트 디스패치 및 재시도 강화" },
      { type: "fix", description: "Job manager 레이스 컨디션, 이벤트 누수 수정" },
      { type: "improvement", description: "Streamlit 제거, FastAPI redirect 및 기본 설정" },
    ],
  },
  {
    version: "2025.02.09",
    date: "2025-02-09",
    title: "작동 원리 페이지",
    commitHash: "81d7b44",
    changes: [
      { type: "feature", description: "번역 파이프라인 11단계를 시각적으로 보여주는 작동 원리 페이지 추가" },
      { type: "fix", description: "취소 후 재시도 시 다운로드 404 에러 수정" },
      { type: "fix", description: "EC2 헬스체크 포트 및 SSH heredoc 문제 수정" },
      { type: "improvement", description: "배포/상태확인 명령어 추가" },
    ],
  },
  {
    version: "2025.02.04",
    date: "2025-02-04",
    title: "안정성 개선 및 버그 수정",
    commitHash: "79b97f0",
    changes: [
      { type: "fix", description: "SSE 메모리 누수, 용어집 매칭, 파일명 공백, Blob URL 문제 수정" },
      { type: "improvement", description: "폴백 언어 목록에 Auto 옵션 추가" },
      { type: "improvement", description: "백엔드 연결 사전 체크 제거로 안정성 향상" },
      { type: "feature", description: "재번역 기능 추가, 폰트 크기 반올림 처리" },
    ],
  },
  {
    version: "2025.01.28",
    date: "2025-01-28",
    title: "색상 분배 및 텍스트 피팅",
    commitHash: "080d1e5",
    changes: [
      { type: "feature", description: "비연속 색상 분배 지원 추가" },
      { type: "feature", description: "텍스트 피팅 시스템 (자동 축소, 박스 확장) 추가" },
      { type: "improvement", description: "색상 매칭 일관성 및 정확도 개선" },
      { type: "improvement", description: "전체 진행률 표시 및 기본 모델 Anthropic으로 변경" },
    ],
  },
  {
    version: "2025.01.22",
    date: "2025-01-22",
    title: "실시간 번역 및 슬라이드 노트",
    commitHash: "8c7dc4e",
    changes: [
      { type: "feature", description: "batch_as_completed 기반 실시간 번역 진행 표시" },
      { type: "feature", description: "슬라이드 노트 번역 지원 추가" },
      { type: "improvement", description: "색상 수정 상태 표시 및 메인 모델 사용" },
    ],
  },
];

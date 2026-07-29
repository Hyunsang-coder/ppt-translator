# Handoff: 번역 검토 화면 재설계 (검토 큐 방식)

## Overview

PPT 번역캣의 **번역 검토 & 수정 화면**(`frontend/src/components/translation/ReviewPanel.tsx`)을
"37개 문구가 흩뿌려진 타일 그리드"에서 **"확인이 필요한 항목만 한 건씩 처리하는 큐"**로 재설계한다.

핵심 목표 세 가지:

1. **처음 쓰는 사람도 무엇을 해야 할지 안다** — 화면에 항상 "지금 몇 번째 / 총 몇 건"이 있고, 한 화면에 판단할 것이 하나뿐이다.
2. **수정 1건이 5단계 → 1클릭** — 카드 → 탭 → 확인 → 비교 모달 → 적용 → 부분일치 시트를 없애고, `적용하고 다음` 버튼(또는 `⏎`) 하나로 끝낸다.
3. **마무리 동선이 명확하다** — 상단 `번역 → 검토 → 저장` 단계 표시, 하단 `PPT 저장` 단일 CTA, `r0 → r3` 같은 내부 리비전 표기 제거.

## About the Design Files

이 번들에 들어 있는 HTML은 **디자인 레퍼런스**다. 의도한 레이아웃·색·타이포·흐름을 보여주는 프로토타입이며,
그대로 복사해 넣을 프로덕션 코드가 **아니다**.

구현 대상은 이미 존재하는 환경이다:

- Next.js 16 + React 19 + TypeScript (`frontend/`)
- Tailwind CSS v4 (`frontend/src/app/globals.css`의 CSS 변수 토큰)
- shadcn/ui 컴포넌트 (`frontend/src/components/ui/*`)
- lucide-react 아이콘, sonner 토스트, zustand 스토어

따라서 **HTML을 옮겨 붙이지 말고**, 위 스택의 기존 패턴(`Button`, `Input`, `Textarea`, `cn()`,
토큰 클래스 `bg-card` / `text-muted-foreground` / `border-border` 등)으로 **다시 구현**한다.
프로토타입의 인라인 `oklch(...)` 값은 전부 `globals.css`에 이미 정의된 토큰과 1:1로 대응하므로,
하드코딩된 색을 넣을 일은 없어야 한다 (아래 Design Tokens 표 참고).

## Fidelity

**High-fidelity (hifi).** 색·타이포·간격·카피가 확정된 목업이다. 레이아웃과 수치를 그대로 재현한다.
단, 프로토타입은 정적 스냅샷이므로 로딩/에러/빈 상태는 아래 "Interactions & Behavior"의 서술을 따른다.

대상 뷰포트: **데스크톱 앱 창 1280×860** (Tauri WebView). 1024px 미만은 고려 대상 아님.

---

## Screens / Views

전체 화면은 기존과 동일하게 **풀스크린 오버레이**다.
루트: `fixed inset-0 z-50 flex flex-col bg-background` — 기존의 `bg-background/95 backdrop-blur-sm`는
뒤 화면이 비쳐 산만하므로 **불투명 `bg-background`로 변경**한다.

세로 구조 (위→아래):

```
┌─ StepHeader                      h=57   (border-b)
├─ ┌─────────┬──────────────────────────────┬──────────────┐
│  │ SlideRail│  QueueItem (main)            │ GlossaryPane │
│  │  w=230   │  flex-1                      │  w=264       │
│  │          │  ├ QueueItemHeader  h=~44    │              │
│  │          │  ├ QueueItemBody   flex-1    │              │
│  │          │  └ FinishBar       h=63      │              │
│  └─────────┴──────────────────────────────┴──────────────┘
```

좌우 패널은 `flex-shrink-0`, 가운데만 `flex-1 min-w-0`. 세로 스크롤은 각 패널 안에서 따로 일어난다.

---

### 1. StepHeader — 상단 단계 바

**Purpose**: "번역은 끝났고 지금은 검토 중이며 다음은 저장"이라는 위치감을 항상 제공.

- 컨테이너: `flex items-center justify-between gap-4 px-5 py-3 border-b border-border bg-card`
- **좌측 스텝퍼** — `flex items-center gap-2.5`
  - 스텝 1 `번역`: 완료 상태. `CheckCircle2` 16px, `text-success` + 라벨 `text-[13px] text-muted-foreground`
  - 구분자: `ChevronRight` 14px, `text-border`, `stroke-width 2.5`
  - 스텝 2 `검토`: 현재. 원형 배지 18×18 `rounded-full bg-primary text-primary-foreground text-[11px]` 안에 `2`, 라벨 `text-[13px] font-bold text-primary`
  - 스텝 3 `저장`: 미도달. 원형 배지 18×18 `rounded-full border-[1.5px] border-border text-[11px]`, 라벨 `text-[13px] text-muted-foreground/70`
- **우측** — `flex items-center gap-3`
  - 결과 파일명 `text-[13px] text-muted-foreground` (예: `2026_상반기_라이브서비스_EN.pptx`) — `filenameSettings`로 계산된 실제 출력 파일명
  - `되돌리기` — `<Button variant="outline" size="sm">` + `Undo2` 15px. `disabled={!dirty}`
  - 닫기 — `<Button variant="ghost" size="icon-sm">` + `X` 16px

> 기존 헤더에 있던 **용어집 빠른 추가 폼(저장할 용어집 / 원문 / 번역 / 용어 추가)은 헤더에서 제거**하고
> 우측 GlossaryPane으로 이동한다.

---

### 2. SlideRail — 좌측 진행 레일 (w=230)

**Purpose**: 큐가 어디까지 왔는지, 어느 슬라이드에 일이 남았는지.

- 컨테이너: `w-[230px] shrink-0 border-r border-border bg-card flex flex-col`
- **진행 요약 블록** — `px-4 pt-4 pb-3 border-b border-border`
  - 라벨 `검토 진행` — `text-xs font-semibold text-muted-foreground mb-2`
  - 카운터 — `flex items-baseline gap-1.5`: 처리 수 `text-[30px] font-bold tracking-[-0.03em] leading-none`, `/ 12건` `text-[15px] text-muted-foreground`
  - 진행 바 — `mt-2.5 h-1.5 rounded-full bg-muted`, 채움 `bg-primary`, width = `resolved / total * 100%`
  - (선택) 잔여 시간 텍스트 `text-xs text-muted-foreground` — **Phase 1에서는 표시하지 않는다.** 근거 데이터가 없다. 실측 평균 처리시간이 쌓이면 추가.
- **슬라이드 목록** — `flex-1 overflow-y-auto px-2.5 pt-2.5 pb-3.5 flex flex-col gap-0.5`
  - 섹션 라벨 `슬라이드별 남은 항목` — `text-[11px] font-semibold text-muted-foreground px-2 pt-1 pb-1.5`
  - **남은 항목이 있는 슬라이드만** 나열한다 (0건인 슬라이드는 숨김).
  - 행: `flex items-center gap-2.5 px-2.5 py-2 rounded-lg`
    - 슬라이드 번호 `w-5 text-xs font-bold text-muted-foreground`
    - 제목 `flex-1 text-[13px] truncate text-muted-foreground` (`slide_title`, 없으면 `슬라이드 N`)
    - 남은 개수 배지 `size-5 rounded-full bg-muted text-[11px] font-bold inline-flex items-center justify-center`
    - **활성 행**: `bg-primary/12`, 번호 `text-primary`, 제목 `font-semibold text-foreground`, 배지 `bg-primary text-primary-foreground`
    - hover: `hover:bg-muted/60`
  - 구분선 `h-px bg-border mx-2 my-2.5`
  - 완료 요약 행 `flex items-center gap-2 px-2.5 py-1.5 text-xs text-muted-foreground` + `Check` 14px `text-success` — `확인 끝난 슬라이드 4개`
  - 전체 보기 링크 `px-2.5 py-1.5 text-xs font-medium text-primary` — `전체 37개 문구 보기` → **전체 목록 모드**(아래 참고)로 전환

---

### 3. QueueItem — 가운데 본문

#### 3-1. QueueItemHeader

`flex items-center gap-2.5 px-7 pt-3.5`

- 검출 유형 배지 — `rounded-full px-[11px] py-[5px] text-xs font-bold`.
  색은 기존 `badgeStyle()` 매핑을 그대로 재사용하되 라벨을 평서문으로 교체:

  | `finding.type` | 배지 라벨 | 색 |
  |---|---|---|
  | `terminology.violation` | 용어집과 다름 | `text-destructive bg-destructive/10` |
  | `terminology.inconsistency`, `consistency.phrase` | 표현이 오락가락함 | `text-info bg-info/10` |
  | `accuracy.omission` | 번역이 빠짐 | `text-destructive bg-destructive/10` |
  | `fit.overflow` | 박스에 넘침 | `text-warning bg-warning/10` |
  | `fit.length_limit` | 길이 초과 | `text-warning bg-warning/10` |
  | `style.mapping_dropped` | 색상 확인 필요 | `text-warning bg-warning/10` |

- 위치 설명 — `text-[13px] text-muted-foreground`:
  `슬라이드 {slide} · {is_note ? "발표자 노트" : "본문 텍스트"}{repeat_count > 1 ? ` · 덱 안 ${repeat_count}곳에 반복` : ""}`
- 우측 페이저 — `ml-auto flex items-center gap-1.5`
  - 이전/다음: `size-[30px] rounded-md border border-border bg-card text-muted-foreground` + `ChevronLeft`/`ChevronRight` 15px
  - 가운데 `text-[13px] text-muted-foreground min-w-16 text-center` — `4번째 / 12`

#### 3-2. QueueItemBody

`flex-1 overflow-y-auto px-7 pt-4.5 pb-5`

1. **설명 문장** — `text-[13px] leading-relaxed text-muted-foreground mb-4.5`.
   `finding.description`을 그대로 쓰되, 핵심 값은 `<b className="text-foreground">`로 강조.
   예) `용어집에 **배틀 패스 → Battle Pass**로 등록되어 있는데, 번역에는 **Combat Pass**가 쓰였습니다.`

2. **원문 블록**
   - 라벨 `원문` — `text-[11px] font-semibold tracking-[0.04em] text-muted-foreground mb-[7px]`
   - 본문 — `text-[22px] leading-[1.45] tracking-[-0.01em] text-foreground/62`

3. **현재 번역 블록**
   - 라벨 줄: 좌측 `현재 번역`(위와 동일 스타일), 우측 길이 게이지 `flex items-center gap-2 text-xs text-muted-foreground`
     - 텍스트 `{target.length}자 / 박스 권장 {length_budget}자`
     - 바 `w-20 h-1 rounded-full bg-muted`, 채움 `min(len/budget,1)*100%` — 초과 아니면 `bg-success`, 초과면 `bg-destructive`
     - `length_budget === null || is_note`이면 게이지 전체를 숨긴다
   - 본문 — `text-[22px] leading-[1.45] tracking-[-0.01em]`.
     기존 `StyledText`(`style_segments` 색/굵기 + 저대비 보정)를 **그대로 재사용**한다.
     문제가 된 부분 문자열은 `bg-destructive/[0.13] rounded px-1` 로 하이라이트.

4. **추천 수정 카드** — `finding.suggested_fix`가 있을 때만
   - 컨테이너 `rounded-xl border border-primary/40 bg-primary/[0.06] px-4 py-3.5`
   - 라벨 줄 — `추천 수정` `text-[11px] font-bold tracking-[0.04em] text-primary` + 근거 `text-[11px] text-muted-foreground` (예: `용어집 기준`)
   - 제안문 — `text-[22px] leading-[1.45] mb-3.5`, 바뀌는 부분 `bg-success/[0.16] rounded px-1`
   - 액션 줄 — `flex items-center gap-2.5`
     - **`적용하고 다음`** — 높이 38, `px-[18px] rounded-lg bg-primary text-primary-foreground text-sm font-semibold` + `Check` 16px
     - 보조 문구 `text-xs text-muted-foreground` — `반복되는 {n}곳도 함께 바뀝니다` + 키 힌트 `⏎`
       (`repeat_count <= 1`이면 앞 문장 생략하고 키 힌트만)
     - 키 힌트 칩: `font-mono text-[11px] bg-muted rounded px-1.5 py-0.5`

5. **보조 액션 줄** — `flex items-center gap-2 pt-0.5`
   - `직접 고치기` — outline 34px + `Pencil` 14px → 인플레이스 편집 모드
   - `AI에게 다시 맡기기` — outline 34px + `RefreshCw` 14px → 지시문 입력 모드
   - `이대로 두기` — ghost 34px, `text-muted-foreground`, 키 힌트 칩 `S`

6. **구분선** `h-px bg-border my-[22px] mb-4`

7. **같은 슬라이드의 다른 문구** — 맥락 유지용 읽기 전용 대조표
   - 라벨 `text-xs font-semibold text-muted-foreground mb-2.5`
   - 표 `rounded-[10px] border border-border overflow-hidden`, 행 사이 1px `bg-border` 구분
   - 각 행 `grid grid-cols-2`, 셀 `px-[13px] py-2.5 text-[13px]`; 좌(원문) `text-foreground/65`, 우(번역) 기본색
   - 방금 바꾼 항목에는 `방금 수정함` `text-[11px] font-semibold text-success`
   - 행 클릭 시 해당 문구로 이동(큐 밖 항목이면 전체 목록 모드로 전환 후 포커스)
   - 최대 6행까지, 초과분은 `+N개 더` 로 접는다

#### 3-3. 편집 모드 (직접 고치기)

추천 수정 카드 자리를 `Textarea`로 치환한다. **모달 없음.**

- `<Textarea autoFocus className="min-h-[88px] text-[15px]">`, 초기값 = 현재 번역
- 아래 줄: 실시간 길이 게이지 + `이전: <s>기존 번역</s>` (`text-xs text-muted-foreground`, 취소선 `text-foreground/50`)
- 체크박스 `똑같은 문구 {n}곳도 함께 바꾸기` — 기본 체크, `repeat_count > 1`일 때만 노출 (`propagate_identical`)
- 버튼 `취소`(outline) / `적용하고 다음`(primary). `⌘/Ctrl + Enter` = 적용

#### 3-4. AI 재번역 모드

- 한 줄 `Input`: placeholder `추가 요청사항 (선택) · 예: 더 짧게, 더 격식있게`
- `Enter` 또는 `AI 번역` 클릭 → `proposeJobFragment(action:"retranslate")`
- 응답이 오면 그 결과를 **추천 수정 카드와 동일한 형태**로 렌더 (`적용하고 다음` / `다시 시도` / `취소`)
- 대기 중에는 카드 자리에 `Loader2` 스피너 + `AI가 다시 번역하는 중…`

#### 3-5. FinishBar — 하단 마무리 바

`flex items-center gap-3.5 px-7 py-3 border-t border-border bg-card`

- 좌측 안내 `text-[13px] text-muted-foreground` — 상태별 카피:
  - 남은 항목 있음: `남은 {n}건은 지금 저장해도 됩니다 — 파일을 다시 열어 이어서 검토할 수 있어요.`
  - 남은 항목 0: `확인이 필요한 항목을 모두 처리했습니다.`
  - 변경 없음(`!dirty`): `아직 고친 곳이 없습니다. 저장하면 번역 결과가 그대로 저장됩니다.`
- 우측 `ml-auto flex items-center gap-2.5`
  - `남은 항목 모두 넘기기` — outline 38px. 남은 항목이 0이면 숨김. 클릭 시 확인 없이 전부 `ignore` 처리(되돌리기 가능)
  - **`PPT 저장`** — primary 38px `px-[18px] rounded-lg text-sm font-semibold` + `Download` 16px.
    `dirty`면 커밋 후 다운로드, 아니면 바로 다운로드 (문구는 두 경우 모두 `PPT 저장`)

---

### 4. GlossaryPane — 우측 용어집 (w=264)

**Purpose**: 헤더에서 쫓아낸 용어집 기능에 맥락 있는 자리를 준다.

- 컨테이너 `w-[264px] shrink-0 border-l border-border bg-card flex flex-col`
- 헤더 `px-4 pt-4 pb-3.5 border-b border-border`
  - `이 덱의 용어집` `text-[13px] font-bold`
  - 부제 `text-xs text-muted-foreground` — `{활성 용어집 이름} · {entries.length}개`. 활성 용어집이 2개 이상이면 이름을 `·`로 잇는다
- 목록 `flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-2.5`
  - 항목: 원문 `text-[13px] font-semibold`, 번역 `text-[13px] text-muted-foreground`, 항목 사이 `border-b border-border/70 pb-2.5`
  - **현재 큐 항목의 원문에 등장하는 용어를 맨 위로 정렬**하고 `bg-primary/[0.06] -mx-2 px-2 rounded` 강조
- 추가 폼 `px-4 pt-3 pb-4 border-t border-border`
  - 라벨 `고른 문구를 용어집에 추가` `text-xs font-semibold text-muted-foreground mb-2`
  - `Input` 원문 / `Input` 번역 (h-8, `text-[13px]`) / `추가` 버튼(outline h-8)
  - 도움말 `text-[11px] leading-normal text-muted-foreground mt-2` — `다음 번역부터 자동 적용됩니다.`
  - 본문에서 텍스트를 드래그 선택하면 원문 칸이 자동으로 채워진다

---

### 5. 전체 목록 모드 (보조 뷰)

큐는 "문제 있는 것"만 다룬다. 문제 없는 문구도 훑고 싶은 사용자를 위해 SlideRail의
`전체 37개 문구 보기`로 진입하는 목록 모드를 둔다.

- 가운데 영역만 교체. 슬라이드별로 묶인 **2열 대조표**(3-2의 7번 표와 동일 스타일)
- 각 행 hover 시 우측에 `수정` 버튼 노출 → 클릭하면 그 항목을 큐 상세로 연다
- 상단에 `← 검토 큐로 돌아가기` 링크

### 6. 완료 상태

큐를 모두 비우면 가운데 영역을 완료 화면으로 교체:

- `CheckCircle2` 40px `text-success`
- `확인이 필요한 항목을 모두 처리했습니다.` `text-lg font-bold`
- `직접 수정 3곳 · AI 재번역 2곳 · 그대로 둠 7곳` `text-[13px] text-muted-foreground`
- `PPT 저장` primary 버튼 (FinishBar와 동일 동작) + `전체 문구 다시 훑어보기` ghost

### 7. 로딩 / 에러

- 로딩: 기존과 동일하게 중앙 `Loader2` + `섹션을 불러오는 중...`, `text-muted-foreground`
- 에러: `m-4 p-3 rounded-lg border border-destructive/30 bg-destructive/10 text-sm text-destructive` + `AlertTriangle` — 문구 `검토 목록을 불러오지 못했습니다.` + `다시 시도` 버튼

---

## Interactions & Behavior

### 큐 구성 규칙

```ts
// findings가 있는 조각만, severity → slide → index 순으로 정렬
const queue = fragments
  .filter(f => f.findings.length > 0 && !reviewState[f.index])
  .sort(bySeverity /* critical > major > minor */, bySlide, byIndex);
```

- `reviewState`는 `Record<fragmentIndex, "applied" | "skipped">`.
  **Phase 1은 프론트 로컬 상태**(`useState` + `localStorage[`review:${jobId}`]`)로 충분하다. 백엔드 변경 불필요.
- 항목을 처리하면 큐에서 빠지되 **인덱스는 유지**해 `이전`으로 되돌아갈 수 있게 한다.
- `updateJobGlossary` 호출 후에는 findings가 다시 스윕되므로, `load()` 후 큐를 재계산하되
  **현재 보고 있던 항목의 위치를 유지**한다 (index 기준으로 재탐색).

### 액션 → API 매핑 (전부 기존 엔드포인트)

| UI 액션 | 호출 |
|---|---|
| 추천 수정 `적용하고 다음` | `proposeJobFragment(action:"edit", target, propagate_identical)` → 즉시 `applyJobFragmentProposal(proposal_id, revision)` (**비교 모달 건너뜀**) |
| 직접 고치기 `적용하고 다음` | 위와 동일 (`target` = 편집한 텍스트) |
| AI 재번역 | `proposeJobFragment(action:"retranslate", instruction)` → 결과 카드 확인 후 `applyJobFragmentProposal` |
| 이대로 두기 | `editJobFragment(action:"ignore", finding_type)` + 로컬 `reviewState[index]="skipped"` |
| 남은 항목 모두 넘기기 | 남은 index마다 `ignore` 순차 호출 (실패해도 나머지 진행, 마지막에 `load()`) |
| 되돌리기 | `undoReview(jobId, revision)` |
| PPT 저장 | `dirty ? commitReview(jobId, revision) : Promise.resolve()` → `downloadResult()` |

### `추천 수정`의 제안문을 만드는 방법

`FragmentFinding.suggested_fix`는 **교체할 단어**만 담고 있고 완성된 문장이 아니다. 처리 순서:

1. `finding.type`이 `terminology.*`이고 `term_source` + `suggested_fix`가 모두 있으면,
   현재 `target`에서 잘못 쓰인 표현을 찾아 `suggested_fix`로 치환한 문자열을 제안문으로 만든다.
   치환 대상을 특정할 수 없으면 **추천 수정 카드를 렌더하지 않는다**.
2. 그 외 유형(`fit.overflow` 등)은 추천 수정 카드 대신 **`AI에게 다시 맡기기`를 기본 강조 버튼**으로 올린다.
   `fit.overflow`일 때 기본 지시문은 기존 코드와 동일하게 `"더 짧게"`.

> **백엔드 개선 제안 (선택, Phase 2)**: `FragmentFinding`에 `suggested_target: string | null`
> (제안이 반영된 완성 번역문)을 추가하면 프론트의 문자열 치환 추측을 없앨 수 있다.
> 서버가 이미 위반 위치를 알고 있으므로 비용이 낮다.

### 부분 일치 후보 (`partial_candidates`)

기존의 화면 하단 플로팅 시트를 없애고, **다음 큐 항목보다 먼저 삽입되는 큐 카드**로 바꾼다.

- 헤더 배지 `비슷한 문구도 바꿀까요?`
- 본문에 후보 목록(체크박스 + 슬라이드 번호 + 원문 + 현재/제안 번역)
- 액션 `선택한 N건 적용` / `건너뛰기` → `applyPartialCandidates` 또는 그냥 폐기
- 후보가 1건이면 체크박스 없이 바로 `적용` / `건너뛰기`

### 키보드

| 키 | 동작 |
|---|---|
| `Enter` | 추천 수정 적용 후 다음 (추천 카드가 있을 때만) |
| `S` | 이대로 두기 |
| `E` | 직접 고치기 진입 |
| `R` | AI에게 다시 맡기기 진입 |
| `←` / `→` | 이전 / 다음 항목 |
| `⌘/Ctrl + Enter` | 편집 모드에서 적용 |
| `Esc` | 편집/재번역 모드 취소 → 한 번 더 누르면 화면 닫기 |

입력 필드에 포커스가 있을 때는 `S`/`E`/`R`/`←`/`→`를 가로채지 않는다.
하단에 키 힌트를 상시 노출하지 말고, 버튼 옆 칩으로만 보여준다.

### 애니메이션

- 항목 전환: `opacity 0 → 1` + `translateY(6px) → 0`, **140ms `ease-out`**. 그 이상은 큐 처리 속도를 방해한다.
- 진행 바: `width` `transition-all duration-500 ease-out` (기존 `ProgressPanel`과 동일)
- 적용 성공 시 toast는 **띄우지 않는다** (진행 카운터가 이미 피드백). 실패할 때만 `toast.error`.

### 낙관적 잠금 / 에러

- 모든 mutation은 `expected_revision`을 보낸다. 409가 오면 `load()` 후
  `toast.error("다른 변경이 먼저 반영됐습니다. 목록을 새로 불러왔어요.")` 하고 같은 항목에 머문다.
- 액션 진행 중에는 해당 버튼만 비활성 + `Loader2`. 전체 화면 블로킹 금지.

---

## State Management

`ReviewPanel` 로컬 상태로 충분하다 (zustand 스토어 신설 불필요).

```ts
fragments: FragmentItem[]          // 서버 원본
revision, committedRevision, dirty // 기존과 동일
reviewState: Record<number, "applied" | "skipped">   // localStorage[`review:${jobId}`]
cursor: number                     // 큐 내 현재 위치 (0-based)
mode: "queue" | "list" | "done"
editorMode: "none" | "manual" | "ai"
editText: string
instruction: string
propagate: boolean                 // 기본 true
proposal: FragmentProposalResponse | null   // AI 재번역 결과 대기용
partialCandidates: PartialCandidate[]
busy: boolean
```

파생값은 `useMemo`:
`queue`, `currentFragment`, `slideRemaining: Map<slide, count>`, `resolvedCount`, `totalCount`.

데이터 페칭은 기존 `apiClient.getJobFragments(jobId)` 그대로. 폴링 없음.

---

## Design Tokens

전부 `frontend/src/app/globals.css`에 이미 있다. **새 토큰을 추가하지 않는다.**

| 용도 | 토큰 / 클래스 | light 값 |
|---|---|---|
| 배경 | `--background` / `bg-background` | `oklch(0.985 0.005 80)` |
| 패널·카드 | `--card` / `bg-card` | `oklch(0.995 0.003 80)` |
| 본문 텍스트 | `--foreground` | `oklch(0.205 0.015 60)` |
| 보조 텍스트 | `--muted-foreground` | `oklch(0.45 0.012 60)` |
| 약한 면 | `--muted` | `oklch(0.94 0.008 80)` |
| 테두리 | `--border` | `oklch(0.88 0.01 80)` |
| 강조 (유일) | `--primary` | `oklch(0.55 0.15 270)` |
| 성공 | `--success` | `oklch(0.723 0.191 142.5)` |
| 경고 | `--warning` | `oklch(0.769 0.188 70.08)` |
| 오류 | `--destructive` | `oklch(0.577 0.245 27.325)` |
| 정보 | `--info` | `oklch(0.623 0.214 259.815)` |

다크 모드 값도 `globals.css`의 `.dark` 블록에 이미 정의되어 있으므로 클래스만 쓰면 자동 대응된다.
**프로토타입의 `oklch(...)` 리터럴을 코드에 옮기지 말 것.**

### 반경 / 그림자

- `--radius: 0.75rem` 기준. 사용하는 값: `rounded-md`(10px, 버튼/입력), `rounded-lg`(12px, 주요 CTA·카드),
  `rounded-xl`(16px, 추천 수정 카드), `rounded-full`(배지·진행 바)
- 그림자는 이 화면에서 **쓰지 않는다**. 기존 `glass-card`(blur + shadow)는 검토 화면에서 제거한다 —
  타일이 사라지면서 필요가 없어졌고, 텍스트 가독성을 떨어뜨린다.

### 타이포 (Geist Sans, `--font-geist-sans`)

| 역할 | 크기 / 굵기 / 기타 |
|---|---|
| 큐 원문·번역·제안문 | 22px / 400 / `leading-[1.45]` / `tracking-[-0.01em]` |
| 진행 카운터 숫자 | 30px / 700 / `tracking-[-0.03em]` / `leading-none` |
| 완료 화면 제목 | 18px / 700 |
| 본문·버튼 | 14px / 500~600 |
| 목록·설명·부제 | 13px / 400~600 |
| 라벨·배지·게이지 | 12px / 600 |
| 미세 라벨·키 힌트 | 11px / 600, `tracking-[0.04em]` (섹션 라벨) |

키 힌트 칩만 `--font-geist-mono`.

### 간격

8px 스케일 기준. 이 화면에서 반복되는 값:
가운데 본문 좌우 패딩 `28px`, 패널 좌우 패딩 `16px`, 블록 사이 `14px`, 라벨-본문 `7px`,
버튼 높이 `38px`(주요) / `34px`(보조) / `32px`(헤더) / `30px`(페이저).

---

## Assets

새 에셋 없음. 아이콘은 전부 기존 `lucide-react`:
`CheckCircle2, Check, ChevronRight, ChevronLeft, X, Undo2, Download, Pencil, RefreshCw, Loader2, AlertTriangle, Plus`.

---

## Files

| 파일 | 내용 |
|---|---|
| `Review Redesign.dc.html` | 세 가지 안이 담긴 디자인 캔버스. **`1c` 섹션이 이 핸드오프의 대상**이다. `1a`는 현재 화면 재현(비교용), `1b`는 채택하지 않은 보수안. |
| `support.js` | 위 HTML을 브라우저에서 그대로 열기 위한 런타임. 함께 두면 더블클릭으로 열린다. |

### 손대야 하는 코드베이스 파일

- `frontend/src/components/translation/ReviewPanel.tsx` — 거의 전면 재작성. `StyledText`,
  `reviewColorContrast`, `styleStatusLabel`, `badgeStyle`은 **그대로 살려서 재사용**한다.
- `frontend/src/components/translation/TranslationForm.tsx` — 검토 오버레이 진입/종료 부분만 확인 (변경 최소)
- `frontend/src/app/globals.css` — `.review-grid`, `.review-span-2`, `.review-span-full` 규칙 **삭제**
  (`.review-style-color` 관련 규칙은 유지)
- `frontend/src/lib/api-client.ts`, `frontend/src/types/api.ts` — Phase 1에서는 변경 없음

### 구현 순서 제안

1. 큐 계산 + 좌/중/우 3분할 셸 + StepHeader (정적)
2. QueueItem 렌더 + 이전/다음 이동 + `이대로 두기`
3. `추천 수정` → propose+apply 원클릭 경로
4. 직접 고치기 / AI 재번역 인플레이스 모드
5. `partial_candidates` 큐 카드
6. FinishBar + 완료 화면 + 저장 동선
7. GlossaryPane
8. 전체 목록 모드
9. 키보드 단축키

1~3만 끝나도 기존 대비 체감이 크다. 각 단계마다 `npm test && npx tsc --noEmit && npm run build`.

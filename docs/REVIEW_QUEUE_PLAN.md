# 번역 검토 화면 재설계 (검토 큐) — 구현 계획

대상 디자인: `design_handoff_review_queue/` 의 **1c 과감안** — "37개 타일 그리드"를
"확인이 필요한 항목만 한 건씩 처리하는 큐"로 재설계.

이 문서는 핸드오프 문서(`design_handoff_review_queue/README.md`)를 **실제 코드로 검증한 뒤**
확정한 실행 계획이다. 핸드오프와 어긋나는 부분은 아래 "핸드오프 문서 정정"에 근거와 함께 적었다.

---

## 1. 확정된 결정

| # | 결정 | 근거 |
|---|---|---|
| D-1 | 추천 수정은 **보수적 휴리스틱** | 서버·클라이언트 모두 "잘못 쓰인 표현"의 위치를 모름 (§2.1) |
| D-2 | 레이아웃은 **반응형 축소** (`xl` 미만 용어집 접기, `lg` 미만 레일 접기) | 앱 기본 창 1100×800 / 최소 800 (`src-tauri/tauri.conf.json`), 디자인은 1280 기준 |
| D-3 | 헤더 결과 파일명은 **백엔드 필드 추가** (`FragmentsResponse.output_filename`) | `job.output_filename`이 완료 시점에 이미 존재 (`api.py:707-718`) |
| D-4 | 큐 항목 = **블록**(문장), 단 보수적 병합 (§3.2) | 한 문장이 여러 문단으로 쪼개져 검토가 불편하다는 실사용 문제 |
| D-5 | 표 셀 식별자 충돌도 **함께 수정** | 블록 그룹핑의 전제조건 + 기존 잠재 버그 |
| D-6 | 무시(`이대로 두기`)는 **되돌릴 수 있게** 만든다 (확인 다이얼로그 아님) | 큐에서 `S` 한 키로 일어나는 비가역 동작 (§3.3) |
| D-7 | **백엔드 먼저**, 측정 의존 여부로 2단계 분할 | 프론트 계약 동결 + 테스트 스위트가 백엔드에만 존재 |

### D-1 상세 — 추천 수정 판정 규칙

`FragmentFinding.suggested_fix`는 *교체할 단어*가 아니라 *있어야 하는 단어*다.
`consistency_sweep.py:198-214`는 "번역에 `Battle Pass`가 **없다**"만 알려주고,
`Combat Pass`라는 잘못된 표현의 위치는 **서버도 모른다**.

| 조건 | 판정 | 카드 |
|---|---|---|
| `term_source`가 target에 그대로 남아 있음 (미번역) | 확정 | 그 스팬 → `suggested_fix` |
| `suggested_fix`의 대소문자/공백 변형이 target에 있음 (확정 용어) | 확정 | 변형 스팬 → 정확한 표기 |
| target의 연속 어절 스팬이 `suggested_fix`와 문자 bigram 유사도 ≥ 임계값 + 공통 토큰 | 추정 | 근거 라벨 `용어집 기준 (추정)` |
| 그 외 / `fit.*` / `accuracy.omission` / `consistency.phrase` | 없음 | `AI에게 다시 맡기기`를 primary로 승격 |

제안문 전체가 22px로 표시되고 사용자가 `⏎` 전에 눈으로 확인하므로, 오추측 비용은 낮다.
`fit.overflow`의 기본 지시문은 기존과 동일하게 `"더 짧게"`.

> 참고: `terminology.inconsistency`는 스윕이 생성하지 않는 죽은 분기다 (`badgeStyle` 매핑만 존재).

---

## 2. 검증에서 확인한 코드 사실

### 2.1 검토 단위 = PPT 문단, 병합 로직 없음

`PPTParser.extract_paragraphs`가 문단마다 `ParagraphInfo`를 만들고
(`src/core/ppt_parser.py:173-188`), `ReviewSession.fragments()`가 그대로 나열한다
(`src/services/review_session.py:270-309`). 중간에 묶는 로직이 없다.

프론트에서 `(slide, shape)`로 묶을 수 없는 이유 — 식별자가 두 군데서 뭉개진다:

- **그룹 도형**: 자식이 전부 부모의 `shape_index`를 물려받는다 (`ppt_parser.py:104-112`)
- **표**: 모든 셀이 표 도형의 `shape_index`를 공유하고 `paragraph_index`는 셀마다 0부터 재시작
  (`_extract_from_table` → `_collect_paragraphs_from_text_frame:173`).
  즉 `(slide, shape, paragraph)`가 **셀 간에 충돌**한다 — `consistency_sweep.py:95-98` 주석이 같은 충돌을 언급.

### 2.2 무시는 서버 영속이되 **비가역**

`dismissed_findings`는 add 전용이다 (`review_session.py:785`). 제거 경로가 코드 어디에도 없다.
`undo()`는 텍스트 스냅샷만 복구한다 (`:767-782`). dismiss는 revision을 올리지 않으므로
`dirty=false` → **`되돌리기` 버튼 자체가 비활성**.

### 2.3 세션 수명 1시간

`_cleanup_old_jobs`가 `completed_at` 기준 1시간 지난 완료 작업을 삭제한다
(`src/services/job_manager.py:293-311`, 300초 주기, `api.py:64`에서 기동).
검토 중인지 여부와 무관. 삭제되면 초안·무시 기록이 증발하고 모든 호출이 404/400.

### 2.4 적용 1회당 서버 비용 (실측)

`run_sweep` (용어집 200개 기준, `python3` 직접 호출):

| 조각 수 | 시간 |
|---|---|
| 300 | 231 ms |
| 600 | 461 ms |

`api.py:1289`가 적용마다 `run_final_sweep()`을 돌리고, 이어지는 `GET /fragments`가
조각마다 `style_preview` + `length_budget`을 다시 계산한다. 600조각 덱에서 `적용하고 다음`
1회에 1~2초로 추정 → **낙관적 커서 전진**이 필요하다.

### 2.5 블록 단위 편집을 원자적으로 적용할 경로가 없다

`apply_edit(index, new_target, ...)`은 인덱스 1개 + 텍스트 1개 (`review_session.py:367`).
`create_proposal`도 단일 인덱스. `applyPartialCandidates`는 여러 인덱스를 받지만 *구절 치환*만 한다.
블록 3문단을 고치면 revision 3회 증가 · undo 스냅샷 3개 · 부분 실패 시 반쯤 적용된 상태.

### 2.6 `length_budget`이 문단마다 도형 전체 용량을 쓴다

`paragraph._parent`(텍스트 프레임) → `_parent`(도형)의 전체 width/height로 계산한다
(`review_session.py:318-355`). 5문단짜리 박스면 각 문단이 박스 전체 용량을 배정받아
overflow가 약 5배 과소 검출된다. 표 셀은 `owner.width`가 None이라 `max(source_len, 8)`로 폴백.

### 2.7 줄바꿈(`<a:br/>`) 손실

`"".join(run.text for run in paragraph.runs)` — python-pptx 0.6.23의 `paragraph.runs`는
`<a:r>`만 돌려주고 `<a:br/>`를 건너뛴다. Shift+Enter로 나눈 두 줄이 **공백 없이 붙는다**.
`paragraph.text`는 `\v`로 보존하지만 `<a:fld>` 텍스트까지 포함해 쓰기 경로(runs 기반)와 어긋난다.
→ **이번 범위 밖. 규모만 측정한다.**

### 2.8 기타

- `apply_proposal`은 `expected_revision == revision` **AND** `proposal.base_revision == revision`을
  둘 다 본다 (`review_session.py:712`) → 409 후 기존 proposal은 재사용 불가, 재-propose 필요
- `create_proposal`의 색상 매핑은 **다색 문단일 때만** LLM을 탄다 (`translation_service.py:497-513`)
- `ppt_writer.apply_translations`는 문단↔번역 1:1로 `runs`에 쓴다 (`ppt_writer.py:708-727`)
  → 번역 단위 병합은 서식·색상 매핑을 깨뜨린다
- 무시 1건마다 `_record_edit`가 JSONL을 파일 열고 append (`quality_records.py:74-81`, 설계상 append-only `:4`)
- 프론트엔드에 React 테스트 러너가 없다 (vitest만, RTL 없음) → 순수 함수만 테스트 가능

---

## 3. 핸드오프 문서 정정

### 3.1 `localStorage[review:${jobId}]` — 불필요

무시는 서버 영속이고(§2.2), `jobId`는 zustand에 persist가 없어 새로고침하면 사라진다.
로컬 상태는 `useState`로 충분.

### 3.2 "블록 = 텍스트 프레임"은 불릿 목록에서 틀리다

불릿 8개짜리 본문 상자도 텍스트 프레임 하나다. 한 항목으로 묶으면 "한 화면에 판단할 것이 하나"가
깨지고 22px 8줄이면 추천 카드가 화면 밖으로 밀린다. **보수적 연속 병합**으로 좁힌다:

- 같은 `container_id`의 **연속** 문단 (`paragraph_index`가 이어질 것 — 사이에 빈 문단이 있으면 구분자로 읽는다)
- `container_kind`가 `body`(불릿 목록) / `notes`가 **아니고**
- 앞 문단이 문장 종결부호로 끝나지 **않고**
- **뒷 문단이 소문자로 시작하거나, 앞 문단이 쉼표로 끝나고** (Step 2 실측으로 추가 — 아래)
- 병합 상한 4문단

> **불릿 마커로는 판별할 수 없다** (실측): 불릿 본문 플레이스홀더와 일반 텍스트 상자 **둘 다
> 문단 `pPr`가 비어 있다** — 불릿은 레이아웃/마스터의 list style에서 상속되기 때문이다.
> 대신 `shape.is_placeholder` + `placeholder_format.type`이 확정적 신호다:
> 불릿 목록은 `body`, 줄바꿈된 문장은 `textbox`로 깨끗하게 갈린다.

> **실측이 전제를 뒤집었다** (Step 2, 실제 덱 5개 / 문단 1,694개 / 병합 후보 404쌍.
> 아래 수치는 그중 업무 덱 4개 = 후보 377쌍): "텍스트 상자의 연속 문단 =
> 손으로 줄바꿈한 한 문장"은 **거의 언제나 틀렸다**. 병합 후보 377쌍 중 진짜로 이어지는 문장은
> **7쌍(2%)**뿐이고, 나머지 370쌍은 텍스트 상자 안의 *제목+설명* (`UIUX` ↳ `Match the military…`),
> *라벨-값* (`Squad Size` ↳ `4 players`), *유사 불릿*이었다. 불릿은 `body` 플레이스홀더에만 있지
> 않다 — 디자인 덱은 일반 텍스트 상자로 목록을 만든다.
>
> 깨끗하게 갈린 유일한 신호는 **뒷 문단의 첫 글자가 소문자인가**다. 진짜 7쌍을 모두 잡고
> 370쌍을 모두 걸렀다.
>
> **한국어 덱**(테스트 덱 `텍스트 색상 테스트.pptx`, 산문형)에서는 후보 27쌍 중 25쌍이
> **개조식 불릿**(`~함` / `~됨` / 명사 종결)이라 거부가 옳았고, 진짜 줄바꿈은 2쌍뿐이었다.
> 그중 1쌍은 **앞 문단이 쉼표로 끝난다**(`...발전속도가 매우 빠르니,`) — 문자 종류와 무관하게
> 안전하므로 규칙에 넣었다. 5개 덱 재측정에서 이 쉼표 경로가 추가한 병합은 그 1건뿐이다.
> 나머지 1쌍은 연결어미(`...하기 위해`)라 한국어 형태소 목록이 필요해 **넣지 않았다** —
> 개조식 종결과 구분하려면 열린 목록이 되고, 1건을 위해 오병합 위험을 살 이유가 없다.
> 결과적으로 대소문자 없는 문자는 쉼표 줄바꿈만 병합된다. 안전한 실패 방향(문단 1개 =
> 항목 1개, 오늘 동작)으로 떨어진다.

그 외에는 항목을 나누되 블록 전체를 맥락으로 표시.

**이 규칙은 프론트의 순수 함수로 둔다.** 서버는 블록이 무엇인지 알 필요가 없다 — `container_id` /
`container_kind`(구조적·확정적)와 일괄 적용 엔드포인트(`{index: text}`)만 제공하면 된다.
규칙을 TS에 두면 vitest로 덮고 임계값도 파이썬 왕복 없이 조정할 수 있다.

### 3.3 "남은 항목 모두 넘기기 → 확인 없이 전부 ignore (되돌리기 가능)" — 되돌릴 수 없다

§2.2 참조. 수십 건을 비가역으로 날리는 버튼이 된다.

**대응**: `ReviewSession.restore_finding()`(`dismissed_findings.discard`) + 엔드포인트 추가.
확인 다이얼로그는 넣지 않는다 — 개별 `이대로 두기`(`S` 한 키)에는 붙일 수 없어
정작 흔한 사고를 못 막는다. undo가 생기면 핸드오프의 "확인 없이"가 그대로 성립한다.

`되돌리기` 버튼이 하나뿐인 문제는 **클라이언트의 순서 있는 액션 로그**로 푼다.
서버의 두 스택이 독립적이고(dismiss는 `_history`를 건드리지 않음) 단일 사용자·단일 창이라
클라이언트가 순서의 유일한 권위자다:

```
[{kind:"edit", revision:6}, {kind:"dismiss", entries:[...]}, {kind:"edit", revision:5}]
되돌리기 → 맨 위 pop → edit이면 undoReview, dismiss면 restore
```

대량 넘기기는 **벌크 dismiss 1회** → 로그 항목 1개 → 되돌리기 한 번에 전부 복구.
(성능 근거: 40건 넘기기 = HTTP 52회 + 파일 열기 52회)

한계: 품질 원장의 `rejected` 기록은 지워지지 않는다 (append-only 설계). 원장 보정은 별건.

### 3.4 `되돌리기` 활성 조건

`disabled={!dirty}` → `disabled={!dirty && 무시한 항목 없음}`

---

## 4. 범위 밖 (별건)

| 항목 | 이유 |
|---|---|
| **번역 단위 병합** | `ppt_writer`가 문단↔번역 1:1 (§2.8). 합쳐 번역하면 색상·서식 매핑이 깨진다 |
| **`<a:br/>` 줄바꿈 손실** | §2.7. 쓰기 경로와 얽혀 간단히 못 바꾼다. Step 2에서 규모만 측정 |
| **`length_budget` 근본 수정** | §2.6. 고치면 `fit.overflow` 검출량이 바뀐다. 이번엔 UI에서만 회피 |
| **RTL 도입** | 범위 확대. 대신 큐 상태 전이를 순수 리듀서로 분리해 vitest로 덮는다 |

---

## 5. 실행 순서

각 단계 끝: 백엔드 `pytest tests/ -v`, 프론트 `npx tsc --noEmit && npm test && npm run build`

### Step 1 — 백엔드 소품 ✅ 완료

오늘 존재하는 결함들. 검토 큐를 안 만들어도 고쳐야 한다.

**1. 무시 되돌리기 + 벌크 dismiss** (§3.3)
- `ReviewSession.restore_finding()` 추가, `dismiss_finding()`은 **실제로 바뀌었을 때만 True** 반환
  → 클라이언트가 되돌릴 대상을 정확히 안다
- `POST /api/v1/jobs/{id}/review/dismissals` — `{action: "dismiss"|"restore", entries: [{index, finding_type}]}`
  → `{changed, revision, committed_revision, dirty}`. `changed`는 **실제로 바뀐 항목만** 담는다
- **`review_lock`도 `expected_revision`도 쓰지 않는다**: 집합 add/discard뿐이라 초안·revision을
  건드리지 않고 교환법칙이 성립한다. 락을 잡으면 `S` 키 한 번이 진행 중인 AI 재번역 뒤에서 멈춘다
- 원장 기록은 executor로 오프로드. **restore는 원장에 아무것도 쓰지 않는다** (append-only 설계)
- `entries`는 1~2000개로 제한

**2. 활동 기반 TTL** (§2.3)
- `Job.last_activity_at` + `Job.touch()`, `JobManager.get_job()`이 조회 시 touch
- `_cleanup_old_jobs`가 `last_activity_at or completed_at` 기준으로 노후 판정
- 검토 요청은 전부 `get_job`을 거치므로 **호출 지점을 8곳 고칠 필요가 없다**.
  정책이 "완료 후 1시간" → "**접근 없이 1시간**"으로 바뀐다 (엄격히 더 안전)
- 완료 후 프론트 폴링은 멈추므로(`lib/sse-client.ts`) 무한 보존 위험 없음

**3. `FragmentsResponse.output_filename`** (D-3) — `api.py` + `frontend/src/types/api.ts`
+ `apiClient.updateReviewDismissals()`

**검증**: `pytest tests/ -q` 329 passed · `npx tsc --noEmit` 통과 · `npm test` 12 passed · `npm run build` 성공
신규 테스트 11건 (`test_review_session.py::DismissRestoreTestCase` 4,
`test_api.py::TestReviewEndpoints` 4, `test_job_manager.py::TestCleanupAgesFromLastActivity` 3)

### Step 3 — 백엔드 본체 ✅ 완료 (프론트 계약 동결)

**1. 컨테이너 식별자** (`ppt_parser.py`) — `ParagraphInfo.container_id` / `container_kind`
- 경로: `s{slide}/sh{shape}` + 그룹 중첩 `/g{n}` + 표 셀 `/r{n}c{n}`, 노트는 `s{slide}/notes`
- **표 셀 충돌 해소** (D-5): 4개 셀이 `(shape, paragraph)`를 공유하지만 `container_id`는 전부 다르다.
  그룹 자식도 마찬가지 (부모 `shape_index`를 공유하지만 경로가 다르다)
- `container_kind`: `title` / `body` / `textbox` / `placeholder` / `table_cell` / `notes`
- `FragmentView` → `FragmentItem` → `frontend/src/types/api.ts`까지 노출

**2. 블록 일괄 적용** (§2.5) — `ReviewSession.apply_block_edit(edits, expected_revision, ...)`
+ `POST /api/v1/jobs/{id}/review/block`
- 스냅샷 1개 + revision 1회 → `되돌리기` 한 번에 문장 전체 복구
- 색상 매핑을 **변경 전에** 계산해 실패해도 초안이 그대로 남는다
- 범위 밖 인덱스는 400, revision 불일치는 409 — 둘 다 초안을 건드리기 전에 거부
- `apiClient.applyReviewBlockEdit()`

**3. 진단 스크립트를 프로덕션 파서로 전환** — `PPTParser`를 직접 쓰고 §3.2 병합 규칙을 시뮬레이션.
병합 거부 사유(`container` / `gap` / `kind` / `cap` / `sentence_end`)를 집계해 임계값 튜닝 근거를 준다.

**검증**: `pytest tests/ -q` 343 passed · `npx tsc --noEmit` 통과 · `npm test` 12 passed · `npm run build` 성공
신규 테스트 14건 (`ContainerIdentityTestCase` 5, `BlockEditTestCase` 5, `TestReviewEndpoints` 4)

### Step 2 — 측정 ✅ 완료 (실제 덱 5개)

```bash
python3 scripts/analyze_fragments.py <덱.pptx> --notes --list-blocks
```

| 덱 | 문단 | 병합 전 규칙 | 소문자 규칙 적용 후 |
|---|---|---|---|
| Moria Design Overview (EN) | 623 | 336항목 (46% 감소) — **거의 전부 오병합** | 623항목 (0%) |
| DedNet Exec Status (EN) | 647 | 576항목 (11%) | 640항목 (1.1%, 블록 4개 = 진짜 줄바꿈) |
| KR_Part1 DedNet (KR) | 175 | 156항목 | 175항목 (0%) |
| PBB AllHands (KR) | 141 | 141항목 (`body` 87문단) | 141항목 (0%) |
| 텍스트 색상 테스트 (KR, 산문) | 42 | 42항목 | 41항목 (블록 1개 = 쉼표 줄바꿈) |

- **병합 규칙**: §3.2 참조. `NON_MERGING_KINDS = {body, notes}` · `MAX_MERGE_PARAGRAPHS = 4`
  (실측된 가장 긴 진짜 줄바꿈 사슬이 정확히 4문단) · `SENTENCE_END` 유지 · **소문자 연속 조건 추가**
- **짧은 문단이 지배적**: 20자 미만이 38~63%. 큐 카드는 짧은 조각을 전제로 설계해도 된다
  (22px 기본 크기가 대부분에서 성립)
- **`<a:br/>` 손실은 미미**: 4개 덱 합쳐 문단 12개 / 13곳. §2.7을 범위 밖으로 둔 판단이 맞았다
- **표 셀 비중이 큼**: 덱당 34~155문단. D-5 식별자 수정이 없었으면 큐 항목이 서로 덮어썼다

### Step 4 — 프론트엔드

동결된 백엔드 계약 (Step 1·3에서 확정, 프론트가 쓸 것 전부):

| 용도 | 계약 |
|---|---|
| 블록 그룹핑 | `FragmentItem.container_id` / `container_kind` |
| 헤더 파일명 | `FragmentsResponse.output_filename` |
| 블록 편집 | `apiClient.applyReviewBlockEdit(jobId, {index: text}, revision)` |
| 무시 / 되돌리기 / 모두 넘기기 | `apiClient.updateReviewDismissals(jobId, "dismiss"\|"restore", entries)` |

| Phase | 내용 |
|---|---|
| 0 ✅ | 순수 헬퍼(`lib/review-queue.ts`: **병합 규칙** + 큐 정렬 + 제안문) + **큐 리듀서** + vitest, `StyledText` 모듈 분리 (동작 무변경) |
| 1 ✅ | 3분할 셸 + StepHeader + SlideRail + **블록 큐** + 페이저 + `이대로 두기` + `refresh()` 분리 |
| 2 ✅ | 추천 수정 원클릭 (propose→apply, 409 재-propose, **낙관적 커서 전진**) + 서식 미리보기 라벨 정정 |
| 3 | 직접 고치기 / AI 재번역 인플레이스 (블록 일괄 적용 API, 다문단은 게이지 1개) |
| 4 | `partial_candidates` 큐 카드 (플로팅 시트 제거) |
| 5 | FinishBar + 완료 화면 + `PPT 저장` + `남은 항목 모두 넘기기`(벌크) |
| 6 | GlossaryPane (헤더 폼 이관, 롤백 로직 유지) |
| 7 | 전체 목록 모드 (블록 단위 2열 대조표) |
| 8 | 키보드 단축키 + 140ms 전환 + `.review-grid` CSS 삭제 |

**Phase 0 검증**: `npx tsc --noEmit` 통과 · `npm test` 36 passed (신규 24) · `npm run build` 성공 ·
`pytest tests/ -q` 343 passed. 병합 규칙은 Step 2 실측 반영본이며 `scripts/analyze_fragments.py`와
동일하게 유지한다 (한쪽만 고치지 말 것). 큐 순서는 리듀서가 처음 본 순서로 **동결**한다
(처리한 항목이 검출을 잃어도 자리를 지켜 `이전`이 성립).

**Phase 1 검증**: `npx tsc --noEmit` 통과 · `npm test` 37 passed · `npm run build` 성공 ·
`pytest tests/ -q` 344 passed. 신규 컴포넌트 `review/{StepHeader,SlideRail,QueueItem,GlossaryPane,
FinishBar,finding-labels}`, `ReviewPanel`은 컨테이너로 축소, `.review-grid`/`.review-span-*` 삭제.

Phase 1에서 **의도적으로 계획을 벗어난 것** (기능 공백을 만들지 않기 위해):
- 편집(`직접 고치기` / `AI에게 다시 맡기기`)과 `partial_candidates` 시트를 **기존 propose→모달→apply
  경로 그대로** 큐 카드 안으로 옮겼다. Phase 2·3·4가 각각 원클릭·인플레이스·큐 카드로 대체한다.
  이대로 두지 않으면 Phase 3까지 검토 화면에서 번역을 고칠 수 없다
- `FinishBar`(저장)와 `GlossaryPane`(빠른 추가 폼)을 최소 형태로 먼저 넣었다.
  `남은 항목 모두 넘기기`는 Phase 5, 용어 목록·선택 자동 채움은 Phase 6
- SlideRail의 `전체 N개 문구 보기`는 Phase 7(전체 목록 모드)과 함께 넣는다
- **버그 수정**: `style.mapping_dropped`는 무시 필터 뒤에서 다시 붙어 `이대로 두기`가 먹지 않았다
  (`review_session.fragments()`). Phase 1이 그 버튼을 만드는 단계라 함께 고치고 테스트를 추가했다

**Phase 2 검증**: `npx tsc --noEmit` 통과 · `npm test` 38 passed · `npm run build` 성공.
낙관적 전진의 실패 복구는 `queueReducer`의 `rollback`이 맡는다 — 커서가 이미 넘어간 뒤라
스택 맨 위가 아닐 수 있어 **위치가 아니라 액션 식별자로** 찾아서 뺀다.

Phase 2에서 **넣지 않은 것**:
- 현재 번역에서 문제 구간을 `bg-destructive`로 하이라이트: 번역문은 `StyledText`의 색상
  세그먼트로 쪼개져 렌더되므로 하이라이트 범위를 겹치려면 세그먼트를 잘라 재조립해야 한다.
  추천 카드가 바뀌는 부분을 보여주므로 비용 대비 이득이 없다
- `⏎` 키 힌트 칩: 단축키는 Phase 8. 동작하지 않는 키를 광고하지 않는다

**서식 미리보기 라벨** (Phase 2에서 정정, 실사용 피드백): `색상 미리보기`라는 고정 라벨이
색이 없는 문단에서 "색이 안 나온다"로 읽힌다. 실제로는 `style_status`에 따라 보여주는 것이 다르다 —
`preserved`/`partial`이면 원문 색이 그대로 보이지만, `dropped`는 **전체가 첫 서식 그룹으로 덮인
결과**(예: 전부 굵게)를 보여준다. 라벨을 상태에 맞춰 갈라 쓴다. 미리보기 자체는 유지한다:
색 강조를 쓰는 덱에서 번역 후 강조가 엉뚱한 단어에 붙었는지가 실제 검토 포인트다.

**긴 문단 대응** (목업은 짧은 제목 조각 기준): `≤60자 22px / ≤160자 18px / 그 이상 15px`,
노트는 항상 15px + `max-h` 클램프. **추천 카드가 항상 첫 화면 안에** 들어오는 것이 우선.

---

## 6. 파일 지도

**손대는 곳**
- `src/core/ppt_parser.py` — 컨테이너 경로
- `src/services/review_session.py` — `restore_finding`, `apply_block_edit`, 블록 노출
- `src/services/job_manager.py` — 활동 기반 TTL
- `api.py` — 엔드포인트 3종 + `output_filename`
- `frontend/src/components/translation/ReviewPanel.tsx` — 컨테이너로 축소
- `frontend/src/components/translation/review/*` — 신규 컴포넌트
- `frontend/src/lib/review-queue.ts` + `.test.ts` — 신규
- `frontend/src/app/globals.css` — `.review-grid` / `.review-span-*` 삭제
  (`.review-style-color`는 **유지**)

**살려서 재사용** — `StyledText`, `reviewColorContrast`, `styleStatusLabel`, `badgeStyle`
(`badgeStyle`은 라벨만 평서문으로 교체)

**도구** — `scripts/analyze_fragments.py` (조각 분할 진단)

# ppt-translator 품질 파이프라인 적용 설계서

- 작성일: 2026-07-03
- 독자: `ppt-translator` 리포에서 구현을 수행하는 Claude Code (Opus)
- 정본 관계: 전체 전략·철학의 정본은 `oddeyes-quality-pipeline-design.md`(OddEyes 중심 설계서)다. 이 문서는 그중 ppt-translator에 적용되는 부분을 **독립 실행 가능하게** 발췌·구체화한 적용판이다. 공유 데이터 계약(4장)은 편의상 이 문서에 복제돼 있으며, 계약을 변경할 일이 생기면 반드시 정본 문서와 함께 개정한다.
- 코드 앵커(파일명·모듈명)는 2026-07-03 기준 참고 정보다. 구현 시점의 실제 코드가 우선하며, 어긋나면 "의도"를 기준으로 해당 지점을 다시 찾는다.

---

## 0. 배경: 이 리포는 이미 절반을 갖췄다

번역팀은 AI 번역 품질을 높이기 위한 공통 파이프라인·데이터 체계를 도입하고 있다. 핵심은 세 가지다: 번역 품질 문제(번역투, 원문 오류 전파, 일관성 붕괴)를 전담 단계로 잡는 파이프라인, 모든 지적·수정을 기록하는 품질 장부(오답노트), 그리고 기록에서 승격된 팀 규칙이 다음 번역에 자동 주입되는 순환.

ppt-translator는 OddEyes에 통합하지 않고 **별도 앱으로 유지하면서 데이터 계약만 공유하는 세 번째 클라이언트**로 편입된다 (레코드의 `client` 필드로 구분).

이 리포에는 파이프라인의 상당 부분이 이미 있다. 새로 짓기 전에 아래 기존 자산과의 대응을 확인할 것:

| 기존 자산 (코드 앵커) | 파이프라인에서의 의미 |
|---|---|
| `src/chains/summarization_chain.py` (덱 300자 요약) | 경량 원문 사전 분석(S0)의 원형. 확장 대상 |
| `src/chains/context_manager.py` (전역 컨텍스트) | 덱 단위 일관성 컨텍스트. 유지 |
| `src/utils/repetition.py` (반복 문단 1회 번역 + 캐시) | 완전 동일 텍스트의 일관성은 이미 구조적으로 보장됨 |
| `src/utils/glossary_loader.py` (용어집 지원) | 용어 주입 경로 존재. 위반 "검사"는 아직 없음 |
| text fit 검사 (`docs/KEY_PATTERNS.md`) | 결정론 QA 층의 선례. 같은 층에 검사를 추가하면 됨 |

따라서 이 문서의 작업은 "파이프라인 신설"이 아니라 **네 가지 보강**이다: 팀 규칙 주입(WP-C1), 덱 일관성 스윕(WP-C3), 품질 기록(WP-C2), 장문 텍스트의 폴리싱·검증(WP-C4, 선택).

---

## 1. 적용되는 설계 철학

구현 중 판단이 필요하면 아래 원칙을 기준으로 한다.

- **P1. 기계적인 것은 코드로, 판단은 바깥으로.** 용어집 위반·표기 불일치·미번역 검출은 코드가 100% 재현율로 잡는다. LLM은 의미·뉘앙스에만 쓴다. **자동 치환은 절대 하지 않는다** (한국어 형태소 변형 오탐 위험. 검출·보고까지만).
- **P2. 모든 검토 산출물은 기록된다.** 수집은 별도 작업이 아니라 파이프라인의 부산물이다. 검출·판정·반려가 품질 레코드로 남는다.
- **P3. 어떤 모델도 자기 출력의 유일한 검증자가 되지 않는다.** WP-C4에서 폴리싱 모델과 검증 모델은 다르게 배치한다.
- **P6. No Auto-Apply.** 검출 결과는 사용자에게 목록으로 제시하고, 반영 여부는 사용자가 결정한다.
- **P7. 측정 없는 개선 없음.** 기록이 쌓이면 "덱당 일관성 위반 수" 같은 지표로 개선 효과를 숫자로 확인한다.
- **토큰 프로파일 사고.** 슬라이드 본문은 짧은 조각 대량이므로 경량 처리(조각별 추가 검증 호출 금지)가 기본이고, 문서처럼 긴 텍스트(발표자 노트·장문 박스)만 문서형 파이프라인(폴리싱+변경분 검증)의 대상이다. 토큰 0인 검사(일관성 스윕)와 기록은 항상 실행한다.
- **PPT 특수성: 공간이 품질 기준이다.** 덱 번역은 제한된 박스 안에서 정보가 명확히 읽히는 것이 핵심이라 일반 문서보다 어렵다. 길이는 스타일이 아니라 제약이다. 이 원칙은 세 곳에 배선된다: 오버플로의 결정론 검출(`fit.overflow`, WP-C3), 길이 예산을 제약으로 받는 부분 재번역(WP-C5), 슬라이드 문체 지시의 규칙 주입(WP-C1).
- **수정 루프가 곧 데이터 수집 지점이다.** 오답노트의 `corrected`(수정본)는 사람이 고치는 순간에만 생긴다. 앱 안에 고치는 자리(WP-C5)가 없으면 PPT 쪽 기록은 검출만 있고 정답이 없는 반쪽이 된다.

---

## 2. 스테이지 매핑

| 스테이지 | ppt-translator 적용 | 상태 |
|---|---|---|
| S0 원문 사전 분석 | 덱당 1회 경량: 기존 요약 체인을 확장해 주제·독자에 더해 **핵심 반복 용어 후보**와 **원문 이상(오타·수치 의심) 노트**를 함께 산출, 번역 프롬프트에 주입 | 원형 있음, 확장 |
| S1 번역 | 기존 번역 체인에 팀 규칙 슬라이스 주입(WP-C1). 고정 컨텍스트(규칙·용어·덱 요약)는 덱당 1회 구성해 조각 배치 호출에 재사용(프롬프트 캐싱 유도: 안정된 프리픽스로 배치) | 보강 |
| S2 폴리싱 | 발표자 노트·장문 텍스트박스만 선택 적용(WP-C4) | 신규, 선택 |
| S3 변경분 검증 | S2를 했을 때만: 폴리싱 전후 diff의 변경 부분만 원문 대조. 검증 없는 폴리싱은 금지 | 신규, 선택 |
| S4 일관성 스윕 | **핵심 가치.** 번역 완료 후 덱 전체 결정론 검사(WP-C3) | 신규 |
| S5 적용 | 기존 흐름 유지. 스윕 결과는 목록으로 제시, 반영은 사용자 결정 | 유지 |
| 기록 | 품질 레코드·작업 기록 JSONL(WP-C2) | 신규 |

---

## 3. 공유 데이터 계약 (정본 설계서 4장의 복제)

**이 장의 필드명·값 어휘는 코드가 바뀌어도 유지한다.** 확장은 optional 필드 추가로만.

### 3.1 품질 레코드 (quality_record)

지적·수정·판정 하나가 레코드 하나. JSONL(줄당 1건, snake_case)로 기록한다.

```jsonc
{
  "id": "qr_...",
  "client": "ppt_translator",        // 이 리포는 항상 이 값
  "project_id": "...",               // 작업(job) 식별자
  "created_at": 1780000000000,       // epoch ms

  "doc_ref": "deck:quarterly_report.pptx",
  "route_id": null,                  // 팀 라우트 id를 알면 기록, 모르면 null
  "direction": "ko_to_en",
  "content_type": "presentation",

  // PPT 전용 확장 (optional): 문제 위치
  "location": { "slide": 12, "shape": "TextBox 3", "paragraph": 0 },

  "segment": {
    "source": "원문 텍스트",
    "output": "문제가 된 번역",
    "corrected": "확정 수정본",        // nullable
    "context": null
  },

  "finding": {
    "type": "terminology.inconsistency",  // 3.2의 어휘
    "severity": "major",                  // critical | major | minor
    "description": "왜 문제인지 한 줄",
    "suggested_fix": null
  },

  "origin": {
    "stage": "s1_translate",         // s0_preflight | s1_translate | s2_polish
                                     // | s3_verify | s4_consistency | manual_edit
    "caught_by": "s4_consistency",   // s3_verify | s4_consistency | script | human
    "executor": "app",               // app | claude_agent | human
    "producer_model": "gpt-5.5",     // nullable
    "reviewer_model": null
  },

  "disposition": "proposed",         // proposed | accepted | rejected | superseded

  "promotion": {
    "status": "candidate",           // candidate | promoted | rejected | not_applicable
    "matched_rule": null
  }
}
```

### 3.2 오류 유형 어휘 (type)

`accuracy.omission` `accuracy.addition` `accuracy.mistranslation` `accuracy.nuance` `fluency.collocation` `fluency.wording` `fluency.structure` `fluency.grammar` `fluency.repetition` `fluency.verbosity` `fluency.weak_ending` `terminology.violation` `terminology.inconsistency` `consistency.phrase` `source.error` `source.ambiguity`

PPT에서 주로 쓰이는 값: `terminology.violation`(용어집 위반), `terminology.inconsistency`(덱 내 용어 불일치), `consistency.phrase`(용어 외 표현의 불일치), `source.error`(S0가 찾은 원문 이상). WP-C4를 구현하면 accuracy.*, fluency.*도 쓰인다.

PPT 전용 확장 값: `fit.overflow` (번역이 텍스트박스 공간을 초과. 결정론 검출). 정본 계약에도 등재돼 있다.

### 3.3 severity

`critical | major | minor`. 의미 훼손(accuracy.*)은 critical, 용어·콜로케이션 오류는 major, 개선 제안 수준은 minor.

### 3.4 작업 기록 (quality_run)

스테이지 실행 1회가 1행. 레코드의 분모다.

```jsonc
{
  "id": "run_...",
  "client": "ppt_translator",
  "project_id": "...",
  "started_at": 1780000000000,
  "stage": "s1_translate",
  "executor": "app",
  "model": "gpt-5.5",
  "direction": "ko_to_en",
  "route_id": null,
  "doc_words": 3200,                 // 덱 전체 번역 대상 단어 수
  "findings_count": { "critical": 0, "major": 2, "minor": 7 },
  "notes": null
}
```

### 3.5 저장과 교환

- ppt-translator는 DB를 추가하지 않는다. **JSONL 파일 기록이 정본**이다: `quality_records.jsonl`, `quality_runs.jsonl`. 위치는 설정 가능(기본: 앱 데이터 디렉토리, 작업별 파일이 아니라 누적 append).
- 팀의 규칙 승격 도구(trans_agent의 mine-feedback)가 이 파일을 읽는다. 경로만 알려주면 되도록 위치를 설정에 노출한다.
- 기록은 best-effort: 기록 실패가 번역 작업을 막지 않는다 (경고 로그만).

### 3.6 팀 규칙집(translation-rules.json) 소비 계약

- 정본은 trans_agent 리포의 `references/translation-rules.json`. 구조: `meta` + `kr_target_rules`(EN→KR) / `en_target_rules`(KR→EN) / `bidirectional`. 엔트리 스키마: `id, summary, avoid[], prefer[], why, examples[{bad,good}], severity(red|yellow), locked_term, memory_ref`.
- ppt-translator는 이 파일을 **파싱해 소비만** 한다. 편집 UI를 만들지 않는다.
- 주입 슬라이스 (생성용): 도착어 방향 버킷 + bidirectional에서 `summary`, `avoid`, `prefer`, `locked_term`만. **severity=red 룰은 examples 1쌍(bad → good)을 포함한다.** `why`·`memory_ref`·승격 메타는 주입하지 않는다.
- 렌더 형식 예시 (한 룰당):

```
- {summary}
  avoid: {avoid 항목들} → use: {prefer 항목들}
  [LOCKED TERM, use exactly: {locked_term}]   (locked_term 있을 때만)
  e.g. "{examples[0].bad}" → "{examples[0].good}"   (red 룰만)
```

- `severity=red`의 `locked_term`이 있는 룰은 규칙 주입과 별개로 WP-C3 스윕의 검사 대상으로도 쓴다 (용어집과 동일 취급).

---

## 4. 작업 패키지 (WP-C)

각 패키지는 독립 배포 가능 단위. 구현 형식은 리포 컨벤션(Python >= 3.12, pytest, frontend는 React)과 구현자 판단에 맡기되, "요구사항"의 각 항목은 참이어야 한다.

### WP-C1. 팀 규칙 주입 (최우선, 토큰 대비 효과 최대)

**목적**: 팀이 축적한 도착어 하드룰(금지 표현·확정 용어)이 번역 생성 시점에 작동하게 한다.

**요구사항**:
1. 설정에 translation-rules.json 파일 경로를 지정할 수 있다 (미지정 시 기능 비활성, 기존 동작 무변화).
2. 번역 방향에 맞는 슬라이스(3.6)가 번역 프롬프트에 주입된다. 덱당 1회 구성해 모든 조각 호출에 재사용한다.
3. 규칙·용어·덱 요약 등 고정 컨텍스트는 프롬프트의 안정된 앞부분에 배치한다 (프롬프트 캐싱 히트 유도. 조각 대량 호출 구조라 절감 효과가 크다).
4. 파일 없음·파싱 실패는 조용히 넘어가지 않고 UI에 상태를 표시한다 (주입 누락은 조용한 품질 저하).
5. 시스템 프롬프트에 **슬라이드 문체 지시**가 포함된다: 슬라이드 본문은 완전한 문장보다 명사구·간결체가 관용이며, 도착어 확장(특히 KR→EN 장문화)을 억제하고 공간 안에서 명확히 읽히는 표현을 우선한다는 취지. 발표자 노트에는 적용하지 않는다(노트는 문서형 텍스트).

**코드 앵커 (참고)**: `src/chains/translation_chain.py`(프롬프트 조립), `src/utils/config.py`(설정), `src/utils/glossary_loader.py`(외부 파일 로딩 선례).

**수용 기준**: 규칙 파일을 연결하고 EN→KR 번역 시, 프롬프트(디버그 로그)에 kr_target_rules + bidirectional 슬라이스와 red 룰 예시가 포함된다. 미연결 시 기존과 완전히 동일하게 동작한다.

### WP-C3. 덱 일관성 스윕 (품질 체감 효과 최대)

**목적**: "3번 슬라이드와 27번 슬라이드에서 같은 용어가 다르게 번역"되는 PPT 대표 사고를 토큰 0으로 잡는다.

**요구사항**:
1. 번역 완료 후 결정론 검사 4종이 실행된다 (LLM 호출 없이):
   - **용어집 위반**: 용어집(기존 glossary_loader) + 규칙집의 locked_term에 대해, 원문에 등장하는 용어의 대응 번역이 결과물에 없는 조각 검출.
   - **표현 분산**: 원문에서 2회 이상 반복되는 표현(정규화 후 비교, 최소 길이 임계)이 서로 다르게 번역된 조각 쌍 검출. 완전 동일 문단은 기존 repetition 캐시가 이미 1회 번역으로 보장하므로, 이 검사의 초점은 **부분 반복**(문장 안에 포함된 용어·구, 표기 변형)이다.
   - **미번역 조각**: 번역 대상인데 원문과 동일하게 남은 조각 (언어 감지 활용).
   - **공간 초과**: 기존 text fit 로직을 검출 항목(`fit.overflow`)으로 승격. 초과 조각은 WP-C5의 "길이 예산 재번역" 액션과 연결된다.
2. 오탐 완화 수단(정규화, 임계값)을 두되 자동 치환은 하지 않는다 (P1).
3. 결과는 슬라이드 번호·위치와 함께 검토 목록으로 사용자에게 제시된다 (WP-C5의 검토·수정 화면과 통합. C5 이전에 임시로 만든다면 읽기 전용 목록으로 충분).
4. 검출 각각이 품질 레코드로 기록된다 (caught_by=s4_consistency).

**코드 앵커 (참고)**: `src/utils/repetition.py`(정규화·반복 검출 선례), `src/core/text_extractor.py`·`ppt_parser.py`(조각·위치 정보), `src/utils/language_detector.py`, text fit 검사(결정론 QA 층의 자리).

**수용 기준**: 같은 원문 용어가 두 가지로 번역된 테스트 덱에서 해당 조각 쌍이 슬라이드 번호와 함께 검출된다. 용어집 등록 용어를 어긴 조각이 검출된다. 깨끗한 덱에서는 검출 0건이다.

**비범위**: LLM 기반 "정당한 변주" 판정, 자동 수정, 서식 관련 검사(기존 text fit이 담당).

### WP-C2. 품질 기록

**목적**: 3.1/3.4 계약대로 JSONL 장부를 남겨, 팀 규칙 승격 루프와 KPI 측정에 합류한다.

**요구사항**:
1. `quality_records.jsonl` / `quality_runs.jsonl` 기록 모듈이 추가된다 (3.5).
2. 다음이 자동 기록된다: 번역 실행마다 run 1행(단어 수·모델·방향 포함), WP-C3 검출 각각(record, disposition=proposed), S0가 찾은 원문 이상(source.error/ambiguity).
3. disposition 갱신(accepted/rejected)과 corrected 캡처는 WP-C5의 검토·수정 화면이 담당한다. C5 이전 단계에서는 proposed로만 기록해도 된다 (기록이 없는 것보다 disposition 없는 기록이 낫다).
4. 기록 파일 경로가 설정에 노출된다.

**수용 기준**: 테스트 덱 1건을 번역하면 run 1행과 스윕 검출 레코드가 계약 스키마대로 파일에 append된다.

**비범위**: 통계 대시보드 (분석은 팀의 에이전트가 파일을 읽어 수행), 번역 후 외부에서 수정된 pptx를 재수입해 diff하는 기능 (가치는 있으나 후순위, 별도 논의).

### WP-C5. 검토·수정 루프 (편의성의 핵심이자 데이터 수집 지점)

**목적**: 번역 직후 결과를 보고 즉시 고치고, 일부 표현만 재번역해 반영하고, 고친 내용이 덱 전체의 반복 조각에 전파되게 한다. PPT는 완성 파일을 열어보기 전까지 결과를 알 수 없는 블랙박스가 되기 쉬운데, 이 루프가 그 간극을 없앤다. 동시에 **오답노트의 corrected(수정본) 삼중항을 캡처하는 유일한 지점**이다.

**요구사항**:
1. **검토 화면**: 번역 완료 후 슬라이드 순서대로 원문·번역 조각을 나란히 목록으로 보여준다. 완전한 슬라이드 렌더링은 불필요하다. 슬라이드 번호·도형 라벨(가능하면 슬라이드 썸네일)로 위치를 식별하고, WP-C3 스윕 검출과 `fit.overflow` 배지가 해당 조각에 표시된다.
2. **조각 단위 액션 3종**:
   - 직접 수정: 인라인 편집 → pptx에 재기록. 서식 보존은 기존 writer 경로를 재사용한다.
   - 조건부 재번역: 지시("더 짧게", "용어 X 사용", 톤 조정)와 함께 해당 조각만 재번역하고, 전후 비교로 확인한 뒤 적용한다. **길이 예산 파라미터**: 대상 박스에서 산출한 최대 글자 수를 프롬프트 제약으로 전달한다 (공간 제약 대응의 핵심).
   - 무시(반려): 스윕 검출을 무시 처리한다.
3. **반복 전파**: 수정·재번역이 적용될 때 같은 원문의 다른 조각에도 반영한다.
   - 완전 동일 조각(정규화 후 일치): 자동 전파. 기존 repetition의 canonical map을 전파 지도로 재사용하고, 전파된 개수를 사용자에게 표시한다.
   - 부분 포함(수정한 표현이 다른 조각의 일부로 등장): 후보 목록을 제시하고 사용자가 선택 적용한다. 한국어 형태소 변형 오탐 때문에 부분 일치의 자동 치환은 금지한다 (P1).
4. **기록 배선**: 모든 액션이 품질 레코드가 된다. 직접 수정·재번역 적용은 corrected가 채워진 accepted 레코드(원문·기존 번역·수정본 삼중항 + 전파 대상 수), 무시는 rejected. WP-C2 모듈로 기록한다.
5. 수정 결과는 산출물 pptx에 반영하되 원본 파일은 손상하지 않는다.

**구현 가능성 확인 (2026-07-03 코드 기준 검증됨)**:
- 조각 주소 지정: `ParagraphInfo`가 `slide_index / shape_index / paragraph_index / original_text / slide_title / is_note`를 이미 보유한다. 사후 재기록의 위치 추적자로 그대로 쓸 수 있다 (도형 추가·삭제가 없는 한 인덱스 유효).
- 서식 보존 재기록: `ppt_writer.py`의 run 그룹핑·적용 계열(`_group_runs_by_format`, `apply_translations`)이 단일 문단 재기록에 재사용 가능하다.
- 길이 예산: text fit 계열(`apply_text_fit`, `_calculate_available_expansion` 등)이 도형 공간 계산을 이미 하므로, 여기서 최대 글자 수 예산을 산출한다.
- 노트 구분: `is_note` 플래그가 있어 슬라이드 본문·노트 분리(WP-C1 문체 지시, WP-C4 대상 선별)가 즉시 가능하다.

**필요한 신규 배관 (이 패키지의 실제 작업량)**:
- 작업(job) 완료 후에도 조각 목록(원문·번역·위치·검출 플래그)이 유지돼야 한다. 현재 API는 작업 생성 → 진행 폴링 → 완성 파일 다운로드 구조라, job 상태에 조각 목록을 보존하거나 산출물 재파싱 경로를 만든다.
- API 2종 추가: 조각 목록 조회(GET, 검출 배지 포함), 조각 편집·재번역 적용(POST, 전파 옵션 포함).

**코드 앵커 (참고)**: `frontend/src/components/translation/`(결과 화면), `src/core/ppt_writer.py`·`ppt_parser.py`(조각 위치·재기록), `src/utils/repetition.py`(canonical map), `src/services/job_manager.py`(작업 상태), `api.py`(jobs 엔드포인트).

**UI 참고**: 검토 화면 목업이 `ppt-review-ui-mockup.html`(이 문서와 같은 폴더)에 있다. 레이아웃·배지·액션 상태의 의도를 보여주는 참고물이며, 구현은 리포의 기존 디자인 시스템을 따른다.

**수용 기준**: 덱에서 4회 반복되는 문구를 검토 화면에서 한 번 수정하면 4곳 모두 반영되고, 삼중항과 전파 정보가 담긴 accepted 레코드가 기록된다. `fit.overflow` 조각을 길이 예산과 함께 재번역하면 결과가 예산을 준수한다 (초과 시 1회 자동 재시도 후 경고).

**비범위**: 슬라이드 위지윅 편집(텍스트 조각 편집만), 서식 편집, 원본 pptx 직접 수정.

### WP-C4. 장문 텍스트 폴리싱 + 변경분 검증 (선택, 후순위)

**목적**: 덱 안의 "문서적 텍스트"(발표자 노트, 장문 텍스트박스)에 문서형 파이프라인을 적용한다.

**요구사항**:
1. 대상 선별이 기계적이다: 발표자 노트 전부 + 길이 임계(예: 200자) 이상 텍스트박스.
2. 폴리싱은 도착어 단독(원문 비노출)으로 다듬고, 폴리싱 전후 diff의 **변경 부분만** 원문과 대조 검증한다. 검증 없는 폴리싱 적용은 금지 (P4).
3. 폴리싱 모델과 검증 모델은 다르게 배치한다 (P3). 문제 판정 부분은 검토 목록에 추가되고 레코드로 기록된다.
4. 기본 꺼짐(옵트인). 슬라이드 본문에는 적용하지 않는다.

**수용 기준**: 노트에서 문장 하나가 폴리싱 중 누락되도록 유도한 테스트에서 해당 변경이 accuracy.omission으로 검출된다.

---

## 5. 구현 순서와 검증

| 순서 | 패키지 | 근거 |
|---|---|---|
| 1 | WP-C1 (규칙 주입) | 구현이 가장 가볍고, 생성 품질을 즉시 올린다 |
| 2 | WP-C3 (일관성 스윕) | PPT 대표 사고를 토큰 0으로 차단. 검출이 C5 검토 화면의 재료가 된다 |
| 3 | WP-C2 + WP-C5 (기록 + 검토·수정 루프) | 함께 구현한다. C5가 편의성의 핵심이자 corrected 캡처 지점이고, C2는 그 배선이다 |
| 4 | WP-C4 (장문 폴리싱) | 선택. 노트가 많은 덱에서만 가치 |

KPI (기록에서 도출): 덱 1천 단어당 일관성 위반 수(스윕 검출), 용어집 위반 수, 미번역 조각 수, 규칙 주입 전후의 동일 지표 비교, 1천 단어당 토큰 비용.

검증: 각 WP는 pytest 유닛 테스트를 갖춘다. 고정 테스트 덱 2~3개(반복 용어 포함 덱, 용어집 위반 유도 덱, 깨끗한 덱)를 tests/에 두고 회귀 기준으로 쓴다.

---

## 6. 비범위 (하지 않는 것)

- OddEyes로의 통합, OddEyes SQLite 장부 공유 (ppt-translator는 JSONL로 충분).
- 자동 치환·자동 수정. 검출과 보고까지만 (P1, P6).
- 규칙집·용어집 편집 UI. 정본 관리는 팀의 Claude Code(trans_agent) 쪽 책임.
- 슬라이드 본문 조각별 폴리싱·검증 호출 (토큰 과잉. 경량 프로파일 원칙).
- 모델 파인튜닝.

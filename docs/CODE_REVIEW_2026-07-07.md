# 코드 리뷰: 동시성 / 성능 / 로직 / 아키텍처 (2026-07-07)

4개 영역 병렬 심층 리뷰 결과. 모든 발견은 실제 코드 확인을 거쳤고, 상위 항목은 교차 검증 완료.

## 총평

코어 설계는 견고하다: Job 이벤트 브리징(`call_soon_threadsafe` + 루프 캡처), 원자적 승인(`try_create_job` + 429), 세마포어 누수 없음, 색 세그먼트 concat 검증, 반복 dedup 정합성, keychain → env 주입 키 흐름은 모두 올바르게 구현되어 있다. 문제의 중심은 (1) 취소/리뷰 편집 경로의 동시성, (2) 배치 파라미터가 병렬성을 무력화하는 성능 문제, (3) 리뷰 세션 재렌더의 비멱등성, (4) api.py 비대화와 프런트 계약 사본이다.

---

## High

### C-1. 취소가 워커 스레드를 멈추지 못함: 3-스레드 풀 고갈
`api.py:51` `ThreadPoolExecutor(max_workers=3)`가 `api.py:58`에서 기본 executor로 설정됨. `api.py:544`의 `run_in_executor(None, service.translate, request)`는 취소 불가능하고, DELETE는 코루틴만 취소(`api.py:593`, `job_manager.py:219-224`). 취소된 번역의 스레드는 LLM 호출을 계속하며 토큰을 소모한다. 취소 3회면 풀 전체가 고아 스레드로 점유되어 서버가 사실상 정지한다.
수정: `service.translate`에 `threading.Event` 취소 플래그를 전달해 배치 사이마다 확인.

### C-2. ReviewSession 편집/렌더 경로 무락 + 비멱등 재적용 (3개 버그 결합)
1. **락 부재**: `api.py:1076-1106` 편집 엔드포인트가 executor에서 재번역/`render()` 실행 중에도 다른 편집 요청을 받는다. `render()`(review_session.py:230-245)는 공유 live `Presentation`을 제자리 변형하므로 스레드 겹침 시 XML 손상 가능.
2. **텍스트핏 누적**: `render()`가 이미 축소/확장된 presentation 위에 `text_fit_mode`를 전체 재적용. 편집 k회 후 폰트 최대 0.8^k 배, 박스는 매번 30% 캡 기준으로 계속 팽창 (ppt_writer.py:534-586).
3. **색상 그룹 재유도**: 재렌더 시 `group_rprs`를 원본이 아닌 "현재 run 상태"에서 재계산(ppt_writer.py:184-222)해 어순 변경 세그먼트의 색이 편집 때마다 스왑되거나, 그룹 수 감소로 검증에 걸려 색 전체 소실.
수정: Job별 `asyncio.Lock`으로 편집+렌더 직렬화, 렌더 시 원본 폰트/도형 크기와 원본 그룹 rPr 스냅샷을 세션에 보존해 멱등 재적용.

### P-1. 배치 크기 하한이 동시성(8)을 무력화
`translation_service.py:393-394`의 legacy 보정이 기본 설정에서 항상 발동해 배치 크기를 80으로 고정. 480문단 미만 덱은 `max_concurrency=8`을 채우지 못한다. 300문단 덱 기준 wall time 약 2배, 160문단 덱은 최대 4배 손해.
수정: 보정 제거 또는 `ceil(N / max_concurrency)` 기반 제안값 존중.

### P-2. 다색 문단 색상 패스가 완전 직렬 + 이중 번역
`color_distribution_chain.py:270-285`가 8개씩 동기 `invoke()` 순차 실행. 다색 문단 40개면 진행률 80% 지점에서 추가 1.5~2.5분. 게다가 다색 문단은 메인 배치에서 이미 번역된 뒤(translation_service.py:723-746) 다시 번역되어 토큰 2배 지출(translation_service.py:924).
수정: `batch_as_completed`로 병렬화, 메인 배치에서 다색 문단 제외.

### P-3. Anthropic `max_tokens=4096` 고정 vs 배치 80의 충돌
`llm_factory.py:55` 기본 4096, 번역 체인 미지정(translation_chain.py:107). 배치 80문단의 JSON 응답이 잘리면 tenacity가 같은 크기로 3회 재시도 후 실패. 문단당 51 출력 토큰만 허용되는 셈.
수정: 모델별 상한 명시 전달, 배치 크기를 출력 토큰 예산과 연동.

### A-1. `_retranslate_fragment`가 라우트 파일에서 서비스 private 호출
`api.py:911-993`(83줄)이 `TranslationService._translate_colored_paragraphs_with_segments`(private) 직접 호출 + chains/utils 직접 import. 재번역 도메인 로직이 HTTP 계층에 존재.
수정: `ReviewSession` 또는 서비스의 공개 메서드로 이동.

### A-2. MODEL_REGISTRY "단일 소스"가 프런트에서 깨짐
`src/services/models.py:19-36` 레지스트리의 전체 사본이 `frontend/src/hooks/useConfig.ts:13-19`(FALLBACK_MODELS), `translation-store.ts:62`(기본 모델)에 존재. 언어 목록도 사본 관계. 모델 교체 시 사이드카 기동 지연 중 낡은 ID가 노출되어 400 유발.
수정: 빌드 타임 codegen(레지스트리 → TS 상수) 또는 폴백 제거 후 로딩 상태 처리.

---

## Medium

### 동시성
- **C-3. `update_job_progress` 터미널 가드 없음** (job_manager.py:233-251): 취소된 작업에 고아 스레드가 진행 이벤트를 계속 기록, 프런트가 취소 작업을 "진행 중"으로 재표시 가능. `complete_job`과 동일한 가드 추가.
- **C-4. SSE 이벤트 큐 단일 소비자 설계** (job_manager.py:109-120, 336-368): 히스토리 재생 + 같은 큐 소비로 이벤트 중복 전달, 재연결 시 분산. 단, 프런트가 SSE를 아예 안 쓰므로(A-4 참조) 제거가 정답.
- **C-5. 레이트리미터가 인스턴스/작업 단위** (llm_factory.py:76, translation_service.py:810-817): `create_llm`마다 새 `InMemoryRateLimiter`(1 rps), TPM 예산도 job 단위 계산. 동시 job 2개면 공급자 한도 2배 초과. 전역 싱글턴 공유 + 예산을 동시 작업 수로 분할.
- **C-6. 이벤트 루프 블로킹 CPU 작업**: 이미지 압축(api.py:690, 1520), 전체 덱 파싱(api.py:1240), glossary 파싱(api.py:700-703)이 async 핸들러에서 동기 실행. 대형 덱이면 수십 초간 폴링/health 전체 정지. `run_in_executor` 이관 (단, C-1의 풀 크기와 함께 조정).
- **C-7. 레거시 `/translate`가 세마포어 우회** (api.py:1566): 동시 3건이면 풀 고갈. A-3과 함께 삭제 권장.

### 로직
- **L-1. 배치 결과 개수 기반 정렬: 중간 누락 시 전체 시프트** (translation_chain.py:196-232): 모델이 중간 항목 하나를 누락하면 이후 번역이 한 칸씩 밀려 잘못된 문단에 기록됨. 경고 로그만 남고 조용히 오정렬. 출력 스키마에 index/id를 포함해 id 기준 정렬.
- **L-2. 용어집 치환 3종 결함** (glossary_loader.py:98-124): (a) 길이 우선순위 없음: "게임"이 "게임팀"보다 먼저 치환되면 긴 용어 영구 미매칭, (b) CJK plain replace의 부분 문자열 오염("공지사항" → "ball지사항"), (c) 다단어 Latin 구문이 `\b` 우회("smart director" → "sm아트 디렉터"). 길이 내림차순 + 단일 alternation 정규식으로 단일 패스 치환.
- **L-3. `re.sub` 치환 문자열 escape 미처리** (glossary_loader.py:100): target에 백슬래시 포함 시 `re.error`로 작업 전체 실패. `lambda m: target` 사용.
- **L-4. 다색 문단이 glossary 후처리/일관성 sweep 우회** (translation_service.py:891-937): colored 결과가 후처리 이후에 덮어써 용어 강제와 sweep 배지가 실제 결과와 어긋남.
- **L-5. 빈 번역("")이 원문을 조용히 지움** (ppt_writer.py:507-531): 개수는 맞지만 빈 항목이면 문단이 통째로 비워짐. 빈 번역 + 비어있지 않은 원문이면 원문 유지.
- **L-6. 소프트 개행(`a:br`)이 공백 없이 병합** (ppt_parser.py:174): "Revenue Summary" + "First Half"가 "SummaryFirst"로 붙어 번역 입력 오염.
- **L-7. auto_shrink 모드에서도 박스 폭 확장 수행** (ppt_writer.py:552-580): 확장 블록이 모드로 게이트되지 않아 "글자 축소만" 선택한 사용자의 레이아웃이 변형됨.

### 아키텍처
- **A-3. `/translate` 죽은 표면 + 90줄 중복** (api.py:1478-1539): 프런트 호출 0건, 검증 로직 드리프트 진행 중. 삭제 권장.
- **A-4. SSE 3중 혼선**: `sse-client.ts`는 실제로 2초 폴링, 백엔드 SSE(`/events`)는 프런트 호출 0건인 죽은 코드인데 이벤트 큐/스레드 브리지 복잡도를 유지비로 요구. 폴링으로 확정하고 SSE 기계장치 제거.
- **A-5. chains → services 역방향 의존** (summarization_chain.py:12, llm_factory.py:47): models.py가 leaf라 우연히 안전. 레지스트리를 최하층 모듈로 강등.
- **A-6. 프로바이더 추가 시 9개 파일 13개 지점 수정 필요**: api.py 키 검증 4개소, lib.rs env 주입 등. provider descriptor를 레지스트리에 통합하면 데이터 구동으로 전환 가능.
- **A-7. `generate-instructions`가 llm_factory 우회 + LLM 응답 INFO 로깅** (api.py:1360-1413): rate limiter 미적용, 사용자 문서 내용 로그 유출. chain으로 이동, DEBUG 강등.
- **A-8. 테스트 공백**: 리뷰 루프 API 3종(fragments GET/POST, 재번역) 무테스트: 가장 복잡한 상태 변이 경로. `_execute_translation` 오케스트레이션, 프런트 전체(테스트 러너 자체가 없음)도 공백.

### 성능
- **P-4. 완료 job의 메모리 상주** (job_manager.py:373, review_session.py:47-72): 출력 pptx 바이트 + live Presentation이 1시간 TTL 동안 유지. 500MB 덱이면 1GB 이상 점유. 다운로드는 `read()` 전체 사본(api.py:882-883) 대신 StreamingResponse, 리뷰 미사용 시 조기 해제.
- **P-5. `batch_as_completed` 예외 시 미수확 성공분 유실** (translation_chain.py:298): `return_exceptions=False`라 1개 실패가 같은 wave의 최대 7개 성공 배치를 버리고 재과금. `return_exceptions=True`로 배치별 판별.
- **P-6. 용어집 매칭 O(P×G) 3패스 + 패턴 반복 컴파일** (glossary_loader.py, consistency_sweep.py:196-197): P=1,000 × G=500이면 정규식 스캔 150만 회. 로드 시 사전 컴파일.

---

## Low

- delete_job이 COMPLETED를 CANCELLED로 덮어씀 (job_manager.py:218-227): 완료 직후 취소 시 결과 다운로드 400.
- RUNNING이 세마포어 획득 전 설정됨 (api.py:776): /health 카운트 부정확. QUEUED 상태 도입.
- SSE 300초 하드 타임아웃 (job_manager.py:328), `except (CancelledError, Exception): pass` (job_manager.py:223).
- tenacity 재시도 시 진행률 일시 역행 (translation_service.py:239-299): tracker에 최고 퍼센트 클램프.
- 미매핑 언어가 조용히 "영어" 라벨링 (language_detector.py:62-64).
- 재번역 fragment에 glossary 전/후처리 미적용 (api.py:944-963).
- 커스텀 파일명 ".pptx" 중복 (api.py:309-311).
- `LANGUAGE_CODE_MAP` 죽은 사본 (api.py:394-402), `rules_file` 좌초 기능(프런트 미구현), `_rpr_key`의 run당 deepcopy 비용 (ppt_writer.py:50-63).
- restart_sidecar가 실행 중 작업 확인 없이 kill (lib.rs:113-118).
- 폴링 setInterval이 이전 요청 완료를 안 기다림 (sse-client.ts:71): 백엔드 지연 시 요청 중첩.

---

## 잘 구현된 부분 (검증 완료)

- Job 이벤트 브리징: 루프 캡처, 루프 스레드 판별, 종료 폴백까지 정확 (job_manager.py:75-107).
- `try_create_job` 원자적 승인 + 실패 시 롤백, 세마포어 누수 없음 (api.py:520, 787-797).
- `complete_job`/`fail_job` 터미널 가드 + `_state_lock`, await 없는 락 구간.
- tenacity 공유 accumulator로 성공 배치 재과금 방지 (translation_chain.py:272-306).
- 색 세그먼트 "concat == translation" 엄격 검증으로 텍스트 유실 차단.
- placeholder xfrm 4속성 materialize로 cy=0 붕괴 방지 (ppt_writer.py:319-328).
- 반복 dedup(repetition.py), `chunk_paragraphs` 인덱싱, `_force_match_expected` 원문 패딩 정확.
- keychain → Rust env 주입 → Settings 키 흐름: 키가 HTTP를 경유하지 않음.
- 사이드카 콜드스타트 핸드셰이크(소켓 먼저, 무거운 import 나중), stderr drain, 종료 시 child kill.
- 모델 allowlist, 파일 시그니처 검증, loopback 바인딩.

---

## 권장 우선순위

1. **C-1 + C-2**: 취소 협조 플래그 + 리뷰 세션 락/멱등 렌더. 데이터 손상과 서버 정지를 막는 안정성 핵심.
2. **P-1 + P-2 + P-3**: 배치/병렬 파라미터 정리. 코드 몇 줄로 체감 속도 2~4배.
3. **L-1 + L-2 + L-3**: 조용한 오번역(시프트, 용어집 오염) 제거. 번역 품질 신뢰성.
4. **A-3 + A-4 + C-7**: 죽은 표면(/translate, SSE) 삭제. 이후 리팩터링 비용 절감.
5. **A-1 + A-2, api.py 라우터 분리**: 구조 정리는 위 수정과 함께 점진 진행.

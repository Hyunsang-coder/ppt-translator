# ADR-0001: 진행률은 SSE 대신 작업 상태 폴링으로 받는다

- **상태**: 채택됨
- **날짜**: 2026-07-08

## 맥락

백엔드에는 `GET /api/v1/jobs/{job_id}/events` SSE 엔드포인트와 이를 떠받치는 기계장치가 있었다. `JobManager.stream_events`, 작업마다의 `_event_queue`/`_loop` 필드, 워커 스레드에서 이벤트 루프로 넘기는 `call_soon_threadsafe` 브리지까지. 그런데 프런트엔드는 이 스트림을 한 번도 구독하지 않았다. `lib/sse-client.ts`는 이름과 달리 2초 간격 폴러였고, 진행률은 전부 `GET /api/v1/jobs/{job_id}` 응답에서 나오고 있었다.

즉 스레드 안전성 부담이 가장 큰 코드 경로가 아무도 쓰지 않는 기능을 위해 유지되고 있었다.

## 결정

SSE를 걷어내고 폴링을 유일한 진행률 전달 방식으로 확정한다. SSE 엔드포인트, `stream_events`, 작업별 이벤트 큐, `call_soon_threadsafe` 브리지를 삭제했다. `add_event()`는 평범한 deque append가 됐다(GIL 하에서 원자적이고, 동시 순회하는 소비자가 더 이상 없다). 이벤트 히스토리와 `JobEvent`는 상태 조회·디버깅용으로 남긴다.

같은 작업에서 프런트엔드가 호출하지 않던 동기 `POST /api/v1/translate` 레거시 엔드포인트도 함께 제거했다. 이 엔드포인트는 작업 동시성 세마포어를 우회하고 있었다.

## 결과

- 진행률 지연 상한이 2초로 고정된다. 로컬 사이드카를 상대로 하는 데스크톱 앱이라 체감 차이가 없다고 보고 감수한다.
- 워커 스레드 → 이벤트 루프 브리지가 사라져 작업 상태의 동시성 표면이 줄었다.
- 백엔드 약 185줄과 그에 딸린 API 테스트 2개가 사라졌다.
- 나중에 스트리밍이 필요해지면 다시 만들어야 한다. 폴링으로 감당 안 되는 요구가 생기기 전까지는 그 비용을 지불하지 않는다.

## 참고

- 커밋: `fec3bb6`
- 코드: `frontend/src/lib/sse-client.ts`, `src/services/job_manager.py`
- 문서: [`docs/KEY_PATTERNS.md`](../KEY_PATTERNS.md) "Async Job Flow"

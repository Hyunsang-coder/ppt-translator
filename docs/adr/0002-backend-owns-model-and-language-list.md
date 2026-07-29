# ADR-0002: 모델·언어 목록의 단일 소스는 백엔드다

- **상태**: 채택됨
- **날짜**: 2026-07-08

## 맥락

`MODEL_REGISTRY`(`src/services/models.py`)가 지원 모델의 단일 소스였는데, 프런트엔드가 하드코딩 사본을 따로 들고 있었다. `useConfig`의 `FALLBACK_MODELS`/`LANGUAGES`, `translation-store`의 기본 모델 ID. 백엔드에서 모델을 갈아끼울 때마다 두 곳을 맞춰야 했고, 안 맞추면 조용히 어긋났다.

사본의 존재 이유였던 "설정을 아직 못 받아온 순간" 자체가 실제로는 없었다. 데스크톱 셸의 `SidecarProvider`가 사이드카에 닿을 때까지 UI 전체를 막기 때문에, 설정 소비자가 마운트되는 시점엔 `/api/v1/config|models|languages`가 이미 응답한다.

## 결정

프런트엔드의 하드코딩 사본을 전부 지우고 백엔드 응답만 쓴다. 설정 로드에 실패하면 stale 사본을 내놓는 대신 빈 목록을 노출하고 드롭다운을 비활성으로 렌더한다. 기본 모델은 `config.default_model`로 채우고, 없으면 백엔드가 광고한 첫 모델을 쓴다.

## 결과

- 모델 추가·교체가 백엔드 한 곳에서 끝난다.
- 설정 로드 실패가 "빈 드롭다운"으로 드러난다. stale 목록으로 조용히 동작하는 것보다 낫다고 판단했다.
- 설정이 로드되는 짧은 구간 동안 모델이 빈 문자열이므로, `useTranslation`의 `canStart`가 `settings.model !== ""`을 확인해야 한다. 이 가드가 없으면 빈 모델로 제출될 수 있다.

## 참고

- 커밋: `4aa2544`
- 코드: `src/services/models.py`, `frontend/src/hooks/useConfig.ts`, `frontend/src/stores/translation-store.ts`

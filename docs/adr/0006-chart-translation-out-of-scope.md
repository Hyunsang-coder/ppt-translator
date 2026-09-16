# ADR-0006: 차트 번역은 지원하지 않고 원문 유지임을 명시한다

- **상태**: 채택됨
- **날짜**: 2026-09-16

## 맥락

`PPTParser.extract_paragraphs`는 텍스트 프레임·표·그룹만 수집하고 차트(`MSO_SHAPE_TYPE.CHART`)를 통째로 건너뛴다
(`src/core/ppt_parser.py`). 번역 결과물에서 차트 제목·카테고리·시리즈명은 조용히 원문 그대로 남는다.
반면 텍스트 추출(`src/core/text_extractor.py`, `charts="labels"`)은 차트 제목+데이터 표를 Markdown으로 뽑는다.
번역은 미지원인데 추출은 지원하는 비대칭이라, 사용자가 차트까지 번역될 거라 기대하면 침묵 실패가 된다.
장기 과제 큐의 첫 번째 항목이 이 범위 결정이었다.

## 결정

차트 번역을 지원하지 않고 원문을 유지함을 명시한다. 파서는 차트가 있으면 경고 로그를 남기고,
README·번역 화면에 "차트·SmartArt·OLE 내 텍스트는 번역되지 않습니다"를 적는다.

## 결과

얻는 것: 침묵 실패 제거. 차트가 있는 덱도 번역 범위를 예측할 수 있다.
잃는 것: 차트 텍스트는 수동 번역이 필요하다. 추출 Markdown의 차트 표를 번역해도 PPT에 역적용되지 않는다.

## 대안

제목만 지원: 차트 제목(`chart_title.text_frame`)은 텍스트 프레임이라 기술적으로 쉽지만,
제목만 번역되고 축·범례는 남으면 더 혼란스럽다. 기각.
완전 지원(제목+카테고리+시리즈명): 내장 워크북 재작성이 필요하고 레이아웃 깨짐 검증이 없어 장기 과제로 미룬다.

## 참고

- 코드: `src/core/ppt_parser.py`, `src/core/text_extractor.py`
- 문서: [`docs/MD_EXTRACTION_IMPROVEMENT_PLAN.md`](../MD_EXTRACTION_IMPROVEMENT_PLAN.md)

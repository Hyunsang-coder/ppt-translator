# 마크다운 추출 개선 계획 (LLM 최적화)

> 2026-07-07 세션에서 조사한 내용의 인수인계 문서.
> 목표: 텍스트 추출 탭의 Markdown 출력을 LLM이 PPT 내용을 정확히 파악할 수 있는 형태로 개선한다.

## 대상 코드

- 핵심 로직: `src/core/text_extractor.py` (전체 로직이 이 파일 하나에 있음)
  - `extract_slide()` : 슬라이드 1장 → `SlideDoc` (제목 선정 + 블록 수집)
  - `_iter_shapes()` : 도형 순회 (그룹 평탄화, **z-order 순서 그대로**)
  - `blocks_to_markdown()` / `docs_to_markdown()` : 블록 → Markdown 렌더링
- API: `api.py`의 `POST /api/v1/extract` (1188행 부근), `ExtractionOptions`로 옵션 전달
- 프론트: `frontend/src/hooks/useExtraction.ts`, `frontend/src/components/extraction/` (수정 불필요, 옵션 추가 시에만)
- 테스트: **전용 테스트 없음.** `tests/test_text_extractor.py` 신규 작성 필요 (기존 네이밍: `tests/test_*.py`, 픽스처는 `tests/fixtures/`)

## 검증된 문제 (데모로 재현 완료)

재현 스크립트로 데모 덱을 만들어 실제 출력에서 확인한 문제:

1. **읽기 순서 붕괴 (P1)**: 도형이 z-order(삽입 순서)로 출력됨. 2단 레이아웃에서 오른쪽 칼럼이 왼쪽보다 먼저 나옴. 시각적 순서 무관.
2. **제목 오선정 + 중복 (P1)**: 제목 placeholder가 없으면 "첫 텍스트박스 첫 줄"을 제목으로 사용하는데 이것이 z-order 기준이라 엉뚱한 텍스트가 제목이 됨. 또 제목 도형이 본문 루프에서 다시 순회되어 헤딩과 첫 bullet에 같은 텍스트가 중복 출력됨.
3. **표 셀 개행으로 구조 파괴 (P1)**: `_md_escape()`가 `|`만 이스케이프. 셀 내 `\n`이 그대로 출력되어 Markdown 표가 깨짐.
4. **차트 데이터 전량 소실 (P1)**: "레이블 포함" 옵션이 실제로는 차트 제목만 추출. 카테고리/시리즈/값 모두 버려짐.
5. **전부 bullet화 (P2)**: 부제목, 평문, 라벨 구분 없이 모두 `- `로 렌더링. 위계 소실.
6. **보일러플레이트 노이즈 (P2)**: 슬라이드 번호/날짜/푸터 placeholder가 매 슬라이드 텍스트로 포함됨.
7. **문서 레벨 컨텍스트 부재 (P2)**: 덱 제목(H1), 총 슬라이드 수, PPTX 섹션명 없음.
8. **이미지 alt text 미활용 (P2)**: `shape.name`("Picture 3") 사용. `descr`(대체 텍스트) 우선해야 함.
9. **노트 개행 손실 (P2)**: 다중 문단 노트가 `> NOTE: ` 한 줄로 합쳐짐.
10. **P3**: SmartArt 텍스트 통째로 누락(python-pptx 미지원, diagram XML 파싱 필요), 병합 셀 colspan 정보 소실, 문단 내 `<a:br>` 소프트 개행 무시, 빈 슬라이드 표기 없음.

### 재현 스크립트

```python
# python3 스크립트로 /tmp/demo.pptx 생성 후 추출 실행
from pptx import Presentation
from pptx.util import Inches
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE

prs = Presentation()
s1 = prs.slides.add_slide(prs.slide_layouts[1])
s1.shapes.title.text = "2026 사업 전략"
body = s1.placeholders[1].text_frame
body.text = "시장 규모 12% 성장"
p = body.add_paragraph(); p.text = "경쟁 심화"; p.level = 1
s1.notes_slide.notes_text_frame.text = "첫 문단입니다.\n둘째 문단: 발표 시 강조."

s2 = prs.slides.add_slide(prs.slide_layouts[6])  # 오른쪽을 먼저 추가 (z-order 함정)
s2.shapes.add_textbox(Inches(5.2), Inches(1.5), Inches(4), Inches(3)).text_frame.text = "[오른쪽 칼럼] 하반기 계획"
s2.shapes.add_textbox(Inches(0.5), Inches(1.5), Inches(4), Inches(3)).text_frame.text = "[왼쪽 칼럼] 상반기 실적"
s2.shapes.add_textbox(Inches(0.5), Inches(0.4), Inches(8), Inches(0.8)).text_frame.text = "연간 로드맵"

s3 = prs.slides.add_slide(prs.slide_layouts[6])
tbl = s3.shapes.add_table(3, 3, Inches(0.5), Inches(0.5), Inches(5), Inches(2)).table
for c, h in enumerate(["구분", "2025", "2026"]): tbl.cell(0, c).text = h
tbl.cell(1, 0).text = "매출\n(단위: 억)"; tbl.cell(1, 1).text = "120"; tbl.cell(1, 2).text = "150"
tbl.cell(2, 0).merge(tbl.cell(2, 2)); tbl.cell(2, 0).text = "출처: 내부 집계"
cd = CategoryChartData(); cd.categories = ["Q1", "Q2", "Q3"]
cd.add_series("매출", (30, 40, 50)); cd.add_series("이익", (5, 8, 12))
s3.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(0.5), Inches(3), Inches(5), Inches(3), cd)
prs.save("/tmp/demo.pptx")

from src.core.text_extractor import ExtractionOptions, extract_pptx_to_docs, docs_to_markdown
opts = ExtractionOptions(figures="placeholder", charts="labels", table_header=True, with_notes=True)
print(docs_to_markdown(extract_pptx_to_docs("/tmp/demo.pptx", opts), opts))
```

## 구현 가이드 (우선순위 순)

### P1-1. 읽기 순서 정렬
`_iter_shapes()`에서 도형을 `(shape.top, shape.left)` 기준으로 정렬 (None은 뒤로).
그룹 도형은 그룹 자체의 위치를 기준으로 정렬하고 내부는 재귀 정렬.
제목 fallback도 정렬된 순서에서 최상단 텍스트를 선택하도록 변경.

### P1-2. 제목 중복 제거
`extract_slide()`에서 제목으로 채택한 shape의 `shape_id`를 기억하고 블록 수집 루프에서 스킵.

### P1-3. 표 셀 개행 이스케이프
`_md_escape()`에 `\n` → `<br>` 치환 추가. (`\r` 포함 정규화)

### P1-4. 차트 데이터 추출
`charts="labels"`일 때 제목 + 데이터 표 출력:

```python
categories = [str(c) for c in chart.plots[0].categories]
for series in chart.plots[0].series:
    series.name, list(series.values)
```

출력 형태 예시:

```markdown
[Chart: 분기별 실적]
| 구분 | Q1 | Q2 | Q3 |
| --- | --- | --- | --- |
| 매출 | 30 | 40 | 50 |
| 이익 | 5 | 8 | 12 |
```

시리즈/카테고리 없는 차트는 기존 placeholder로 폴백. try/except로 감싸 실패 시 placeholder.

### P2 항목
- **bullet 지양**: placeholder 유형 확인(`shape.placeholder_format.type`). SUBTITLE은 굵게 또는 `###`, 단일 라인 텍스트박스는 평문, 목록 placeholder(BODY)만 bullet 유지.
- **노이즈 필터**: `PP_PLACEHOLDER.SLIDE_NUMBER / DATE / FOOTER` 유형 스킵. (`from pptx.enum.shapes import PP_PLACEHOLDER`)
- **문서 헤더**: 출력 최상단에 `# {core_properties.title or 1번 슬라이드 제목}` + 총 슬라이드 수. PPTX 섹션(`p14:sectionLst`, presentation.xml)이 있으면 섹션명을 `##`로, 슬라이드를 `###`로 강등하는 옵션 고려.
- **alt text**: `shape._element._nvXxPr.cNvPr.get("descr")` 우선, 없으면 `shape.name`.
- **노트**: 줄 단위로 `> ` prefix를 붙여 다중 라인 blockquote로.

### P3 항목 (여유 있을 때)
SmartArt(diagram XML 파싱 또는 `[Figure: SmartArt]` placeholder), 병합 셀 표기, `<a:br>` 개행 처리, 빈 슬라이드 `(내용 없음)` 표기, 숨김 슬라이드 제외 옵션.

## 수용 기준

재현 스크립트 출력 기준:
1. Slide 2가 "연간 로드맵 → 왼쪽 → 오른쪽" 순서로 출력되고 제목이 "연간 로드맵"
2. Slide 1 제목이 헤딩에만 1회 등장
3. Slide 3 표가 유효한 Markdown 표 (`매출<br>(단위: 억)`)
4. 차트가 카테고리/시리즈/값이 담긴 표로 출력
5. `pytest tests/ -v` 통과 (신규 `test_text_extractor.py` 포함)
6. 기존 API 응답 스키마(`markdown`, `slide_count`) 불변, 프론트 수정 없이 동작

## 주의사항

- `blocks_to_markdown`은 번역 파이프라인이 아닌 추출 전용. 번역 쪽(`ppt_parser.py`)과 무관하므로 번역 회귀 걱정 없음. 단, `api.py`가 `ExtractionOptions`를 직접 만들므로 옵션 필드 추가 시 API 파라미터와 프론트 타입(`frontend/src/types/api.ts`) 동기화 필요.
- 참고 선례: MarkItDown(microsoft/markitdown)은 차트를 표로 변환하고 alt text를 활용한다.

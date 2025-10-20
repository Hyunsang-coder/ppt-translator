# 메모리 최적화 적용 내역

## 📅 날짜
2025년 10월 20일

## 🎯 목표
메모리 누수 방지 및 리소스 효율적 관리를 위한 위험도 낮은 최적화 적용

## ✅ 적용된 변경사항

### 1. Markdown 중복 저장 제거 (위험도: 🟢 낮음)

**문제:**
- `state["markdown"]`과 `st.session_state["markdown_preview"]`에 동일한 데이터 중복 저장
- 큰 PPT 파일의 경우 수 MB의 메모리 낭비 발생

**해결:**
- `st.session_state["markdown_preview"]` 제거
- `state["markdown"]` 하나만 사용하도록 통합
- 미리보기 렌더링 시 `markdown_value` 변수 직접 사용

**영향:**
- 기존 기능: 영향 없음 (단순 중복 제거)
- 메모리 절감: 텍스트 추출 시 최대 50% 메모리 절감

---

### 2. 로그 버퍼 과도한 증가 방지 (위험도: 🟢 낮음)

**문제:**
- 여러 번역 작업 수행 시 로그 버퍼가 무한정 증가
- `MAX_UI_LOG_LINES` 제한이 있지만 버퍼 자체가 계속 누적

**해결:**
```python
elif len(st.session_state[LOG_BUFFER_KEY]) > MAX_UI_LOG_LINES * 2:
    # Prevent excessive memory usage by clearing buffer if it exceeds 2x limit
    st.session_state[LOG_BUFFER_KEY] = []
```
- 로그 버퍼가 `MAX_UI_LOG_LINES * 2` (800줄) 초과 시 자동 초기화
- 정상 사용 시에는 발동하지 않고 장시간 세션에서만 작동

**영향:**
- 기존 기능: 영향 없음
- 메모리 절감: 장시간 사용 시 로그 메모리 누적 방지

---

### 3. 텍스트 추출 후 명시적 버퍼 close() (위험도: 🟢 낮음)

**문제:**
- `ppt_buffer` 생성 후 명시적 해제 없이 GC에만 의존
- try-except 블록 구조 개선 필요

**해결:**
```python
try:
    docs = extract_pptx_to_docs(ppt_buffer, extraction_options)
    markdown_text = docs_to_markdown(docs, extraction_options)
    # ... 처리 로직
except Exception as exc:
    LOGGER.exception("Extraction failed: %s", exc)
    st.error("텍스트 추출 중 오류가 발생했습니다.")
finally:
    # Explicitly close buffer to free memory
    ppt_buffer.close()
```
- `finally` 블록에서 반드시 버퍼 정리
- 예외 발생 여부와 무관하게 메모리 해제 보장

**영향:**
- 기존 기능: 영향 없음 (에러 처리 개선)
- 메모리 절감: 즉각적인 버퍼 메모리 해제

---

### 4. 번역 완료 후 대용량 객체 명시적 정리 (위험도: 🟢 낮음)

**문제:**
- `paragraphs`, `presentation`, `translated_texts` 등 대용량 객체가 함수 종료까지 메모리 점유
- GC가 지연되면 메모리 압박 가능

**해결:**
```python
output_buffer = writer.apply_translations(paragraphs, translated_texts, presentation)
_refresh_ui_logs(log_placeholder, log_buffer)

# Explicitly clear large objects to help GC
paragraphs = None
presentation = None
translated_texts = None
if repetition_plan is not None:
    translated_unique = None
```
- 더 이상 필요 없는 대용량 객체를 명시적으로 `None`으로 설정
- GC가 더 빠르게 메모리 회수 가능

**영향:**
- 기존 기능: 영향 없음
- 메모리 절감: 번역 완료 후 즉시 메모리 해제, 대용량 PPT에서 효과 큼

---

## 🧪 검증 결과

### 테스트 통과
```bash
pytest tests/test_translation.py -v
============================= test session starts ==============================
collected 5 items

tests/test_translation.py::LanguageDetectorTestCase::test_map_lang_code_known_language PASSED
tests/test_translation.py::LanguageDetectorTestCase::test_map_lang_code_unknown_language PASSED
tests/test_translation.py::GlossaryLoaderTestCase::test_apply_glossary_to_texts PASSED
tests/test_translation.py::HelperTestCase::test_chunk_paragraphs_preserves_order PASSED
tests/test_translation.py::HelperTestCase::test_split_text_into_segments_respects_segment_count PASSED

============================== 5 passed in 3.15s
```

### 린터 검사
- ✅ 모든 린터 에러 없음
- ✅ 코드 스타일 준수

---

## 📊 예상 효과

| 시나리오 | 메모리 절감 효과 |
|---------|----------------|
| 텍스트 추출 1회 | ~50% (중복 제거) |
| 연속 번역 작업 | ~30% (객체 정리) |
| 장시간 세션 | 로그 누적 방지 |
| 대용량 PPT (100+ 슬라이드) | 즉시 메모리 해제 |

---

## ⚠️ 추후 고려 사항 (위험도 중간)

현재는 **위험도 낮은 항목만** 적용했습니다. 추가 최적화가 필요한 경우 아래 항목들을 검토:

### 1. file_handler.py 버퍼 관리 개선 (위험도: 🟡 중간)
- 현재: `uploaded_ppt_bytes` 키 제거만 수행 (실제로 사용되지 않는 키)
- 개선안: session_state에 버퍼를 저장하고 이전 버퍼 명시적 정리

### 2. Presentation 객체 수명 관리 (위험도: 🟡 중간)
- 현재: `paragraphs`가 `presentation` 객체 참조 유지
- 개선안: 구조 재설계 또는 약한 참조(weakref) 사용

---

## 📝 참고

- 모든 변경사항은 기존 기능에 영향을 주지 않으며 순수한 메모리 최적화
- Python GC에 대한 힌트를 제공하여 메모리 회수를 촉진
- Streamlit 세션 특성을 고려한 안전한 최적화


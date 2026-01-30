# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPT 번역캣 is a PowerPoint translation application using LangChain with OpenAI GPT and Anthropic Claude models. It provides both a Streamlit web UI and FastAPI REST API. Features include slide text translation while preserving original formatting, glossary support, auto language detection, and detailed progress tracking.

## Development Commands

```bash
# Run the Streamlit app
streamlit run app.py

# Run the FastAPI server
uvicorn api:app --reload --port 8000

# Run all tests
pytest tests/ -v

# Run a specific test
pytest tests/test_translation.py::HelperTestCase::test_chunk_paragraphs_preserves_order -v

# Run slow tests (require API calls)
pytest tests/ -v -m slow
```

## Environment Setup

1. Create `.env` from `.env.example` and set API keys:
   - `OPENAI_API_KEY` - Required for OpenAI models (GPT)
   - `ANTHROPIC_API_KEY` - Required for Anthropic models (Claude)
2. Optional environment variables for tuning:
   - `TRANSLATION_MAX_CONCURRENCY` (default: 8)
   - `TRANSLATION_BATCH_SIZE` (default: 80)
   - `TRANSLATION_MIN_BATCH_SIZE` / `TRANSLATION_MAX_BATCH_SIZE`
   - `TRANSLATION_TPM_LIMIT` (default: 30000)

## Architecture

### Entry Points
- `app.py`: Streamlit UI orchestrating two workflows:
  - PPT Translation (main feature)
  - Text Extraction (PPT → Markdown)
- `api.py`: FastAPI REST API server
  - `GET /health`: Health check endpoint
  - `POST /translate`: PPT translation endpoint (multipart/form-data)
  - Supports AWS Lambda deployment via Mangum

### Core Components (`src/core/`)
- `ppt_parser.py`: Extracts `ParagraphInfo` objects from PPTX (handles shapes, tables, groups)
- `ppt_writer.py`: Applies translations back using run-based text distribution to preserve formatting
- `text_extractor.py`: Converts PPTX to structured markdown

### Service Layer (`src/services/`)
- `models.py`: Data models (`TranslationRequest`, `TranslationResult`, `TranslationProgress`, `TranslationStatus`)
- `translation_service.py`: `TranslationService` class encapsulating translation workflow logic
  - Shared by both Streamlit (`app.py`) and FastAPI (`api.py`)
  - Progress callback support for real-time updates

### Translation Chain (`src/chains/`)
- `llm_factory.py`: Factory for creating LLM instances (OpenAI/Anthropic) with provider-specific configuration
- `translation_chain.py`: LangChain pipeline with batch retry logic via tenacity, concurrent execution with wave-based batching
- `context_manager.py`: Builds global presentation context for consistent translations

### Utilities (`src/utils/`)
- `config.py`: Settings dataclass loaded from environment
- `glossary_loader.py`: Excel glossary loading and term substitution (pre/post translation)
- `language_detector.py`: Uses langdetect with Korean↔English inference rules
- `repetition.py`: Deduplicates repeated phrases to reduce API calls
- `helpers.py`: Batch chunking, text segmentation for run distribution
- `security.py`: File validation, filename sanitization, HTML content escaping

### UI Components (`src/ui/`)
- `progress_tracker.py`: Streamlit progress bar and log updates
- `settings_panel.py`: Translation settings sidebar
- `file_handler.py`: Upload validation and buffer management
- `extraction_settings.py`: Text extraction options UI

### Tests (`tests/`)
- `test_translation.py`: Unit tests for translation helpers and language detection

## Key Patterns

### Translation Flow
1. `PPTParser.extract_paragraphs()` → `List[ParagraphInfo]` + `Presentation`
2. `ContextManager.build_global_context()` for cross-slide consistency
3. Optional: `build_repetition_plan()` to deduplicate identical text
4. `chunk_paragraphs()` creates batches with context/glossary
5. `translate_with_progress()` handles concurrent API calls
6. `expand_translations()` maps unique results back to duplicates
7. `PPTWriter.apply_translations()` writes back preserving run formatting

### Formatting Preservation
`PPTWriter` uses `split_text_into_segments()` with character-length weights to distribute translated text across original runs, preserving bold/italic/font styling.

### Error Handling
- Translation chain uses tenacity with exponential backoff (3 attempts)
- JSON parsing falls back to `|||` delimiter then newline splitting
- Futures cleanup with timeout for graceful shutdown on errors

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Translate PPT with OpenAI
curl -X POST http://localhost:8000/translate \
  -F "ppt_file=@presentation.pptx" \
  -F "target_lang=한국어" \
  -F "provider=openai" \
  -F "model=gpt-5.2" \
  -o translated.pptx

# Translate PPT with Anthropic Claude
curl -X POST http://localhost:8000/translate \
  -F "ppt_file=@presentation.pptx" \
  -F "target_lang=한국어" \
  -F "provider=anthropic" \
  -F "model=claude-sonnet-4-5-20250929" \
  -o translated.pptx
```

## Libraries
- **LangChain**: Translation chain with `ChatOpenAI`/`ChatAnthropic` and `PromptTemplate`
- **langchain-anthropic**: Anthropic Claude model integration
- **python-pptx**: PPTX parsing and writing
- **Streamlit**: Web UI with session state management
- **FastAPI**: REST API server with automatic OpenAPI docs
- **Mangum**: AWS Lambda adapter for FastAPI
- **tenacity**: Retry logic with exponential backoff
- **langdetect**: Automatic language detection

## Claude Code Customization Best Practices

### Skills
커스텀 슬래시 명령어 정의 시 준수사항:
- **명확한 트리거**: `/command` 형식의 직관적 명령어명 사용
- **단일 책임**: 하나의 skill은 하나의 명확한 작업만 수행
- **문서화**: 각 skill의 목적, 입력, 출력을 명시
- **에러 핸들링**: 실패 시 명확한 피드백 제공
- **멱등성**: 동일 입력에 동일 결과 보장

```markdown
# Example skill definition in .claude/skills/
name: translate-slide
description: Translate a single slide with glossary support
trigger: /translate-slide
```

### Subagents (Task Tool)
전문화된 에이전트 위임 시 준수사항:
- **적절한 agent type 선택**:
  - `Explore`: 코드베이스 탐색, 파일 검색
  - `Plan`: 구현 전략 설계
  - `Bash`: Git, 빌드, 시스템 명령
  - `general-purpose`: 복합 멀티스텝 작업
- **명확한 프롬프트**: 컨텍스트와 기대 결과를 구체적으로 명시
- **병렬 실행**: 독립적인 작업은 동시에 여러 Task 호출
- **결과 검증**: 서브에이전트 결과를 맹신하지 않고 검증

```python
# Parallel task example
Task(subagent_type="Explore", prompt="Find all translation-related files")
Task(subagent_type="Explore", prompt="Find all test files")  # 병렬 호출
```

### Hooks
이벤트 기반 자동화 시 준수사항:
- **최소 권한**: 필요한 작업만 수행, 과도한 권한 요청 금지
- **빠른 실행**: 훅은 빠르게 완료되어야 함 (blocking 최소화)
- **실패 허용**: 훅 실패가 전체 워크플로우를 중단시키지 않도록 설계
- **로깅**: 디버깅을 위한 적절한 로그 출력
- **조건부 실행**: 불필요한 실행을 피하기 위한 조건 검사

```json
// .claude/settings.json hook example
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "command": "echo 'File modification detected'"
      }
    ]
  }
}
```

### 공통 원칙
1. **보안 우선**: 민감 정보(API 키, 자격 증명) 노출 금지
2. **테스트 가능성**: 커스텀 확장은 독립적으로 테스트 가능하게 설계
3. **버전 관리**: `.claude/` 설정 파일은 git으로 관리
4. **점진적 개선**: 작게 시작하여 필요에 따라 확장
5. **문서화**: 팀원이 이해할 수 있도록 충분한 주석과 README 제공

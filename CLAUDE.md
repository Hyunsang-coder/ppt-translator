# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PPT 번역캣 is a Streamlit-based PowerPoint translation prototype using LangChain and OpenAI GPT models. It translates slide text while preserving original formatting, supports glossaries, auto language detection, and provides detailed progress tracking.

## Development Commands

```bash
# Run the app
streamlit run app.py

# Run all tests
python -m unittest tests/test_translation.py
# Or with pytest
pytest tests/ -v

# Run a specific test
pytest tests/test_translation.py::HelperTestCase::test_chunk_paragraphs_preserves_order -v

# Run slow tests (require API calls)
pytest tests/ -v -m slow
```

## Environment Setup

1. Create `.env` from `.env.example` and set `OPENAI_API_KEY`
2. Optional environment variables for tuning:
   - `TRANSLATION_MAX_CONCURRENCY` (default: 8)
   - `TRANSLATION_BATCH_SIZE` (default: 80)
   - `TRANSLATION_MIN_BATCH_SIZE` / `TRANSLATION_MAX_BATCH_SIZE`
   - `TRANSLATION_TPM_LIMIT` (default: 30000)

## Architecture

### Entry Point
- `app.py`: Streamlit UI orchestrating three workflows:
  - PPT Translation (main feature)
  - Text Extraction (PPT → Markdown)
  - PDF → PPT Conversion (Vision API-based)

### Core Components (`src/core/`)
- `ppt_parser.py`: Extracts `ParagraphInfo` objects from PPTX (handles shapes, tables, groups)
- `ppt_writer.py`: Applies translations back using run-based text distribution to preserve formatting
- `text_extractor.py`: Converts PPTX to structured markdown
- `pdf_processor.py`: Uses OpenAI Vision for semantic layout analysis of PDFs
- `pdf_to_ppt_writer.py`: Creates PPTX from Vision OCR results with precise positioning

### Translation Chain (`src/chains/`)
- `translation_chain.py`: LangChain pipeline with `ChatOpenAI`, batch retry logic via tenacity, concurrent execution with wave-based batching
- `context_manager.py`: Builds global presentation context for consistent translations

### Utilities (`src/utils/`)
- `config.py`: Settings dataclass loaded from environment
- `glossary_loader.py`: Excel glossary loading and term substitution (pre/post translation)
- `language_detector.py`: Uses langdetect with Korean↔English inference rules
- `repetition.py`: Deduplicates repeated phrases to reduce API calls
- `helpers.py`: Batch chunking, text segmentation for run distribution

### UI Components (`src/ui/`)
- `progress_tracker.py`: Streamlit progress bar and log updates
- `settings_panel.py`: Translation settings sidebar
- `file_handler.py`: Upload validation and buffer management

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

## Libraries
- **LangChain**: Translation chain with `ChatOpenAI` and `PromptTemplate`
- **python-pptx**: PPTX parsing and writing
- **Streamlit**: Web UI with session state management
- **PyMuPDF**: PDF to image conversion for Vision processing

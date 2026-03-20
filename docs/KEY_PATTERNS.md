# Key Patterns

## Translation Flow
0. Optional: `compress_pptx_images()` pre-compresses images at ZIP level
1. `PPTParser.extract_paragraphs()` → `List[ParagraphInfo]` + `Presentation`
2. `ContextManager.build_global_context()` for cross-slide consistency
3. Optional: `build_repetition_plan()` to deduplicate identical text
4. `chunk_paragraphs()` creates batches with context/glossary
5. `translate_with_progress()` → `_batch_translate_with_retry()` with tenacity retry
6. `expand_translations()` maps unique results back to duplicates
7. `_fix_color_distributions()` preserves multi-color formatting via LLM
8. `PPTWriter.apply_translations()` writes back preserving formatting, applies text fit

## Async Job Flow (FastAPI + Next.js)
1. Frontend `POST /api/v1/jobs` with file, settings, filename_settings, compress_images, length_limit
2. Backend `try_create_job()` atomic admission (429 if full)
3. Returns `job_id`; job waits on `running_semaphore`
4. Frontend polls `GET /api/v1/jobs/{job_id}` (2s interval)
5. Backend tracks progress: `started`, `progress`, `complete`, `error`, `cancelled`
6. Frontend downloads via `GET /api/v1/jobs/{job_id}/result`
7. Output filename generated server-side from `FilenameSettings`

## Formatting Preservation
`PPTWriter` uses `split_text_into_segments()` with character-length weights to distribute text across original runs (bold/italic/font). For multi-color paragraphs, `color_distribution_chain` uses LLM to map translated segments to original format groups.

## Text Fit
- `none`: No adjustment
- `auto_shrink`: Reduce font size (down to `min_font_ratio`%)
- `expand_box`: Widen text box (max 30%, skips rotated/grouped/table). Preserves original auto_size
- `shrink_then_expand`: Shrink first, then expand if needed

Width expansion applied before text fit. Font sizes rounded to nearest whole point (12700 EMU).
Placeholder shapes: all 4 positional attributes (left/top/width/height) must be materialised before modifying width to prevent python-pptx xfrm cy=0 collapse.

## Progress Tracking
Monotonic `TranslationProgress.percent`:
- Parsing 2% → Language detection 5% → Batch prep 8% → Translation 10–80% → Color fix 80–90% → Apply 95% → Complete 100%

## Error Handling
- tenacity: 3 attempts, 2–10s exponential backoff
- Structured output via Pydantic (no JSON fallback)
- Fail-fast batch validation: missing results → `RuntimeError` → tenacity retries
- `asyncio.CancelledError` caught separately from `Exception`
- Frontend: API call-time error handling, no pre-flight health checks

## Thread Safety
- Translation in thread pool (`run_in_executor`), SSE/job state on asyncio event-loop
- `Job.add_event()`: `call_soon_threadsafe` bridges worker→event-loop
- `Job._state_lock`: protects terminal state transitions
- `ServiceProgressTracker.reset()`: prevents counter accumulation across retries

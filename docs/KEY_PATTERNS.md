# Key Patterns

## Translation Flow
0. Optional: `compress_pptx_images()` pre-compresses images at ZIP level
1. `PPTParser.extract_paragraphs()` → `List[ParagraphInfo]` + `Presentation`
2. `ContextManager.build_global_context()` for cross-slide consistency
3. Optional: `build_repetition_plan()` to deduplicate identical text
4. `chunk_paragraphs()` creates batches with context/glossary
5. `translate_with_progress()` → `_batch_translate_with_retry()` with tenacity retry
6. `expand_translations()` maps unique results back to duplicates
7. Glossary post-processing finalizes the aligned target strings
8. `_fix_color_distributions()` maps those final strings onto source style groups without translating them a second time
9. `run_sweep()` checks the actual final strings shown in review
10. `PPTWriter.apply_translations()` writes back preserving formatting and applies text fit

## Async Job Flow (FastAPI + Next.js)
1. Frontend `POST /api/v1/jobs` with file, settings, filename_settings, compress_images, length_limit
2. Backend `try_create_job()` atomic admission (429 if full)
3. Returns `job_id`; job waits on `running_semaphore`
4. Frontend polls `GET /api/v1/jobs/{job_id}` (2s interval)
5. Backend tracks progress: `started`, `progress`, `complete`, `error`, `cancelled`
6. Completed jobs open the review draft; proposal endpoints generate edit/retranslation candidates without mutation
7. Applying a proposal changes only the versioned draft; undo and partial propagation are revision-checked
8. `POST /review/commit` renders once from pristine source bytes and atomically publishes the output
9. Frontend downloads via `GET /api/v1/jobs/{job_id}/result`
10. Output filename generated server-side from `FilenameSettings`

## Formatting Preservation
Uniform paragraphs keep the first original run's formatting. Multi-style paragraphs keep the main translation unchanged and use `color_distribution_chain.distribute_colors()` to assign semantic `ColoredSegment` spans to immutable source style-group IDs. Responses carry stable `item_id` values so a missing model item cannot shift mappings onto another paragraph. The writer applies colors only when validated segments concatenate exactly to the translation. If mapping fails, it uses one neutral/base style and exposes a review warning instead of position-based splitting.

Review sessions retain pristine source PPTX bytes. Every committed render reparses those bytes before applying the draft, so reordered color spans and text fitting cannot drift across repeated edits. Direct edits and retranslations are previewed with styled spans before they are staged; the published file changes only at review commit.

## Color Mapping Audit
Use `scripts/evaluate_color_mapping.py SOURCE.pptx TARGET.pptx` to review translated decks for suspicious highlight/style carryover. The audit is heuristic: `position_like_highlight` means the highlighted source and target runs occupy similar relative positions and should be reviewed for accidental position-based mapping. `dropped_highlight` is lower risk because the target avoided a potentially wrong emphasis.

Add `--json` when the findings need to be consumed by another script.

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
- `Job.review_lock`: serializes proposal apply, undo, partial propagation, and final commit
- Review revision checks reject stale browser proposals with HTTP 409
- `ServiceProgressTracker.reset()`: prevents counter accumulation across retries

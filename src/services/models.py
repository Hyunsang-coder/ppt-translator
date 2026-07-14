"""Data models for the translation service."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Model registry — single source of truth for supported LLM models.
#
# Each entry is (id, display_name). The order here is the order shown to users.
# api.py builds its Pydantic ModelInfo list from this, llm_factory derives its
# allowlists from it, and default model IDs below reference it — so adding or
# bumping a model is a one-file change.
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, list[tuple[str, str]]] = {
    "openai": [
        ("gpt-5.6-sol", "GPT-5.6 Sol (High)"),
        ("gpt-5.6-luna", "GPT-5.6 Luna (High)"),
    ],
    "anthropic": [
        ("claude-opus-4-8", "Claude Opus 4.8"),
        ("claude-sonnet-5", "Claude Sonnet 5"),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5"),
    ],
}

# Default models referenced by request schemas and endpoints.
DEFAULT_TRANSLATION_MODEL = "claude-sonnet-5"
DEFAULT_LIGHT_MODEL = {
    "openai": "gpt-5.6-luna",
    "anthropic": "claude-haiku-4-5-20251001",
}

# GPT-5.6 reasoning effort is configured per model so all OpenAI call paths
# (translation, summarization, and post-processing) use the same setting.
MODEL_REASONING_EFFORT: Dict[str, str] = {
    "gpt-5.6-sol": "high",
    "gpt-5.6-luna": "high",
}


def model_ids(provider: str) -> list[str]:
    """Return the allowed model ids for a provider (empty if unknown)."""
    return [model_id for model_id, _ in MODEL_REGISTRY.get(provider, [])]


class TextFitMode(Enum):
    """Text fitting mode for translated text boxes."""

    NONE = "none"
    AUTO_SHRINK = "auto_shrink"
    EXPAND_BOX = "expand_box"
    SHRINK_THEN_EXPAND = "shrink_then_expand"


class TranslationStatus(Enum):
    """Translation workflow status."""

    PENDING = "pending"
    PARSING = "parsing"
    DETECTING_LANGUAGE = "detecting_language"
    PREPARING_BATCHES = "preparing_batches"
    TRANSLATING = "translating"
    FIXING_COLORS = "fixing_colors"
    APPLYING_TRANSLATIONS = "applying_translations"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TranslationRequest:
    """Request model for PPT translation."""

    ppt_file: io.BytesIO
    source_lang: str = "Auto"
    target_lang: str = "Auto"
    model: str = DEFAULT_TRANSLATION_MODEL
    provider: str = "anthropic"
    context: Optional[str] = None
    instructions: Optional[str] = None
    glossary: Optional[Dict[str, str]] = None
    preprocess_repetitions: bool = False
    translate_notes: bool = False
    text_fit_mode: TextFitMode = TextFitMode.NONE
    min_font_ratio: int = 80
    length_limit: Optional[int] = None
    # Parsed team translation-rules document (WP-C1). None -> feature off.
    team_rules: Optional[Dict[str, Any]] = None
    # C-1: cooperative cancellation flag. When set mid-run, the translation
    # loop stops at the next batch boundary and raises. None -> not cancellable.
    cancel_event: Optional[Any] = None  # threading.Event


@dataclass
class TranslationResult:
    """Result model for PPT translation."""

    success: bool
    output_file: Optional[io.BytesIO] = None
    error_message: Optional[str] = None
    source_language_detected: str = ""
    target_language_used: str = ""
    total_paragraphs: int = 0
    unique_paragraphs: int = 0
    batch_count: int = 0
    elapsed_seconds: float = 0.0
    # Consistency-sweep findings (WP-C3). List[consistency_sweep.Finding].
    findings: List[Any] = field(default_factory=list)
    # In-memory review/edit session (WP-C5). review_session.ReviewSession | None.
    # Holds the live presentation + fragments for post-translation editing.
    review_session: Optional[Any] = None


@dataclass
class TranslationProgress:
    """Progress update model for translation workflow."""

    status: TranslationStatus
    current_batch: int = 0
    total_batches: int = 0
    current_sentence: int = 0
    total_sentences: int = 0
    percent: int = 0
    message: str = ""
    details: Dict[str, object] = field(default_factory=dict)


# Type alias for progress callback function
ProgressCallback = Callable[[TranslationProgress], None]

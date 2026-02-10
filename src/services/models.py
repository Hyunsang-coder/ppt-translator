"""Data models for the translation service."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Optional


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
    model: str = "gpt-5.2"
    provider: str = "openai"
    context: Optional[str] = None
    instructions: Optional[str] = None
    glossary: Optional[Dict[str, str]] = None
    preprocess_repetitions: bool = False
    translate_notes: bool = False
    text_fit_mode: TextFitMode = TextFitMode.NONE
    min_font_ratio: int = 80
    length_limit: Optional[int] = None


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

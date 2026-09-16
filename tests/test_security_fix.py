"""Tests for security.py XSS sanitization fix."""

from __future__ import annotations

import io
import math
import zipfile

from PIL import Image

from src.chains.llm_factory import create_rate_limiter
from src.utils.security import (
    MAX_IMAGE_PIXELS,
    sanitize_html_content,
    validate_pptx_file,
)


def _office_zip(*extra_entries: tuple[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", b"types")
        archive.writestr("ppt/presentation.xml", b"presentation")
        for name, content in extra_entries:
            archive.writestr(name, content)
    buffer.seek(0)
    return buffer


class TestOfficeArchiveValidation:
    """ZIP metadata and media limits must be enforced before parsing."""

    def test_rejects_path_traversal_entry(self):
        valid, message = validate_pptx_file(_office_zip(("../escape.txt", b"x")))
        assert not valid
        assert message

    def test_rejects_high_compression_ratio(self):
        valid, message = validate_pptx_file(
            _office_zip(("ppt/slides/slide1.xml", b"0" * 1_000_000))
        )
        assert not valid
        assert message

    def test_rejects_oversized_raster_image(self):
        image_buffer = io.BytesIO()
        # Just over the cap (1-bit PNG stays tiny on disk and in RAM).
        side = int(math.sqrt(MAX_IMAGE_PIXELS)) + 10
        Image.new("1", (side, side)).save(image_buffer, format="PNG")
        valid, message = validate_pptx_file(
            _office_zip(("ppt/media/oversized.png", image_buffer.getvalue()))
        )
        assert not valid
        assert message


def test_rate_limiter_is_shared_per_provider():
    """Models from one provider must consume one shared limiter budget."""
    create_rate_limiter.cache_clear()
    try:
        assert create_rate_limiter("openai") is create_rate_limiter("openai")
        assert create_rate_limiter("openai") is not create_rate_limiter("anthropic")
    finally:
        create_rate_limiter.cache_clear()


class TestSanitizeHtmlContent:
    """Tests that XSS protection works correctly."""

    def test_escapes_script_tags(self):
        """Script tags should be escaped by html.escape."""
        result = sanitize_html_content('<script>alert("xss")</script>')
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_escapes_event_handlers(self):
        """Event handlers in HTML attributes should be escaped."""
        result = sanitize_html_content('<img onerror="alert(1)">')
        assert "onerror" not in result or "&lt;" in result

    def test_escapes_javascript_protocol(self):
        """javascript: in an HTML tag is safe because the tag itself is escaped."""
        result = sanitize_html_content('<a href="javascript:alert(1)">click</a>')
        # The <a> tag is fully escaped, so javascript: can't execute
        assert "&lt;a" in result
        assert "<a " not in result

    def test_preserves_normal_text(self):
        """Normal text without HTML should be preserved."""
        result = sanitize_html_content("Hello World 2024")
        assert result == "Hello World 2024"

    def test_preserves_ampersands_in_text(self):
        """Ampersands should be escaped but readable."""
        result = sanitize_html_content("AT&T revenue $5B")
        assert "&amp;" in result

    def test_truncation(self):
        """Text exceeding max_length should be truncated."""
        long_text = "a" * 20000
        result = sanitize_html_content(long_text, max_length=100)
        assert len(result) <= 104  # 100 + "..."

    def test_empty_input(self):
        """Empty string should return empty string."""
        assert sanitize_html_content("") == ""

    def test_none_like_empty(self):
        """None-ish input should return empty string."""
        assert sanitize_html_content("") == ""

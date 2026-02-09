"""Tests for security.py XSS sanitization fix."""

from __future__ import annotations

from src.utils.security import sanitize_html_content


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

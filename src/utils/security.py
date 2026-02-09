"""Security utilities for file validation and XSS prevention."""

from __future__ import annotations

import io
import logging
from typing import Optional

LOGGER = logging.getLogger(__name__)

# File signature magic bytes
PPTX_SIGNATURE = b"PK\x03\x04"  # ZIP-based format (PPTX)
PPT_OLD_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE2 format (PPT)
XLSX_SIGNATURE = b"PK\x03\x04"  # ZIP-based format (Excel 2007+)
XLS_OLD_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE2 format (Excel 97-2003)

# Maximum file name length (Windows: 255, Unix: 255, but we'll be conservative)
MAX_FILENAME_LENGTH = 200


def validate_pptx_file(file_buffer: io.BytesIO) -> tuple[bool, Optional[str]]:
    """Validate PPTX file by checking file signature.

    Args:
        file_buffer: File buffer to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        file_buffer.seek(0)
        header = file_buffer.read(8)
        file_buffer.seek(0)

        # PPTX files are ZIP archives (Office Open XML)
        if header.startswith(PPTX_SIGNATURE):
            return True, None

        # Old PPT format (OLE2)
        if header.startswith(PPT_OLD_SIGNATURE):
            return True, None

        return False, "파일 형식이 올바르지 않습니다. PPT 또는 PPTX 파일만 업로드 가능합니다."
    except Exception as exc:
        LOGGER.error("File validation error: %s", exc)
        return False, "파일을 읽는 중 오류가 발생했습니다."


def validate_excel_file(file_buffer: io.BytesIO) -> tuple[bool, Optional[str]]:
    """Validate Excel file by checking file signature.

    Args:
        file_buffer: File buffer to validate.

    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        file_buffer.seek(0)
        header = file_buffer.read(8)
        file_buffer.seek(0)

        # XLSX files are ZIP archives (Office Open XML)
        if header.startswith(XLSX_SIGNATURE):
            return True, None

        # Old XLS format (OLE2)
        if header.startswith(XLS_OLD_SIGNATURE):
            return True, None

        return False, "파일 형식이 올바르지 않습니다. Excel 파일(xlsx, xls)만 업로드 가능합니다."
    except Exception as exc:
        LOGGER.error("File validation error: %s", exc)
        return False, "파일을 읽는 중 오류가 발생했습니다."


def sanitize_filename(filename: str, max_length: int = MAX_FILENAME_LENGTH, fallback: str = "file") -> str:
    """Sanitize filename by removing dangerous characters and limiting length.

    Args:
        filename: Original filename.
        max_length: Maximum allowed filename length.
        fallback: Fallback name if sanitization results in empty string.

    Returns:
        Sanitized filename.
    """
    if not filename:
        return fallback

    # Remove path separators and dangerous characters, keep spaces
    sanitized = "".join(ch for ch in filename if ch.isalnum() or ch in ("-", "_", ".", " "))
    # Collapse multiple spaces
    sanitized = " ".join(sanitized.split())
    
    # Limit length
    if len(sanitized) > max_length:
        # Keep extension if exists
        if "." in sanitized:
            name_part, ext_part = sanitized.rsplit(".", 1)
            max_name_length = max_length - len(ext_part) - 1
            if max_name_length > 0:
                sanitized = name_part[:max_name_length] + "." + ext_part
            else:
                sanitized = sanitized[:max_length]
        else:
            sanitized = sanitized[:max_length]
    
    return sanitized or fallback


def sanitize_html_content(text: str, max_length: int = 10000) -> str:
    """Sanitize HTML content to prevent XSS attacks.

    Args:
        text: Text content to sanitize.
        max_length: Maximum allowed text length.

    Returns:
        Sanitized text safe for HTML rendering.
    """
    import html

    if not text:
        return ""

    # Limit length first to prevent DoS
    if len(text) > max_length:
        text = text[:max_length] + "..."

    # Escape HTML special characters
    escaped = html.escape(text, quote=True)

    # Remove any remaining script tags and event handlers (defense in depth)
    import re
    escaped = re.sub(r"<script[^>]*>.*?</script>", "", escaped, flags=re.IGNORECASE | re.DOTALL)
    escaped = re.sub(r"javascript:", "", escaped, flags=re.IGNORECASE)
    escaped = re.sub(r"on\w+\s*=", "", escaped, flags=re.IGNORECASE)

    return escaped


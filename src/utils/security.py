"""Security utilities for file validation and XSS prevention."""

from __future__ import annotations

import html
import io
import logging
import os
import stat
import zipfile
from typing import Optional

from PIL import Image

LOGGER = logging.getLogger(__name__)

# File signature magic bytes
PPTX_SIGNATURE = b"PK\x03\x04"  # ZIP-based format (PPTX)
PPT_OLD_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE2 format (PPT)
XLSX_SIGNATURE = b"PK\x03\x04"  # ZIP-based format (Excel 2007+)
XLS_OLD_SIGNATURE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # OLE2 format (Excel 97-2003)

# Maximum file name length (Windows: 255, Unix: 255, but we'll be conservative)
MAX_FILENAME_LENGTH = 200

# ZIP metadata is attacker-controlled even when the uploaded file has a valid
# PPTX signature. Keep these limits shared by every code path that opens an
# Office Open XML archive so a small ZIP cannot expand without bound.
# Sized for ~1 GiB image-heavy decks; the ratio/pixel checks below remain the
# real bomb guards, not these totals.
MAX_ZIP_ENTRIES = 10_000
MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 200
MAX_IMAGE_PIXELS = 100_000_000
_RASTER_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"})


def validate_zip_archive(
    file_buffer: io.BytesIO,
    *,
    required_entries: tuple[str, ...] = (),
    validate_media_images: bool = False,
) -> tuple[bool, Optional[str]]:
    """Validate ZIP metadata before any entry is decompressed.

    This is intentionally a metadata-first check: callers must not pass an
    untrusted Office archive to ``python-pptx`` or Pillow before this function
    accepts it.
    """
    try:
        file_buffer.seek(0)
        if not zipfile.is_zipfile(file_buffer):
            return False, "압축 파일 형식이 올바르지 않습니다."
        file_buffer.seek(0)

        with zipfile.ZipFile(file_buffer, "r") as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ZIP_ENTRIES:
                return False, f"압축 파일 항목 수가 {MAX_ZIP_ENTRIES}개를 초과합니다."

            names: set[str] = set()
            total_uncompressed = 0
            for entry in entries:
                name = entry.filename
                normalized_name = name.replace("\\", "/")
                parts = normalized_name.split("/")
                normalized_key = normalized_name.casefold()

                # ZIP entries must never be usable as filesystem paths. Reject
                # absolute, traversal, NUL-containing, and duplicate names.
                if (
                    not name
                    or "\x00" in name
                    or normalized_name.startswith("/")
                    or normalized_name.startswith("../")
                    or "/../" in normalized_name
                    or normalized_name.endswith("/..")
                    or ".." in parts
                    or normalized_key in names
                ):
                    return False, "압축 파일 내부 경로가 안전하지 않습니다."
                names.add(normalized_key)

                mode = (entry.external_attr >> 16) & 0xFFFF
                if stat.S_ISLNK(mode):
                    return False, "압축 파일에 심볼릭 링크가 포함되어 있습니다."
                if entry.flag_bits & 0x1:
                    return False, "암호화된 압축 파일은 처리할 수 없습니다."
                if entry.file_size < 0 or entry.file_size > MAX_ZIP_ENTRY_UNCOMPRESSED_BYTES:
                    return False, "압축 파일의 개별 항목 크기가 제한을 초과합니다."

                total_uncompressed += entry.file_size
                if total_uncompressed > MAX_ZIP_TOTAL_UNCOMPRESSED_BYTES:
                    return False, "압축 파일의 전체 전개 크기가 제한을 초과합니다."

                if entry.file_size:
                    if not entry.compress_size:
                        return False, "압축 파일의 압축비가 비정상적입니다."
                    if entry.file_size / entry.compress_size > MAX_ZIP_COMPRESSION_RATIO:
                        return False, "압축 파일의 압축비가 제한을 초과합니다."

            missing = [required for required in required_entries if required.casefold() not in names]
            if missing:
                return False, "필수 Office 파일이 누락되었습니다."

            if validate_media_images:
                for entry in entries:
                    _, extension = os.path.splitext(entry.filename.lower())
                    if not (
                        entry.filename.lower().startswith("ppt/media/")
                        and extension in _RASTER_IMAGE_EXTENSIONS
                        and entry.file_size
                    ):
                        continue
                    try:
                        with archive.open(entry, "r") as image_stream:
                            image = Image.open(image_stream)
                            width, height = image.size
                    except Exception:
                        # The normal PPTX parser will handle an unsupported or
                        # corrupt media entry; do not add another full decode.
                        continue
                    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                        return False, "PPTX 이미지의 픽셀 수가 제한을 초과합니다."

        file_buffer.seek(0)
        return True, None
    except (OSError, ValueError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        LOGGER.warning("Unsafe ZIP archive rejected: %s", exc)
        file_buffer.seek(0)
        return False, "압축 파일을 안전하게 검증할 수 없습니다."


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

        # PPTX files are ZIP archives (Office Open XML). Check ZIP metadata
        # before any library is allowed to decompress an entry.
        if header.startswith(b"PK"):
            return validate_zip_archive(
                file_buffer,
                required_entries=("[Content_Types].xml", "ppt/presentation.xml"),
                validate_media_images=True,
            )

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

        # XLSX files are ZIP archives (Office Open XML).
        if header.startswith(b"PK"):
            return validate_zip_archive(
                file_buffer,
                required_entries=("[Content_Types].xml", "xl/workbook.xml"),
            )

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
    if not text:
        return ""

    # Limit length first to prevent DoS
    if len(text) > max_length:
        text = text[:max_length] + "..."

    # Escape HTML special characters — sufficient for XSS prevention
    # since all < > " ' & are converted to HTML entities.
    return html.escape(text, quote=True)

"""PPTX image compression utility.

Compresses images inside PPTX files at the ZIP level (before python-pptx parsing)
to reduce memory usage on resource-constrained servers.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path
from typing import BinaryIO, Dict, Literal

from PIL import Image

LOGGER = logging.getLogger(__name__)

# Supported raster image extensions (case-insensitive)
_RASTER_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"})

# Extensions that should never be touched (vector / unsupported)
_SKIP_EXTENSIONS = frozenset({".emf", ".wmf", ".svg", ".gif"})

# Quality presets: (jpeg_quality, max_dimension_px)
CompressionPreset = Literal["high", "medium", "low"]

PRESETS: Dict[CompressionPreset, tuple[int, int]] = {
    "high": (85, 2560),
    "medium": (70, 1920),
    "low": (50, 1280),
}


def _compress_image(
    image_data: bytes,
    extension: str,
    quality: int,
    max_dimension: int,
) -> bytes | None:
    """Compress a single image, returning new bytes or None to keep original.

    Args:
        image_data: Raw image bytes.
        extension: Lowercase file extension (e.g. ".png").
        quality: JPEG quality (1-100).
        max_dimension: Maximum width/height in pixels.

    Returns:
        Compressed image bytes, or None if the image should be kept as-is.
    """
    try:
        img = Image.open(io.BytesIO(image_data))
    except Exception:
        LOGGER.debug("Failed to open image, keeping original.")
        return None

    original_size = len(image_data)
    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )

    # Resize if exceeds max dimension (preserves aspect ratio)
    if max(img.size) > max_dimension:
        img.thumbnail((max_dimension, max_dimension), Image.LANCZOS)

    buf = io.BytesIO()

    if has_alpha:
        # Preserve transparency: save as optimized PNG (no JPEG conversion)
        if img.mode == "P" and "transparency" in img.info:
            img = img.convert("RGBA")
        img.save(buf, format="PNG", optimize=True)
    else:
        # Convert to RGB and save as JPEG
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)

    compressed = buf.getvalue()

    # Only use compressed version if it's actually smaller
    if len(compressed) >= original_size:
        LOGGER.debug(
            "Compressed size (%d) >= original (%d), keeping original.",
            len(compressed),
            original_size,
        )
        return None

    return compressed


def compress_pptx_images(
    input_buffer: BinaryIO,
    preset: CompressionPreset = "medium",
) -> io.BytesIO:
    """Backward-compatible in-memory wrapper for PPTX image compression."""
    result = compress_pptx_images_to_file(input_buffer, preset=preset)
    if isinstance(result, Path):
        buffer = io.BytesIO(result.read_bytes())
        result.unlink(missing_ok=True)
        buffer.seek(0)
        return buffer
    return result


def compress_pptx_images_to_file(
    input_buffer: BinaryIO,
    preset: CompressionPreset = "medium",
    output_path: str | Path | None = None,
) -> io.BytesIO | Path:
    """Compress images inside a PPTX file to reduce memory usage.

    Operates at the ZIP level: reads the PPTX as a ZIP archive, compresses
    raster images in ``ppt/media/``, and writes a new ZIP. File names and
    extensions are preserved so no XML modifications are needed.

    Args:
        input_buffer: File-like object containing the original PPTX file.
        preset: Compression preset (``"high"``, ``"medium"``, or ``"low"``).
        output_path: Optional path to write the compressed PPTX directly.

    Returns:
        A new BytesIO or the output path. If compression fails entirely,
        returns the original buffer with seek position reset.
    """
    import zipfile

    quality, max_dimension = PRESETS.get(preset, PRESETS["medium"])

    input_buffer.seek(0)

    # Verify it's a valid ZIP before attempting compression
    if not zipfile.is_zipfile(input_buffer):
        LOGGER.warning("Input is not a valid ZIP file, returning original.")
        input_buffer.seek(0)
        return input_buffer

    input_buffer.seek(0)
    output_buffer: io.BytesIO | None = None
    output_target = Path(output_path) if output_path is not None else None
    compressed_count = 0
    saved_bytes = 0

    try:
        with zipfile.ZipFile(input_buffer, "r") as zin:
            if output_target is not None:
                with output_target.open("wb") as raw_output:
                    with zipfile.ZipFile(raw_output, "w", zipfile.ZIP_DEFLATED) as zout:
                        compressed_count, saved_bytes = _copy_zip_with_compressed_images(
                            zin, zout, quality, max_dimension
                        )
            else:
                output_buffer = io.BytesIO()
                with zipfile.ZipFile(output_buffer, "w", zipfile.ZIP_DEFLATED) as zout:
                    compressed_count, saved_bytes = _copy_zip_with_compressed_images(
                        zin, zout, quality, max_dimension
                    )

    except Exception:
        LOGGER.exception("Failed to compress PPTX images, returning original.")
        if output_target is not None:
            output_target.unlink(missing_ok=True)
        input_buffer.seek(0)
        return input_buffer

    original_size = input_buffer.seek(0, 2)
    if output_target is not None:
        output_size = output_target.stat().st_size
    else:
        assert output_buffer is not None
        output_size = output_buffer.seek(0, 2)
        output_buffer.seek(0)

    LOGGER.info(
        "Image compression complete: %d images compressed, "
        "saved %.1f MB (%.0f%% reduction, %s preset). "
        "Original: %.1f MB → Compressed: %.1f MB",
        compressed_count,
        saved_bytes / (1024 * 1024),
        (1 - output_size / original_size) * 100 if original_size > 0 else 0,
        preset,
        original_size / (1024 * 1024),
        output_size / (1024 * 1024),
    )

    if output_target is not None:
        return output_target

    assert output_buffer is not None
    return output_buffer


def _copy_zip_with_compressed_images(
    zin,
    zout,
    quality: int,
    max_dimension: int,
) -> tuple[int, int]:
    """Copy ZIP entries, compressing raster PPT media entries opportunistically."""
    compressed_count = 0
    saved_bytes = 0

    for item in zin.infolist():
        data = zin.read(item.filename)

        if item.filename.startswith("ppt/media/"):
            _, ext = os.path.splitext(item.filename.lower())

            if ext in _RASTER_EXTENSIONS:
                compressed = _compress_image(
                    data, ext, quality, max_dimension
                )
                if compressed is not None:
                    saved_bytes += len(data) - len(compressed)
                    data = compressed
                    compressed_count += 1

        zout.writestr(item, data)

    return compressed_count, saved_bytes

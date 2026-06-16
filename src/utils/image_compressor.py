"""PPTX image compression utility.

Compresses images inside PPTX files at the ZIP level (before python-pptx parsing)
to reduce memory usage on resource-constrained servers.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Dict, Literal

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
    input_buffer: io.BytesIO,
    preset: CompressionPreset = "medium",
) -> io.BytesIO:
    """Compress images inside a PPTX file to reduce memory usage.

    Operates at the ZIP level: reads the PPTX as a ZIP archive, compresses
    raster images in ``ppt/media/``, and writes a new ZIP. File names and
    extensions are preserved so no XML modifications are needed.

    Args:
        input_buffer: BytesIO containing the original PPTX file.
        preset: Compression preset (``"high"``, ``"medium"``, or ``"low"``).

    Returns:
        A new BytesIO containing the compressed PPTX. If compression fails
        entirely, returns the original buffer with seek position reset.
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
    output_buffer = io.BytesIO()
    compressed_count = 0
    saved_bytes = 0

    try:
        with zipfile.ZipFile(input_buffer, "r") as zin:
            with zipfile.ZipFile(output_buffer, "w", zipfile.ZIP_DEFLATED) as zout:
                for item in zin.infolist():
                    data = zin.read(item.filename)

                    # Only process images in ppt/media/
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

                    # Write entry (compressed or original)
                    zout.writestr(item, data)

    except Exception:
        LOGGER.exception("Failed to compress PPTX images, returning original.")
        input_buffer.seek(0)
        return input_buffer

    output_buffer.seek(0)
    original_size = input_buffer.seek(0, 2)
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

    return output_buffer

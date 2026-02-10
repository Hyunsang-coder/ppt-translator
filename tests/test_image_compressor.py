"""Tests for PPTX image compression utility."""

from __future__ import annotations

import io
import zipfile

import pytest
from PIL import Image

from src.utils.image_compressor import (
    PRESETS,
    _compress_image,
    compress_pptx_images,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png(width: int = 800, height: int = 600, *, alpha: bool = False) -> bytes:
    """Create a PNG image in memory with noisy pixel data (realistic size)."""
    import random
    random.seed(42)  # Deterministic
    mode = "RGBA" if alpha else "RGB"
    channels = 4 if alpha else 3
    # Generate noisy pixel data so PNG doesn't compress to near-zero
    pixels = bytes(random.randint(0, 255) for _ in range(width * height * channels))
    img = Image.frombytes(mode, (width, height), pixels)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg(width: int = 800, height: int = 600, quality: int = 95) -> bytes:
    """Create a JPEG image in memory."""
    img = Image.new("RGB", (width, height), color=(200, 100, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _make_pptx_zip(media_files: dict[str, bytes]) -> io.BytesIO:
    """Create a minimal PPTX-like ZIP with given media files.

    ``media_files`` maps ``ppt/media/...`` paths to raw bytes.
    Also writes a minimal ``[Content_Types].xml`` so the ZIP looks like a PPTX.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "[Content_Types].xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="xml" ContentType="application/xml"/>'
            "</Types>",
        )
        zf.writestr("ppt/presentation.xml", "<p:presentation/>")
        for path, data in media_files.items():
            zf.writestr(path, data)
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# _compress_image unit tests
# ---------------------------------------------------------------------------


class TestCompressImage:
    """Unit tests for the single-image compression function."""

    def test_compresses_large_png(self):
        """A large opaque PNG should be compressed (converted to JPEG)."""
        png_data = _make_png(2000, 1500)
        result = _compress_image(png_data, ".png", quality=70, max_dimension=1920)
        assert result is not None
        assert len(result) < len(png_data)

    def test_preserves_alpha_png_as_png(self):
        """A PNG with alpha channel should stay PNG (not converted to JPEG)."""
        png_data = _make_png(800, 600, alpha=True)
        result = _compress_image(png_data, ".png", quality=70, max_dimension=1920)
        # Whether compressed or kept original, if result is produced it must be a valid PNG
        if result is not None:
            img = Image.open(io.BytesIO(result))
            assert img.format == "PNG"

    def test_resizes_oversized_image(self):
        """An image exceeding max_dimension should be resized."""
        png_data = _make_png(4000, 3000)
        result = _compress_image(png_data, ".png", quality=85, max_dimension=1920)
        assert result is not None
        img = Image.open(io.BytesIO(result))
        assert max(img.size) <= 1920

    def test_returns_none_if_not_smaller(self):
        """If compression doesn't save space, return None (keep original)."""
        # Create a well-optimized JPEG — recompression at same quality won't help
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70, optimize=True)
        optimized_jpeg = buf.getvalue()
        # Re-compressing at same quality with same dimensions should not be smaller
        result = _compress_image(optimized_jpeg, ".jpg", quality=70, max_dimension=2560)
        assert result is None

    def test_handles_corrupt_data(self):
        """Corrupt image data should return None (fallback)."""
        result = _compress_image(b"not-an-image", ".png", quality=70, max_dimension=1920)
        assert result is None


# ---------------------------------------------------------------------------
# compress_pptx_images integration tests
# ---------------------------------------------------------------------------


class TestCompressPptxImages:
    """Integration tests for the PPTX-level compression function."""

    def test_compresses_png_in_media(self):
        """A large PNG in ppt/media/ should be compressed."""
        large_png = _make_png(3000, 2000)
        pptx = _make_pptx_zip({"ppt/media/image1.png": large_png})

        original_size = pptx.seek(0, 2)
        pptx.seek(0)

        result = compress_pptx_images(pptx, preset="medium")
        result_size = result.seek(0, 2)
        result.seek(0)

        assert result_size < original_size

        # Verify it's still a valid ZIP
        assert zipfile.is_zipfile(result)

    def test_preserves_xml_files(self):
        """Non-media files (XML, rels) should be preserved unchanged."""
        pptx = _make_pptx_zip({"ppt/media/image1.png": _make_png(2000, 1500)})

        result = compress_pptx_images(pptx, preset="medium")
        result.seek(0)

        with zipfile.ZipFile(result, "r") as zf:
            assert "[Content_Types].xml" in zf.namelist()
            assert "ppt/presentation.xml" in zf.namelist()

    def test_skips_emf_files(self):
        """EMF vector files should be kept as-is."""
        emf_data = b"\x01\x00\x00\x00" + b"\x00" * 100  # Fake EMF
        pptx = _make_pptx_zip({"ppt/media/image1.emf": emf_data})

        result = compress_pptx_images(pptx, preset="low")
        result.seek(0)

        with zipfile.ZipFile(result, "r") as zf:
            assert zf.read("ppt/media/image1.emf") == emf_data

    def test_skips_wmf_files(self):
        """WMF vector files should be kept as-is."""
        wmf_data = b"\xd7\xcd\xc6\x9a" + b"\x00" * 100  # Fake WMF
        pptx = _make_pptx_zip({"ppt/media/image1.wmf": wmf_data})

        result = compress_pptx_images(pptx, preset="low")
        result.seek(0)

        with zipfile.ZipFile(result, "r") as zf:
            assert zf.read("ppt/media/image1.wmf") == wmf_data

    def test_skips_svg_files(self):
        """SVG files should be kept as-is."""
        svg_data = b'<svg xmlns="http://www.w3.org/2000/svg"><circle r="10"/></svg>'
        pptx = _make_pptx_zip({"ppt/media/image1.svg": svg_data})

        result = compress_pptx_images(pptx, preset="low")
        result.seek(0)

        with zipfile.ZipFile(result, "r") as zf:
            assert zf.read("ppt/media/image1.svg") == svg_data

    def test_fallback_on_corrupt_image(self):
        """Corrupt image data should be kept as original (not crash)."""
        corrupt = b"this-is-not-an-image"
        pptx = _make_pptx_zip({"ppt/media/image1.png": corrupt})

        result = compress_pptx_images(pptx, preset="medium")
        result.seek(0)

        with zipfile.ZipFile(result, "r") as zf:
            assert zf.read("ppt/media/image1.png") == corrupt

    def test_returns_original_on_non_zip(self):
        """Non-ZIP input should return the original buffer."""
        non_zip = io.BytesIO(b"this is not a zip file")
        result = compress_pptx_images(non_zip, preset="medium")
        result.seek(0)
        assert result.read() == b"this is not a zip file"

    def test_presets_differ(self):
        """Different presets should produce different results for same input."""
        large_png = _make_png(3000, 2000)
        pptx_high = _make_pptx_zip({"ppt/media/image1.png": large_png})
        pptx_low = _make_pptx_zip({"ppt/media/image1.png": large_png})

        result_high = compress_pptx_images(pptx_high, preset="high")
        result_low = compress_pptx_images(pptx_low, preset="low")

        size_high = result_high.seek(0, 2)
        size_low = result_low.seek(0, 2)

        # Low quality should produce smaller (or equal) file
        assert size_low <= size_high

    def test_file_names_preserved(self):
        """All file entry names should be preserved after compression."""
        media = {
            "ppt/media/image1.png": _make_png(2000, 1500),
            "ppt/media/photo.jpg": _make_jpeg(2000, 1500),
            "ppt/media/diagram.emf": b"\x01" * 50,
        }
        pptx = _make_pptx_zip(media)

        result = compress_pptx_images(pptx, preset="medium")
        result.seek(0)

        with zipfile.ZipFile(result, "r") as zf:
            names = zf.namelist()
            for path in media:
                assert path in names

    def test_files_outside_media_untouched(self):
        """Files not under ppt/media/ should never be modified."""
        xml_content = b"<some-xml>content</some-xml>"
        pptx = _make_pptx_zip({})
        # Add a non-media file manually
        pptx.seek(0)
        buf = io.BytesIO()
        with zipfile.ZipFile(pptx, "r") as zin:
            with zipfile.ZipFile(buf, "w") as zout:
                for item in zin.infolist():
                    zout.writestr(item, zin.read(item.filename))
                zout.writestr("ppt/slides/slide1.xml", xml_content)
        buf.seek(0)

        result = compress_pptx_images(buf, preset="low")
        result.seek(0)

        with zipfile.ZipFile(result, "r") as zf:
            assert zf.read("ppt/slides/slide1.xml") == xml_content

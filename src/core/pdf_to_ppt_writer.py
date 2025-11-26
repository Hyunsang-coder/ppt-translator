"""Convert Vision-based semantic OCR results to PowerPoint."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Pt

from src.core.pdf_processor import PageOCRResult, TextBlock

LOGGER = logging.getLogger(__name__)

SLIDE_WIDTH_16_9 = Emu(12192000)   # 13.333 inches
SLIDE_HEIGHT_16_9 = Emu(6858000)  # 7.5 inches


@dataclass
class TextBoxStyle:
    """Base style configuration (fallback)."""
    font_name: str = "맑은 고딕"
    background_color: Tuple[int, int, int] = (255, 255, 255)
    text_color: Tuple[int, int, int] = (0, 0, 0)
    padding_percent: float = 0.02


class PDFToPPTWriter:
    """Convert Semantic TextBlocks to PowerPoint."""

    def __init__(
        self,
        slide_width: Optional[int] = None,
        slide_height: Optional[int] = None,
        text_style: Optional[TextBoxStyle] = None,
    ) -> None:
        self._slide_width = slide_width or SLIDE_WIDTH_16_9
        self._slide_height = slide_height or SLIDE_HEIGHT_16_9
        self._text_style = text_style or TextBoxStyle()

    def _calculate_scale(
        self,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, int, int]:
        image_aspect = image_width / image_height
        slide_aspect = self._slide_width / self._slide_height

        if image_aspect > slide_aspect:
            scale = self._slide_width / image_width
            scaled_height = int(image_height * scale)
            offset_x = 0
            offset_y = (self._slide_height - scaled_height) // 2
        else:
            scale = self._slide_height / image_height
            scaled_width = int(image_width * scale)
            offset_x = (self._slide_width - scaled_width) // 2
            offset_y = 0

        return scale, scale, offset_x, offset_y

    def _pixel_to_emu(
        self,
        block: TextBlock,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, int, int, int]:
        scale_x, scale_y, offset_x, offset_y = self._calculate_scale(
            image_width, image_height
        )

        # Apply minimal padding
        pad = self._text_style.padding_percent
        pad_x = int(block.width * pad)
        pad_y = int(block.height * pad)

        left = int((block.left - pad_x) * scale_x) + offset_x
        top = int((block.top - pad_y) * scale_y) + offset_y
        width = int((block.width + 2 * pad_x) * scale_x)
        height = int((block.height + 2 * pad_y) * scale_y)

        # Ensure minimum dimensions
        width = max(width, int(Pt(30)))
        height = max(height, int(Pt(15)))

        # Clamp
        left = max(0, min(left, self._slide_width - width))
        top = max(0, min(top, self._slide_height - height))

        return left, top, width, height

    def _get_style_for_type(self, block_type: str) -> dict:
        """Get formatting rules based on semantic type."""
        # Default style
        style = {
            "font_size": 14,
            "bold": False,
            "align": PP_ALIGN.LEFT
        }
        
        if block_type == "title":
            style.update({"font_size": 32, "bold": True, "align": PP_ALIGN.CENTER})
        elif block_type == "subtitle":
            style.update({"font_size": 24, "bold": True, "align": PP_ALIGN.CENTER})
        elif block_type == "body":
            style.update({"font_size": 14, "bold": False, "align": PP_ALIGN.LEFT})
        elif block_type == "list":
            style.update({"font_size": 14, "bold": False, "align": PP_ALIGN.LEFT})
        elif block_type == "caption":
            style.update({"font_size": 11, "bold": False, "align": PP_ALIGN.LEFT})
            
        return style

    def _add_background_image(self, slide, image: Image.Image) -> None:
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        scale_x, scale_y, offset_x, offset_y = self._calculate_scale(
            image.width, image.height
        )

        slide.shapes.add_picture(
            img_buffer,
            left=offset_x,
            top=offset_y,
            width=int(image.width * scale_x),
            height=int(image.height * scale_y),
        )

    def _add_text_box(
        self,
        slide,
        block: TextBlock,
        image_width: int,
        image_height: int,
    ) -> None:
        left, top, width, height = self._pixel_to_emu(
            block, image_width, image_height
        )

        textbox = slide.shapes.add_textbox(left, top, width, height)
        tf = textbox.text_frame
        tf.word_wrap = True
        tf.auto_size = None
        
        # Get semantic style
        style_props = self._get_style_for_type(block.block_type)

        p = tf.paragraphs[0]
        p.text = block.text
        p.alignment = style_props["align"]

        if p.runs:
            run = p.runs[0]
        else:
            run = p.add_run()
            run.text = block.text

        run.font.name = self._text_style.font_name
        run.font.size = Pt(style_props["font_size"])
        run.font.bold = style_props["bold"]
        
        # Use Vision-detected color if valid (not black default), else user pref
        # Actually, user wants consistent readable text, so let's stick to user pref or black
        # But Vision detected color might be useful for headers (e.g. orange).
        # Let's prioritize user pref if set to non-black, otherwise try Vision color
        # For now, stick to reliable black/white contrast as requested "opaque box"
        run.font.color.rgb = RGBColor(*self._text_style.text_color)

        # Opaque background
        fill = textbox.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(*self._text_style.background_color)

    def create_presentation(
        self,
        ocr_results: List[PageOCRResult],
    ) -> io.BytesIO:
        prs = Presentation()
        prs.slide_width = self._slide_width
        prs.slide_height = self._slide_height
        blank_layout = prs.slide_layouts[6]

        for result in ocr_results:
            slide = prs.slides.add_slide(blank_layout)
            self._add_background_image(slide, result.image)
            
            for block in result.text_blocks:
                self._add_text_box(
                    slide,
                    block,
                    result.image_width,
                    result.image_height
                )

        buffer = io.BytesIO()
        prs.save(buffer)
        buffer.seek(0)
        return buffer

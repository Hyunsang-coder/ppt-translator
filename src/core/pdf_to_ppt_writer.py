"""Convert hybrid OCR results to PowerPoint with precise positioning."""

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

# Standard slide dimensions in EMU (914400 EMU = 1 inch)
SLIDE_WIDTH_16_9 = Emu(12192000)   # 13.333 inches
SLIDE_HEIGHT_16_9 = Emu(6858000)  # 7.5 inches


@dataclass
class TextBoxStyle:
    """Style configuration for text boxes."""
    
    font_name: str = "맑은 고딕"
    background_color: Tuple[int, int, int] = (255, 255, 255)  # RGB white
    text_color: Tuple[int, int, int] = (0, 0, 0)  # RGB black
    padding_percent: float = 0.05  # 5% padding around text


class PDFToPPTWriter:
    """Convert PDF pages with OCR results to PowerPoint using precise coordinates."""

    def __init__(
        self,
        slide_width: Optional[int] = None,
        slide_height: Optional[int] = None,
        text_style: Optional[TextBoxStyle] = None,
    ) -> None:
        """Initialize the writer."""
        self._slide_width = slide_width or SLIDE_WIDTH_16_9
        self._slide_height = slide_height or SLIDE_HEIGHT_16_9
        self._text_style = text_style or TextBoxStyle()

    def _calculate_scale(
        self,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, int, int]:
        """Calculate scale and offset to fit image to slide.
        
        Returns:
            Tuple of (scale_x, scale_y, offset_x, offset_y)
        """
        image_aspect = image_width / image_height
        slide_aspect = self._slide_width / self._slide_height

        if image_aspect > slide_aspect:
            # Image is wider - fit to width
            scale = self._slide_width / image_width
            scaled_height = int(image_height * scale)
            offset_x = 0
            offset_y = (self._slide_height - scaled_height) // 2
        else:
            # Image is taller - fit to height
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
        """Convert pixel coordinates to EMU coordinates on slide.
        
        Returns:
            Tuple of (left, top, width, height) in EMU.
        """
        scale_x, scale_y, offset_x, offset_y = self._calculate_scale(
            image_width, image_height
        )

        # Apply padding
        pad = self._text_style.padding_percent
        pad_x = int(block.width * pad)
        pad_y = int(block.height * pad)

        left = int((block.left - pad_x) * scale_x) + offset_x
        top = int((block.top - pad_y) * scale_y) + offset_y
        width = int((block.width + 2 * pad_x) * scale_x)
        height = int((block.height + 2 * pad_y) * scale_y)

        # Ensure minimum dimensions
        min_width = Pt(50)  # Minimum 50pt width
        min_height = Pt(20)  # Minimum 20pt height
        width = max(width, min_width)
        height = max(height, min_height)

        # Clamp to slide bounds
        left = max(0, min(left, self._slide_width - width))
        top = max(0, min(top, self._slide_height - height))

        return left, top, width, height

    def _add_background_image(
        self,
        slide,
        image: Image.Image,
    ) -> None:
        """Add image as slide background, covering entire slide."""
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)

        scale_x, scale_y, offset_x, offset_y = self._calculate_scale(
            image.width, image.height
        )

        img_width = int(image.width * scale_x)
        img_height = int(image.height * scale_y)

        slide.shapes.add_picture(
            img_buffer,
            left=offset_x,
            top=offset_y,
            width=img_width,
            height=img_height,
        )

    def _add_text_box(
        self,
        slide,
        block: TextBlock,
        image_width: int,
        image_height: int,
    ) -> None:
        """Add a text box at precise coordinates."""
        left, top, width, height = self._pixel_to_emu(
            block, image_width, image_height
        )

        textbox = slide.shapes.add_textbox(
            left=left,
            top=top,
            width=width,
            height=height,
        )

        # Configure text frame
        tf = textbox.text_frame
        tf.word_wrap = True
        tf.auto_size = None
        tf.anchor = MSO_ANCHOR.MIDDLE

        # Add text
        p = tf.paragraphs[0]
        p.text = block.text
        p.alignment = PP_ALIGN.LEFT

        # Style the run
        if p.runs:
            run = p.runs[0]
        else:
            run = p.add_run()
            run.text = block.text

        run.font.name = self._text_style.font_name
        run.font.size = Pt(block.font_size)
        run.font.color.rgb = RGBColor(*self._text_style.text_color)

        # Set opaque background
        fill = textbox.fill
        fill.solid()
        fill.fore_color.rgb = RGBColor(*self._text_style.background_color)

    def create_presentation(
        self,
        ocr_results: List[PageOCRResult],
    ) -> io.BytesIO:
        """Create PowerPoint presentation from OCR results."""
        prs = Presentation()
        prs.slide_width = self._slide_width
        prs.slide_height = self._slide_height

        blank_layout = prs.slide_layouts[6]  # Blank layout

        LOGGER.info("Creating presentation with %d slides...", len(ocr_results))

        for page_result in ocr_results:
            slide = prs.slides.add_slide(blank_layout)

            # Add background image first
            self._add_background_image(slide, page_result.image)

            # Add text boxes at precise positions
            for block in page_result.text_blocks:
                self._add_text_box(
                    slide,
                    block,
                    page_result.image_width,
                    page_result.image_height,
                )

            LOGGER.info(
                "Slide %d: %d text boxes",
                page_result.page_number,
                len(page_result.text_blocks),
            )

        buffer = io.BytesIO()
        prs.save(buffer)
        buffer.seek(0)

        LOGGER.info("Presentation created: %d slides", len(ocr_results))
        return buffer

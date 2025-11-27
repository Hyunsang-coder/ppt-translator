"""Convert Vision-based semantic OCR results to PowerPoint."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.util import Emu, Pt

from src.core.pdf_processor import PageOCRResult, TextBlock

LOGGER = logging.getLogger(__name__)


@dataclass
class TextBoxStyle:
    """Style configuration."""
    font_name: str = "맑은 고딕"
    background_color: Optional[Tuple[int, int, int]] = (255, 255, 255) # None means adaptive
    text_color: Optional[Tuple[int, int, int]] = (0, 0, 0)             # None means adaptive
    # Reduced padding for tighter fit to original text
    padding_percent: float = 0.01


@dataclass
class LayoutRect:
    """Rectangle in EMU coordinates for layout calculations."""
    left: int
    top: int
    width: int
    height: int
    block_index: int  # Reference to original TextBlock index
    block_type: str = "body"
    text: str = ""
    
    @property
    def right(self) -> int:
        return self.left + self.width
    
    @property
    def bottom(self) -> int:
        return self.top + self.height
    
    def overlaps_with(self, other: "LayoutRect") -> bool:
        """Check if this rect overlaps with another."""
        # No overlap if one is completely to the left, right, above, or below
        if self.right <= other.left or other.right <= self.left:
            return False
        if self.bottom <= other.top or other.bottom <= self.top:
            return False
        return True
    
    def overlap_area(self, other: "LayoutRect") -> int:
        """Calculate overlap area with another rect."""
        x_overlap = max(0, min(self.right, other.right) - max(self.left, other.left))
        y_overlap = max(0, min(self.bottom, other.bottom) - max(self.top, other.top))
        return x_overlap * y_overlap

    def expand(self, x_ratio: float = 0.05, y_ratio: float = 0.01) -> None:
        """
        [안전장치 1] 비대칭 팽창
        좌우(x)는 넉넉히(5%), 상하(y)는 최소한(1%)으로 늘려서
        윗줄/아랫줄 침범 및 이미지 가림 현상을 방지함
        """
        w_expand = int(self.width * x_ratio)
        h_expand = int(self.height * y_ratio)
        
        self.left -= w_expand
        self.top -= h_expand
        self.width += (w_expand * 2)
        self.height += (h_expand * 2)

    def should_merge_with(self, other: "LayoutRect") -> bool:
        """
        [안전장치 2] 조건부 병합 판단 로직
        """
        # 1. 의미론적 타입이 다르면 병합 금지 (예: 제목과 본문이 합쳐지는 참사 방지)
        if self.block_type != other.block_type:
            return False
            
        # 2. 거리가 너무 멀면 병합 금지 (예: 20pt 이상 떨어져 있으면 별개로 취급)
        # 세로 거리 계산
        y_gap = max(0, other.top - self.bottom) if self.top < other.top else max(0, self.top - other.bottom)
        if y_gap > Pt(20): 
            return False
            
        return True

    def union(self, other: "LayoutRect") -> "LayoutRect":
        """Merge with another rect."""
        new_left = min(self.left, other.left)
        new_top = min(self.top, other.top)
        new_right = max(self.right, other.right)
        new_bottom = max(self.bottom, other.bottom)
        
        # Merge text based on top position
        if self.top <= other.top:
            new_text = f"{self.text}\n{other.text}"
        else:
            new_text = f"{other.text}\n{self.text}"
            
        return LayoutRect(
            left=new_left,
            top=new_top,
            width=new_right - new_left,
            height=new_bottom - new_top,
            block_index=self.block_index, # Keep reference to one
            block_type=self.block_type,
            text=new_text
        )


@dataclass
class DetectedStyle:
    """Style metadata returned from Vision analysis."""
    color_hex: str = "#000000"
    is_bold: bool = False
    font_category: str = "sans-serif"
    alignment: str = "left"


class PDFToPPTWriter:
    """Convert Semantic TextBlocks to PowerPoint with robust layout handling."""

    # Minimum gap between text boxes (in EMU) to prevent visual collision
    MIN_GAP_EMU = int(Pt(3))  # 3pt gap

    def __init__(
        self,
        text_style: Optional[TextBoxStyle] = None,
        openai_api_key: Optional[str] = None,
        vision_model: str = "gpt-5.1",
        enable_inpainting: bool = True,
        enable_style_inference: bool = False,
    ) -> None:
        self._text_style = text_style or TextBoxStyle()
        # Slide dimensions will be set dynamically based on the first page
        self._slide_width = Emu(0)
        self._slide_height = Emu(0)
        self._openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self._vision_model = vision_model
        self._enable_inpainting = enable_inpainting
        self._enable_style_inference = enable_style_inference

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Encode a PIL image to base64 JPEG for Vision calls."""
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _crop_image_for_rect(
        self,
        image: Image.Image,
        rect: LayoutRect,
        scale: float,
        offset_x: int,
        offset_y: int,
    ) -> Optional[Image.Image]:
        """Crop a PIL image using LayoutRect (EMU → pixel)."""
        try:
            left = int((rect.left - offset_x) / scale)
            top = int((rect.top - offset_y) / scale)
            width = int(rect.width / scale)
            height = int(rect.height / scale)

            left = max(0, left)
            top = max(0, top)
            right = min(image.width, left + width)
            bottom = min(image.height, top + height)

            if right <= left or bottom <= top:
                return None
            return image.crop((left, top, right, bottom))
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to crop image for style inference: %s", exc)
            return None

    def _parse_alignment(self, alignment: str) -> PP_ALIGN:
        """Map string alignment to PPT enum."""
        align_lower = (alignment or "").lower()
        if align_lower == "center":
            return PP_ALIGN.CENTER
        if align_lower == "right":
            return PP_ALIGN.RIGHT
        return PP_ALIGN.LEFT

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex string to RGB tuple."""
        hex_color = hex_color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(c * 2 for c in hex_color)
        try:
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return (0, 0, 0)

    def _analyze_style_with_gpt(
        self,
        image_crop: Image.Image,
        ocr_text: str,
    ) -> DetectedStyle:
        """
        Use OpenAI Vision to infer color/bold/font category/alignment.
        Fallbacks are not provided; caller is expected to supply API key.
        """
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency not installed
            raise ImportError("openai package is required for style inference") from exc

        if not self._openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for style inference.")

        base64_img = self._encode_image_to_base64(image_crop)
        prompt = (
            'Analyze the text style in this image crop. The text content is: '
            f'"{ocr_text}". Return a JSON object ONLY with the following keys: '
            '"color_hex", "is_bold", "font_category" ("serif" or "sans-serif"), '
            'and "alignment" ("left", "center", or "right").'
        )

        client = OpenAI(api_key=self._openai_api_key)
        response = client.chat.completions.create(
            model=self._vision_model,
            messages=[
                {"role": "system", "content": "You are a graphic design expert. Output JSON only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"},
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=200,
        )
        content = response.choices[0].message.content
        parsed = json.loads(content)

        return DetectedStyle(
            color_hex=parsed.get("color_hex", "#000000"),
            is_bold=bool(parsed.get("is_bold", False)),
            font_category=parsed.get("font_category", "sans-serif"),
            alignment=parsed.get("alignment", "left"),
        )

    def _build_inpainted_background(
        self,
        image: Image.Image,
        rects: List[LayoutRect],
        scale: float,
        offset_x: int,
        offset_y: int,
        dilation_kernel: int = 5,
    ) -> Image.Image:
        """Remove detected text by inpainting the regions defined by LayoutRects."""
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover - dependency not installed
            raise ImportError("opencv-python-headless and numpy are required for inpainting") from exc

        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        mask = np.zeros(cv_img.shape[:2], dtype=np.uint8)

        for rect in rects:
            x0 = int((rect.left - offset_x) / scale)
            y0 = int((rect.top - offset_y) / scale)
            w = int(rect.width / scale)
            h = int(rect.height / scale)

            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(cv_img.shape[1], x0 + w)
            y1 = min(cv_img.shape[0], y0 + h)
            if x1 <= x0 or y1 <= y0:
                continue
            cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)

        kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        clean = cv2.inpaint(cv_img, dilated, 3, cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(clean, cv2.COLOR_BGR2RGB))

    def _blocks_to_layout_rects(
        self,
        blocks: List[TextBlock],
        scale: float,
        offset_x: int,
        offset_y: int,
        cover_original: bool = True,
    ) -> List[LayoutRect]:
        """
        Convert TextBlocks to LayoutRects in EMU coordinates.
        
        Applies dynamic size calculations based on text content and font size.
        """
        rects = []
        for idx, block in enumerate(blocks):
            left = int(block.left * scale) + offset_x
            top = int(block.top * scale) + offset_y
            width = int(block.width * scale)
            height = int(block.height * scale)
            
            # Get font size for this block type
            style_props = self._get_style_for_type(block.block_type)
            font_size_pt = style_props["font_size"]
            font_size_emu = int(Pt(font_size_pt))
            
            # Calculate dynamic height based on text content
            dynamic_height = self._calculate_dynamic_height(
                block.text, width, font_size_emu
            )
            dynamic_width = max(int(self._estimate_text_width(block.text, font_size_emu) * 1.1), int(Pt(30)))
            
            # Use the larger of Vision's estimate or our dynamic calculation
            # This ensures we don't make boxes too small
            if cover_original:
                height = max(height, dynamic_height)
                width = max(width, int(Pt(30)))
            else:
                # When not masking the original, prefer tighter boxes
                height = max(dynamic_height, int(Pt(15)))
                width = max(min(width, dynamic_width), int(Pt(30)))
            
            # Apply minimum dimensions
            width = max(width, int(Pt(30)))
            height = max(height, int(Pt(15)))
            
            # Apply maximum height constraint (80% of slide height)
            max_height = int(self._slide_height * 0.8) if self._slide_height > 0 else height
            height = min(height, max_height)
            
            rect = LayoutRect(
                left=left,
                top=top,
                width=width,
                height=height,
                block_index=idx,
                block_type=block.block_type,
                text=block.text
            )
            
            # [안전장치 1] 비대칭 팽창 적용 (only when masking original text)
            if cover_original:
                rect.expand()
            
            rects.append(rect)
        return rects

    def _calculate_dynamic_height(
        self, 
        text: str, 
        width_emu: int, 
        font_size_emu: int
    ) -> int:
        """
        Calculate dynamic height based on text content and available width.
        
        Estimates number of lines needed and calculates height accordingly.
        """
        if not text or width_emu <= 0:
            return int(Pt(15))  # Minimum height
        
        # Estimate text width
        estimated_text_width = self._estimate_text_width(text, font_size_emu)
        
        # Calculate number of lines (with word wrap)
        # Add some buffer for word wrapping (words might not break perfectly)
        num_lines = max(1, int((estimated_text_width / width_emu) + 0.5))
        
        # Also count explicit line breaks
        explicit_lines = text.count('\n') + 1
        num_lines = max(num_lines, explicit_lines)
        
        # Line height is typically 1.2-1.5x font size
        line_height = int(font_size_emu * 1.3)
        
        # Add padding (top + bottom)
        padding = int(font_size_emu * 0.5)
        
        return num_lines * line_height + padding

    def _merge_overlaps(self, rects: List[LayoutRect]) -> List[LayoutRect]:
        if not rects: return []
        
        # Y축 정렬
        sorted_rects = sorted(rects, key=lambda r: r.top)
        merged = []
        while sorted_rects:
            current = sorted_rects.pop(0)
            
            # 병합 후보 찾기 루프
            has_merged = True
            while has_merged:
                has_merged = False
                non_overlapping = []
                
                for other in sorted_rects:
                    # 겹치거나 매우 가깝고, AND [안전장치] 병합해도 되는 녀석인가?
                    if current.overlaps_with(other) and current.should_merge_with(other):
                        current = current.union(other)
                        has_merged = True
                    else:
                        non_overlapping.append(other)
                
                sorted_rects = non_overlapping
            
            merged.append(current)
            
        return merged

    def _find_overlaps(self, rects: List[LayoutRect]) -> List[Tuple[int, int, int]]:
        """
        Find all overlapping pairs in a list of rects.
        
        Returns: List of (index1, index2, overlap_area) tuples
        """
        overlaps = []
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                area = rects[i].overlap_area(rects[j])
                if area > 0:
                    overlaps.append((i, j, area))
        return overlaps

    def _calculate_layout_metrics(
        self, 
        image_width: int, 
        image_height: int
    ) -> Tuple[float, int, int]:
        """
        Calculate scale and offsets to fit image into slide.
        
        Handles mixed aspect ratios by letterboxing (centering) the content
        while preserving the master slide size determined by the first page.
        """
        if self._slide_width == 0 or self._slide_height == 0:
            return 1.0, 0, 0

        image_aspect = image_width / image_height
        slide_aspect = self._slide_width / self._slide_height

        # If aspect ratios match within tolerance, fit completely
        if abs(image_aspect - slide_aspect) < 0.01:
            scale = self._slide_width / image_width
            return scale, 0, 0

        # Letterboxing logic for mixed aspect ratios
        if image_aspect > slide_aspect:
            # Image is wider (relative to slide) -> fit to width
            scale = self._slide_width / image_width
            scaled_height = int(image_height * scale)
            offset_x = 0
            offset_y = (self._slide_height - scaled_height) // 2
        else:
            # Image is taller (relative to slide) -> fit to height
            scale = self._slide_height / image_height
            scaled_width = int(image_width * scale)
            offset_x = (self._slide_width - scaled_width) // 2
            offset_y = 0
            
        return scale, offset_x, offset_y

    def _get_style_for_type(self, block_type: str) -> dict:
        """Get formatting rules based on semantic type."""
        base = {"bold": False, "align": PP_ALIGN.LEFT}
        
        # Font sizes are presets ensuring readability
        if block_type == "title":
            return {**base, "font_size": 28, "bold": True, "align": PP_ALIGN.CENTER}
        elif block_type == "subtitle":
            return {**base, "font_size": 20, "bold": True, "align": PP_ALIGN.CENTER}
        elif block_type == "caption":
            return {**base, "font_size": 10}
        
        # Body, List, etc.
        return {**base, "font_size": 12}

    def _get_average_color(
        self, 
        image: Image.Image, 
        rect: LayoutRect, 
        scale: float, 
        offset_x: int, 
        offset_y: int
    ) -> Tuple[int, int, int]:
        """
        Calculate the average color of the region covered by the text box.
        
        Args:
            image: Original PIL Image
            rect: LayoutRect (EMU coordinates)
            scale: Scale factor (EMU per Pixel)
            offset_x: X offset in EMU
            offset_y: Y offset in EMU
        """
        try:
            # Convert EMU to Pixels
            # Formula: pixel = (emu - offset) / scale
            left = int((rect.left - offset_x) / scale)
            top = int((rect.top - offset_y) / scale)
            width = int(rect.width / scale)
            height = int(rect.height / scale)
            
            # Clamp to image boundaries
            left = max(0, left)
            top = max(0, top)
            right = min(image.width, left + width)
            bottom = min(image.height, top + height)
            
            if right <= left or bottom <= top:
                return (255, 255, 255) # Fallback to white if invalid area
            
            # Crop ROI
            roi = image.crop((left, top, right, bottom))
            
            # Resize for speed (1x1 pixel averaging)
            # This is a fast way to get average color
            stats = roi.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
            
            # Handle RGBA/RGB
            if isinstance(stats, int): # Grayscale
                return (stats, stats, stats)
            elif len(stats) >= 3:
                return stats[:3]
            
            return (255, 255, 255)
            
        except Exception as e:
            LOGGER.warning("Failed to extract average color: %s", e)
            return (255, 255, 255)

    def _get_luminance(self, color: Tuple[int, int, int]) -> float:
        """Calculate relative luminance (0.0 to 1.0)."""
        r, g, b = [c / 255.0 for c in color]
        return 0.299 * r + 0.587 * g + 0.114 * b

    def _get_optimal_text_color(self, bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Return Black or White text based on background luminance."""
        lum = self._get_luminance(bg_color)
        # Threshold 0.5 is standard; if bright, use black text.
        return (0, 0, 0) if lum > 0.5 else (255, 255, 255)

    def _add_text_box(
        self,
        slide,
        layout_rect: LayoutRect,
        image: Optional[Image.Image] = None,
        scale: float = 1.0,
        offset_x: int = 0,
        offset_y: int = 0,
        detected_style: Optional[DetectedStyle] = None,
        vision_text_color: Optional[Tuple[int, int, int]] = None,
        cover_original: bool = True,
    ) -> None:
        """
        Add a text box to the slide using pre-calculated layout coordinates.
        
        Layout Strategy:
        - Position: Always use Top-Left of original text region
        - Size: Keep original region size to fully cover source text
        - Text alignment is handled within the box, not by moving the box
        """
        # Use pre-calculated coordinates from LayoutRect
        # IMPORTANT: Always use Top-Left positioning to fully cover original text
        left = layout_rect.left
        top = layout_rect.top
        width = layout_rect.width
        height = layout_rect.height

        # Get Style Properties
        style_props = self._get_style_for_type(layout_rect.block_type)
        font_size_pt = style_props["font_size"]

        # Keep original width to ensure we cover the source text completely when masking
        # If not masking, allow tighter boxes (size already handled upstream)
        width = max(width, int(Pt(30)))
        height = max(height, int(Pt(15)))
        
        # Boundary Safety - keep box within slide
        left = max(0, min(left, self._slide_width - width))
        top = max(0, min(top, self._slide_height - height))

        # Create TextBox at exact Top-Left position of original region
        textbox = slide.shapes.add_textbox(left, top, width, height)
        tf = textbox.text_frame
        tf.word_wrap = True
        # Auto-fit to reduce overflow when translations get longer
        tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        
        # [안전장치 3] 내부 여백 제거
        # 박스를 키웠기 때문에 텍스트가 중앙에 붕 떠보이는 것을 방지하기 위해
        # 내부 마진을 줄이고, 정렬로 제어합니다.
        tf.margin_left = 0
        tf.margin_right = 0
        tf.margin_top = 0
        tf.margin_bottom = 0
        
        # 수직 정렬: 박스가 커졌을 때 글자가 위나 아래로 치우치지 않게 '중앙' 권장
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE

        # Determine Colors (Adaptive vs Default)
        bg_color = self._text_style.background_color
        text_color = self._text_style.text_color or vision_text_color
        if detected_style and detected_style.color_hex:
            text_color = self._hex_to_rgb(detected_style.color_hex)
        
        # Logic: Use Adaptive if (image exists AND style says None) OR (image exists AND forced adaptive in logic - NO, we want to respect style)
        # Updated Logic: 
        # If style.background_color is None -> Use Adaptive (requires image)
        # If style.background_color is Set -> Use Set Color
        
        if bg_color is None and image:
            # Adaptive Coloring
            bg_color = self._get_average_color(image, layout_rect, scale, offset_x, offset_y)
            text_color = self._get_optimal_text_color(bg_color)
        elif bg_color is None:
             # Fallback if no image but adaptive requested
             bg_color = (255, 255, 255)
             text_color = (0, 0, 0)
        
        # Ensure text color is always a tuple
        text_color = text_color or (0, 0, 0)
        
        # Apply Content & Style
        p = tf.paragraphs[0]
        p.clear()
        # Text alignment happens INSIDE the box (not by moving the box)
        if detected_style:
            p.alignment = self._parse_alignment(detected_style.alignment)
        else:
            p.alignment = style_props["align"]
        
        run = p.add_run()
        run.text = layout_rect.text
        if detected_style and detected_style.font_category == "serif":
            run.font.name = "Times New Roman"
        else:
            run.font.name = self._text_style.font_name
        run.font.size = Pt(font_size_pt)
        run.font.bold = detected_style.is_bold if detected_style else style_props["bold"]
        run.font.color.rgb = RGBColor(*text_color)

        # Background handling: keep opaque only when masking is desired
        fill = textbox.fill
        if self._enable_inpainting or not cover_original:
            fill.background()
        else:
            fill.solid()
            fill.fore_color.rgb = RGBColor(*bg_color)

    def _estimate_text_width(self, text: str, font_size_emu: int) -> int:
        """
        Estimate text width based on character count and font size.
        
        Uses different width ratios for CJK vs Latin characters:
        - Latin chars: ~0.5 of font height
        - CJK chars: ~0.9 of font height
        """
        cjk_count = sum(1 for c in text if self._is_cjk_char(c))
        latin_count = len(text) - cjk_count
        
        # Calculate width based on character types
        cjk_width = cjk_count * font_size_emu * 0.9
        latin_width = latin_count * font_size_emu * 0.5
        
        # Add padding buffer (20%) for safety margin
        return int((cjk_width + latin_width) * 1.2)

    def _is_cjk_char(self, char: str) -> bool:
        """Check if a character is CJK (Chinese, Japanese, Korean)."""
        code = ord(char)
        # CJK Unified Ideographs and common ranges
        return (
            0x4E00 <= code <= 0x9FFF or   # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or   # CJK Extension A
            0xAC00 <= code <= 0xD7AF or   # Korean Hangul Syllables
            0x3000 <= code <= 0x303F or   # CJK Punctuation
            0xFF00 <= code <= 0xFFEF      # Fullwidth Forms
        )

    def create_presentation(
        self,
        ocr_results: List[PageOCRResult],
        include_background: bool = True,
        resolve_overlaps: bool = True,
    ) -> io.BytesIO:
        """
        Create a PowerPoint presentation from OCR results.
        
        Args:
            ocr_results: List of PageOCRResult objects
            include_background: Whether to include the original PDF page as background
            resolve_overlaps: Whether to resolve overlapping text boxes
        
        Returns:
            BytesIO buffer containing the PPTX file
        """
        prs = Presentation()
        
        if not ocr_results:
            buffer = io.BytesIO()
            prs.save(buffer)
            buffer.seek(0)
            return buffer

        # Determine Master Slide Size from First Page
        first_page = ocr_results[0]
        
        # Convert pixels to EMUs using a standard 96 DPI assumption
        self._slide_width = Emu(first_page.image_width * 9525)
        self._slide_height = Emu(first_page.image_height * 9525)

        prs.slide_width = self._slide_width
        prs.slide_height = self._slide_height
        blank_layout = prs.slide_layouts[6]

        for result in ocr_results:
            slide = prs.slides.add_slide(blank_layout)
            
            # Calculate layout metrics for this specific page
            scale, off_x, off_y = self._calculate_layout_metrics(
                result.image_width, 
                result.image_height
            )

            cover_original = include_background

            # Convert blocks to layout rects
            layout_rects = self._blocks_to_layout_rects(
                result.text_blocks, scale, off_x, off_y, cover_original=cover_original
            )

            # Optional background inpainting to remove original text
            background_image = result.image
            if include_background and self._enable_inpainting and layout_rects:
                try:
                    background_image = self._build_inpainted_background(
                        result.image, layout_rects, scale, off_x, off_y
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Inpainting failed for page %d, falling back to original background: %s",
                        result.page_number,
                        exc,
                    )
                    background_image = result.image

            # Add background image
            if include_background:
                img_buffer = io.BytesIO()
                background_image.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                
                slide.shapes.add_picture(
                    img_buffer,
                    left=off_x,
                    top=off_y,
                    width=int(result.image_width * scale),
                    height=int(result.image_height * scale),
                )
            
            # Resolve overlaps if enabled
            if resolve_overlaps and cover_original and layout_rects:
                pre_overlaps = self._find_overlaps(layout_rects)
                if pre_overlaps:
                    LOGGER.info(
                        "Page %d: Found %d overlapping block pairs, resolving...",
                        result.page_number, len(pre_overlaps)
                    )
                
                # Use merge strategy instead of push
                layout_rects = self._merge_overlaps(layout_rects)
                
                post_overlaps = self._find_overlaps(layout_rects)
                if post_overlaps:
                    LOGGER.warning(
                        "Page %d: %d overlaps remain after resolution",
                        result.page_number, len(post_overlaps)
                    )
            
            # Add text boxes using resolved layout
            for rect in layout_rects:
                detected_style = None
                if self._enable_style_inference:
                    crop = self._crop_image_for_rect(result.image, rect, scale, off_x, off_y)
                    if crop:
                        detected_style = self._analyze_style_with_gpt(crop, rect.text)

                vision_text_color = None
                if 0 <= rect.block_index < len(result.text_blocks):
                    vision_text_color = result.text_blocks[rect.block_index].text_color

                self._add_text_box(
                    slide=slide, 
                    layout_rect=rect,
                    image=background_image,  # Use cleaned background for adaptive color
                    scale=scale,
                    offset_x=off_x,
                    offset_y=off_y,
                    detected_style=detected_style,
                    vision_text_color=vision_text_color,
                    cover_original=cover_original,
                )

        buffer = io.BytesIO()
        prs.save(buffer)
        buffer.seek(0)
        return buffer

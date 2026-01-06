"""Helper methods for PDFToPPTWriter - organized by category."""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from PIL import Image
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.util import Pt

from src.core.pdf_processor import TextBlock

if TYPE_CHECKING:
    from src.core.pdf_to_ppt_writer import LayoutRect, DetectedStyle, TextBoxStyle
else:
    # Runtime type stubs
    LayoutRect = None
    DetectedStyle = None
    TextBoxStyle = None

LOGGER = logging.getLogger(__name__)


# ============================================================================
# 1. 이미지 처리 관련 헬퍼 메서드
# ============================================================================

def encode_image_to_base64(image: Image.Image) -> str:
    """Encode a PIL image to base64 JPEG for Vision calls."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def crop_image_for_rect(
    image: Image.Image,
    rect: "LayoutRect",
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


def build_inpainted_background(
    image: Image.Image,
    rects: List["LayoutRect"],
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


def get_average_color(
    image: Image.Image,
    rect: "LayoutRect",
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
            return (255, 255, 255)  # Fallback to white if invalid area
        
        # Crop ROI
        roi = image.crop((left, top, right, bottom))
        
        # Resize for speed (1x1 pixel averaging)
        # This is a fast way to get average color
        stats = roi.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
        
        # Handle RGBA/RGB
        if isinstance(stats, int):  # Grayscale
            return (stats, stats, stats)
        elif len(stats) >= 3:
            return stats[:3]
        
        return (255, 255, 255)
        
    except Exception as e:
        LOGGER.warning("Failed to extract average color: %s", e)
        return (255, 255, 255)


# ============================================================================
# 2. 스타일 관련 헬퍼 메서드
# ============================================================================

def parse_alignment(alignment: str) -> PP_ALIGN:
    """Map string alignment to PPT enum."""
    align_lower = (alignment or "").lower()
    if align_lower == "center":
        return PP_ALIGN.CENTER
    if align_lower == "right":
        return PP_ALIGN.RIGHT
    return PP_ALIGN.LEFT


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex string to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    try:
        return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    except Exception:
        return (0, 0, 0)


def analyze_style_with_gpt(
    image_crop: Image.Image,
    ocr_text: str,
    openai_api_key: str,
    vision_model: str = "gpt-5.1",
) -> "DetectedStyle":
    """
    Use OpenAI Vision to infer color/bold/font category/alignment.
    Fallbacks are not provided; caller is expected to supply API key.
    """
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - dependency not installed
        raise ImportError("openai package is required for style inference") from exc

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY is required for style inference.")

    base64_img = encode_image_to_base64(image_crop)
    prompt = (
        'Analyze the text style in this image crop. The text content is: '
        f'"{ocr_text}". Return a JSON object ONLY with the following keys: '
        '"color_hex", "is_bold", "font_category" ("serif" or "sans-serif"), '
        'and "alignment" ("left", "center", or "right").'
    )

    client = OpenAI(api_key=openai_api_key)
    response = client.chat.completions.create(
        model=vision_model,
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

    from src.core.pdf_to_ppt_writer import DetectedStyle
    return DetectedStyle(
        color_hex=parsed.get("color_hex", "#000000"),
        is_bold=bool(parsed.get("is_bold", False)),
        font_category=parsed.get("font_category", "sans-serif"),
        alignment=parsed.get("alignment", "left"),
    )


def get_style_for_type(block_type: str) -> dict:
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


def get_luminance(color: Tuple[int, int, int]) -> float:
    """Calculate relative luminance (0.0 to 1.0)."""
    r, g, b = [c / 255.0 for c in color]
    return 0.299 * r + 0.587 * g + 0.114 * b


def get_optimal_text_color(bg_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return Black or White text based on background luminance."""
    lum = get_luminance(bg_color)
    # Threshold 0.5 is standard; if bright, use black text.
    return (0, 0, 0) if lum > 0.5 else (255, 255, 255)


# ============================================================================
# 3. 레이아웃 계산 관련 헬퍼 메서드
# ============================================================================

def is_cjk_char(char: str) -> bool:
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


def estimate_text_width(text: str, font_size_emu: int) -> int:
    """
    Estimate text width based on character count and font size.
    
    Uses different width ratios for CJK vs Latin characters:
    - Latin chars: ~0.5 of font height
    - CJK chars: ~0.9 of font height
    """
    cjk_count = sum(1 for c in text if is_cjk_char(c))
    latin_count = len(text) - cjk_count
    
    # Calculate width based on character types
    cjk_width = cjk_count * font_size_emu * 0.9
    latin_width = latin_count * font_size_emu * 0.5
    
    # Add padding buffer (20%) for safety margin
    return int((cjk_width + latin_width) * 1.2)


def calculate_dynamic_height(
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
    estimated_text_width = estimate_text_width(text, font_size_emu)
    
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


def blocks_to_layout_rects(
    blocks: List[TextBlock],
    scale: float,
    offset_x: int,
    offset_y: int,
    slide_height: int,
    cover_original: bool = True,
) -> List["LayoutRect"]:
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
        style_props = get_style_for_type(block.block_type)
        font_size_pt = style_props["font_size"]
        font_size_emu = int(Pt(font_size_pt))
        
        # Calculate dynamic height based on text content
        dynamic_height = calculate_dynamic_height(
            block.text, width, font_size_emu
        )
        dynamic_width = max(int(estimate_text_width(block.text, font_size_emu) * 1.1), int(Pt(30)))
        
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
        max_height = int(slide_height * 0.8) if slide_height > 0 else height
        height = min(height, max_height)
        
        from src.core.pdf_to_ppt_writer import LayoutRect
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


def calculate_layout_metrics(
    image_width: int,
    image_height: int,
    slide_width: int,
    slide_height: int,
) -> Tuple[float, int, int]:
    """
    Calculate scale and offsets to fit image into slide.
    
    Handles mixed aspect ratios by letterboxing (centering) the content
    while preserving the master slide size determined by the first page.
    """
    if slide_width == 0 or slide_height == 0:
        return 1.0, 0, 0

    image_aspect = image_width / image_height
    slide_aspect = slide_width / slide_height

    # If aspect ratios match within tolerance, fit completely
    if abs(image_aspect - slide_aspect) < 0.01:
        scale = slide_width / image_width
        return scale, 0, 0

    # Letterboxing logic for mixed aspect ratios
    if image_aspect > slide_aspect:
        # Image is wider (relative to slide) -> fit to width
        scale = slide_width / image_width
        scaled_height = int(image_height * scale)
        offset_x = 0
        offset_y = (slide_height - scaled_height) // 2
    else:
        # Image is taller (relative to slide) -> fit to height
        scale = slide_height / image_height
        scaled_width = int(image_width * scale)
        offset_x = (slide_width - scaled_width) // 2
        offset_y = 0
        
    return scale, offset_x, offset_y


# ============================================================================
# 4. 병합 관련 헬퍼 메서드
# ============================================================================

def find_overlaps(rects: List["LayoutRect"]) -> List[Tuple[int, int, int]]:
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


def merge_overlaps(rects: List["LayoutRect"]) -> List["LayoutRect"]:
    """Merge overlapping rects based on should_merge_with logic."""
    if not rects:
        return []
    
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


def merge_nearby_rects(
    rects: List["LayoutRect"],
    line_gap_tolerance: float = 1.2,
    align_tolerance_pt: int = 20,
) -> List["LayoutRect"]:
    """
    Merge line-level rects into paragraph-level rects based on proximity and alignment.
    Priority: keep positions stable while reducing fragmentation.
    """
    if not rects:
        return []

    align_tol = int(Pt(align_tolerance_pt))
    sorted_rects = sorted(rects, key=lambda r: r.top)
    merged: List[LayoutRect] = []
    current = sorted_rects[0]

    for nxt in sorted_rects[1:]:
        vert_gap = nxt.top - (current.top + current.height)
        line_h = current.height
        close_vert = vert_gap < line_h * line_gap_tolerance
        overlap_vert = vert_gap < 0

        close_left = abs(current.left - nxt.left) < align_tol
        cur_center = current.left + current.width / 2
        nxt_center = nxt.left + nxt.width / 2
        close_center = abs(cur_center - nxt_center) < align_tol

        if (close_vert or overlap_vert) and (close_left or close_center):
            new_left = min(current.left, nxt.left)
            new_top = min(current.top, nxt.top)
            new_right = max(current.right, nxt.right)
            new_bottom = max(current.bottom, nxt.bottom)

            from src.core.pdf_to_ppt_writer import LayoutRect
            current = LayoutRect(
                left=new_left,
                top=new_top,
                width=new_right - new_left,
                height=new_bottom - new_top,
                block_index=current.block_index,
                text=f"{current.text}\n{nxt.text}",
                block_type=current.block_type,
            )
        else:
            merged.append(current)
            current = nxt

    merged.append(current)
    return merged


def find_gaps_from_intervals(
    intervals: List[Tuple[int, int]],
    min_gap: int,
    domain_end: int,
) -> List[int]:
    """
    Given a set of covered intervals, return midpoints of gaps larger than min_gap.
    All units in EMU for consistency.
    """
    if not intervals:
        return []

    merged = []
    for start, end in sorted(intervals, key=lambda v: v[0]):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    cuts: List[int] = []
    # gap before first
    if merged[0][0] > min_gap:
        cuts.append(merged[0][0] // 2)

    for i in range(len(merged) - 1):
        gap = merged[i + 1][0] - merged[i][1]
        if gap >= min_gap:
            cuts.append(merged[i][1] + gap // 2)

    # gap after last
    if domain_end - merged[-1][1] >= min_gap:
        cuts.append(merged[-1][1] + (domain_end - merged[-1][1]) // 2)

    return cuts


def merge_rect_group(rects: List["LayoutRect"]) -> "LayoutRect":
    """Merge a list of rects into one bounding rect with concatenated text."""
    rects_sorted = sorted(rects, key=lambda r: (r.top, r.left))
    left = min(r.left for r in rects_sorted)
    top = min(r.top for r in rects_sorted)
    right = max(r.right for r in rects_sorted)
    bottom = max(r.bottom for r in rects_sorted)
    text = "\n".join([r.text for r in rects_sorted])
    base = rects_sorted[0]
    from src.core.pdf_to_ppt_writer import LayoutRect
    return LayoutRect(
        left=left,
        top=top,
        width=right - left,
        height=bottom - top,
        block_index=base.block_index,
        block_type=base.block_type,
        text=text,
    )


def xy_cut_merge(
    rects: List["LayoutRect"],
    page_width: int,
    page_height: int,
    min_x_gap_ratio: float = 0.02,
    min_y_gap_ratio: float = 0.015,
) -> List["LayoutRect"]:
    """
    Dynamic grid (XY-cut) merge to group rects into columns then rows using whitespace gaps.
    Focused on positioning accuracy by respecting detected gaps.
    """
    if not rects:
        return []

    min_x_gap = int(page_width * min_x_gap_ratio)
    min_y_gap = int(page_height * min_y_gap_ratio)

    # 1) Column cuts by x-axis gaps
    x_intervals = [(r.left, r.right) for r in rects]
    x_cuts = find_gaps_from_intervals(x_intervals, min_gap=min_x_gap, domain_end=page_width)
    x_boundaries = [0] + sorted(x_cuts) + [page_width]
    columns: List[List[LayoutRect]] = [[] for _ in range(len(x_boundaries) - 1)]

    for r in rects:
        center_x = r.left + r.width / 2
        for idx in range(len(x_boundaries) - 1):
            if x_boundaries[idx] <= center_x < x_boundaries[idx + 1]:
                columns[idx].append(r)
                break

    merged_rects: List[LayoutRect] = []

    # 2) Within each column, find row cuts and merge
    for col in columns:
        if not col:
            continue
        y_intervals = [(r.top, r.bottom) for r in col]
        y_cuts = find_gaps_from_intervals(y_intervals, min_gap=min_y_gap, domain_end=page_height)
        y_boundaries = [0] + sorted(y_cuts) + [page_height]

        for idx in range(len(y_boundaries) - 1):
            group: List[LayoutRect] = []
            y_min, y_max = y_boundaries[idx], y_boundaries[idx + 1]
            for r in col:
                center_y = r.top + r.height / 2
                if y_min <= center_y < y_max:
                    group.append(r)
            if group:
                merged_rects.append(merge_rect_group(group))

    return merged_rects


# ============================================================================
# 5. 텍스트 박스 추가 관련 헬퍼 메서드
# ============================================================================

def add_text_box(
    slide,
    layout_rect: "LayoutRect",
    text_style: "TextBoxStyle",
    slide_width: int,
    slide_height: int,
    image: Optional[Image.Image] = None,
    scale: float = 1.0,
    offset_x: int = 0,
    offset_y: int = 0,
    detected_style: Optional["DetectedStyle"] = None,
    vision_text_color: Optional[Tuple[int, int, int]] = None,
    cover_original: bool = True,
    enable_inpainting: bool = True,
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
    style_props = get_style_for_type(layout_rect.block_type)

    # Keep original width to ensure we cover the source text completely when masking
    # If not masking, allow tighter boxes (size already handled upstream)
    width = max(width, int(Pt(30)))
    height = max(height, int(Pt(15)))
    
    # Boundary Safety - keep box within slide
    left = max(0, min(left, slide_width - width))
    top = max(0, min(top, slide_height - height))

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
    bg_color = text_style.background_color
    text_color = text_style.text_color or vision_text_color
    if detected_style and detected_style.color_hex:
        text_color = hex_to_rgb(detected_style.color_hex)
    
    # Logic: Use Adaptive if (image exists AND style says None) OR (image exists AND forced adaptive in logic - NO, we want to respect style)
    # Updated Logic: 
    # If style.background_color is None -> Use Adaptive (requires image)
    # If style.background_color is Set -> Use Set Color
    
    if bg_color is None and image:
        # Adaptive Coloring
        bg_color = get_average_color(image, layout_rect, scale, offset_x, offset_y)
        text_color = get_optimal_text_color(bg_color)
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
        p.alignment = parse_alignment(detected_style.alignment)
    else:
        p.alignment = style_props["align"]
    
    run = p.add_run()
    run.text = layout_rect.text
    if detected_style and detected_style.font_category == "serif":
        run.font.name = "Times New Roman"
    else:
        run.font.name = text_style.font_name
    line_count = max(1, layout_rect.text.count("\n") + 1)
    # Derive font size from box height to prioritize fit inside the box
    derived_font_size_pt = max(10, (height / line_count) / 9525 * 0.6)
    run.font.size = Pt(derived_font_size_pt)
    run.font.bold = detected_style.is_bold if detected_style else style_props["bold"]
    run.font.color.rgb = RGBColor(*text_color)

    # Background handling: keep opaque only when masking is desired
    fill = textbox.fill
    if enable_inpainting or not cover_original:
        fill.background()
    else:
        fill.solid()
        fill.fore_color.rgb = RGBColor(*bg_color)


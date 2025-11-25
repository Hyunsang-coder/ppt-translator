"""PDF processing with Tesseract + OpenAI Vision hybrid OCR."""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from PIL import Image

LOGGER = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a text block with precise pixel coordinates."""
    
    text: str
    left: int      # pixels
    top: int       # pixels
    width: int     # pixels
    height: int    # pixels
    confidence: float = 1.0
    font_size: int = 16  # estimated pt
    block_id: int = 0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return (left, top, width, height) tuple."""
        return (self.left, self.top, self.width, self.height)


@dataclass
class PageOCRResult:
    """OCR result for a single PDF page."""
    
    page_number: int
    image: Image.Image
    text_blocks: List[TextBlock] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    def __post_init__(self):
        if self.image:
            self.image_width = self.image.width
            self.image_height = self.image.height


class PDFProcessor:
    """Process PDF files using Tesseract + OpenAI Vision hybrid approach."""

    GROUPING_PROMPT = """이 슬라이드 이미지를 분석해주세요. Tesseract OCR이 추출한 텍스트 블록 목록이 있습니다.

각 블록에 대해:
1. 텍스트 오류가 있으면 수정해주세요
2. 의미적으로 같은 문장/단락인 블록들을 그룹으로 묶어주세요
3. 각 그룹의 적절한 폰트 크기(pt)를 추정해주세요

OCR 추출 결과:
{ocr_blocks}

응답 형식 (JSON):
```json
{{
  "groups": [
    {{
      "block_ids": [0, 1, 2],
      "corrected_text": "보정된 전체 텍스트",
      "font_size": 24
    }}
  ]
}}
```

규칙:
- 모든 블록은 반드시 하나의 그룹에 포함되어야 함
- 같은 줄/문장의 단어들은 하나로 묶기
- 제목은 큰 폰트(28-40pt), 본문은 중간(14-20pt), 캡션은 작게(10-14pt)
- JSON만 출력, 다른 설명 없이"""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        dpi: int = 150,
    ) -> None:
        """Initialize PDF processor.

        Args:
            api_key: OpenAI API key for Vision.
            model: OpenAI model to use.
            dpi: Resolution for PDF to image conversion.
        """
        self._api_key = api_key
        self._model = model
        self._dpi = dpi

    def convert_pdf_to_images(
        self,
        pdf_buffer: io.BytesIO,
        max_pages: Optional[int] = None,
    ) -> List[Tuple[int, Image.Image]]:
        """Convert PDF pages to PIL Images."""
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError("PyMuPDF required: pip install PyMuPDF") from e

        pdf_buffer.seek(0)
        doc = fitz.open(stream=pdf_buffer.read(), filetype="pdf")

        images = []
        total_pages = len(doc)
        pages_to_process = min(total_pages, max_pages) if max_pages else total_pages

        LOGGER.info("Converting %d/%d PDF pages (DPI=%d)...", 
                    pages_to_process, total_pages, self._dpi)

        for page_num in range(pages_to_process):
            page = doc[page_num]
            zoom = self._dpi / 72
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append((page_num + 1, img))

        doc.close()
        LOGGER.info("PDF conversion complete: %d pages", len(images))
        return images

    def _extract_with_tesseract(
        self,
        image: Image.Image,
        min_confidence: float = 30.0,
    ) -> List[TextBlock]:
        """Extract text blocks with precise bounding boxes using Tesseract."""
        try:
            import pytesseract
        except ImportError as e:
            raise ImportError(
                "pytesseract required: pip install pytesseract\n"
                "Also install Tesseract: brew install tesseract tesseract-lang"
            ) from e

        # Get word-level data with bounding boxes
        data = pytesseract.image_to_data(
            image,
            lang="kor+eng",
            output_type=pytesseract.Output.DICT,
        )

        blocks = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            if not text or conf < min_confidence:
                continue

            block = TextBlock(
                text=text,
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                confidence=conf / 100.0,
                block_id=len(blocks),
            )
            blocks.append(block)

        LOGGER.info("Tesseract extracted %d text blocks", len(blocks))
        return blocks

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def _format_blocks_for_prompt(self, blocks: List[TextBlock]) -> str:
        """Format blocks for OpenAI prompt."""
        lines = []
        for b in blocks:
            lines.append(
                f"[{b.block_id}] \"{b.text}\" (left={b.left}, top={b.top}, "
                f"w={b.width}, h={b.height})"
            )
        return "\n".join(lines)

    def _extract_json_from_response(self, content: str) -> dict:
        """Extract JSON from OpenAI response."""
        content = content.strip()
        
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            LOGGER.error("Failed to parse JSON: %s", e)
            return {"groups": []}

    def _group_and_correct_with_openai(
        self,
        image: Image.Image,
        blocks: List[TextBlock],
    ) -> List[dict]:
        """Use OpenAI Vision to correct text and group blocks."""
        if not blocks:
            return []

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
        except ImportError as e:
            raise ImportError("langchain-openai required") from e

        base64_image = self._image_to_base64(image)
        blocks_text = self._format_blocks_for_prompt(blocks)
        prompt = self.GROUPING_PROMPT.format(ocr_blocks=blocks_text)

        model = ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            max_tokens=4096,
            temperature=0,
        )

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ]
        )

        try:
            response = model.invoke([message])
            data = self._extract_json_from_response(response.content)
            groups = data.get("groups", [])
            LOGGER.info("OpenAI created %d groups from %d blocks", len(groups), len(blocks))
            return groups
        except Exception as e:
            LOGGER.error("OpenAI grouping failed: %s", e)
            # Fallback: each block is its own group
            return [
                {"block_ids": [b.block_id], "corrected_text": b.text, "font_size": 16}
                for b in blocks
            ]

    def _merge_blocks(
        self,
        blocks: List[TextBlock],
        groups: List[dict],
    ) -> List[TextBlock]:
        """Merge blocks according to OpenAI grouping."""
        if not blocks or not groups:
            return blocks

        # Create lookup by block_id
        block_map = {b.block_id: b for b in blocks}
        merged = []

        for group in groups:
            block_ids = group.get("block_ids", [])
            if not block_ids:
                continue

            # Get blocks in this group
            group_blocks = [block_map[bid] for bid in block_ids if bid in block_map]
            if not group_blocks:
                continue

            # Calculate merged bounding box
            min_left = min(b.left for b in group_blocks)
            min_top = min(b.top for b in group_blocks)
            max_right = max(b.right for b in group_blocks)
            max_bottom = max(b.bottom for b in group_blocks)

            merged_block = TextBlock(
                text=group.get("corrected_text", " ".join(b.text for b in group_blocks)),
                left=min_left,
                top=min_top,
                width=max_right - min_left,
                height=max_bottom - min_top,
                confidence=sum(b.confidence for b in group_blocks) / len(group_blocks),
                font_size=group.get("font_size", 16),
                block_id=len(merged),
            )
            merged.append(merged_block)

        LOGGER.info("Merged %d blocks into %d groups", len(blocks), len(merged))
        return merged

    def process_pdf(
        self,
        pdf_buffer: io.BytesIO,
        max_pages: Optional[int] = None,
    ) -> List[PageOCRResult]:
        """Process PDF using Tesseract + OpenAI hybrid approach."""
        if not self._api_key:
            raise ValueError("OpenAI API key is required")

        images = self.convert_pdf_to_images(pdf_buffer, max_pages)
        results = []

        LOGGER.info("Processing %d pages with hybrid OCR...", len(images))

        for page_num, image in images:
            LOGGER.info("Page %d: Extracting with Tesseract...", page_num)
            
            # Step 1: Tesseract extraction (precise positions)
            raw_blocks = self._extract_with_tesseract(image)
            
            if not raw_blocks:
                LOGGER.warning("Page %d: No text found", page_num)
                results.append(PageOCRResult(
                    page_number=page_num,
                    image=image,
                    text_blocks=[],
                ))
                continue

            # Step 2: OpenAI grouping and correction
            LOGGER.info("Page %d: Grouping with OpenAI Vision...", page_num)
            groups = self._group_and_correct_with_openai(image, raw_blocks)

            # Step 3: Merge blocks
            merged_blocks = self._merge_blocks(raw_blocks, groups)

            result = PageOCRResult(
                page_number=page_num,
                image=image,
                text_blocks=merged_blocks,
            )
            results.append(result)

            LOGGER.info(
                "Page %d: %d raw blocks → %d merged blocks",
                page_num,
                len(raw_blocks),
                len(merged_blocks),
            )

        return results

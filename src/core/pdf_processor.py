"""PDF processing using Pure OpenAI Vision for semantic layout analysis."""

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
    """Represents a semantic text block identified by Vision."""
    
    text: str
    left: int      # pixels
    top: int       # pixels
    width: int     # pixels
    height: int    # pixels
    block_type: str = "body"  # title, subtitle, body, list, caption
    text_color: Tuple[int, int, int] = (0, 0, 0)  # RGB
    
    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


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
    """Process PDF files using OpenAI Vision for human-like layout analysis."""

    # 프롬프트 설계: 구조와 의미 중심
    LAYOUT_PROMPT = """
    You are an expert document layout analyzer. 
    Analyze the provided slide image and extract all text content grouped by their semantic structure.

    Task:
    1. Identify logical text blocks (e.g., a full title, a complete paragraph, a list item).
    2. Do NOT split sentences or paragraphs into small pieces. Keep them as one block.
    3. Classify each block type: "title", "subtitle", "body", "list", "caption".
    4. Estimate the bounding box for each block using a 0-1000 normalized scale (0,0 is top-left, 1000,1000 is bottom-right).
    5. Extract the exact text content.
    6. Estimate the text color (hex code).

    Output JSON Format:
    {
        "blocks": [
            {
                "type": "title",
                "text": "GROUNDED IN PUBG LORE",
                "box_2d": [ymin, xmin, ymax, xmax],  // 0-1000 scale
                "color": "#FFA500"
            },
            {
                "type": "body",
                "text": "Expanding on the existing lore in the PUBG universe...",
                "box_2d": [ymin, xmin, ymax, xmax],
                "color": "#FFFFFF"
            }
        ]
    }
    
    Important:
    - "box_2d" must be [ymin, xmin, ymax, xmax].
    - Ensure the bounding box covers the text area generously.
    - Return ONLY valid JSON.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.1",
        dpi: int = 200,
    ) -> None:
        """Initialize PDF processor."""
        self._api_key = api_key
        self._model = model
        self._dpi = dpi

    # Image size limits for Vision API safety
    MAX_IMAGE_DIMENSION = 4096  # pixels
    MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20MB

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
        try:
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
        finally:
            doc.close()

        return images

    def _validate_and_resize_image(self, image: Image.Image) -> Image.Image:
        """Validate image size and resize if necessary to prevent memory issues."""
        width, height = image.size

        # Check if resize is needed
        if width > self.MAX_IMAGE_DIMENSION or height > self.MAX_IMAGE_DIMENSION:
            # Calculate new dimensions maintaining aspect ratio
            ratio = min(self.MAX_IMAGE_DIMENSION / width, self.MAX_IMAGE_DIMENSION / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            LOGGER.warning(
                "Image too large (%dx%d), resizing to %dx%d",
                width, height, new_width, new_height
            )
            try:
                resample = Image.Resampling.LANCZOS
            except AttributeError:
                resample = Image.LANCZOS
            image = image.resize((new_width, new_height), resample)

        return image

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string with size validation."""
        # Validate and resize if necessary
        image = self._validate_and_resize_image(image)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        encoded = base64.b64encode(buffer.read()).decode("utf-8")

        # Check encoded size
        if len(encoded) > self.MAX_IMAGE_BYTES:
            LOGGER.warning(
                "Base64 image size (%d bytes) exceeds limit, reducing quality",
                len(encoded)
            )
            # Retry with JPEG and lower quality
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode("utf-8")

        return encoded

    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex string to RGB tuple."""
        try:
            hex_color = hex_color.lstrip("#")
            if len(hex_color) == 3:
                hex_color = "".join(c * 2 for c in hex_color)
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
        except Exception:
            return (0, 0, 0)  # Default black

    def _analyze_layout_with_vision(self, image: Image.Image) -> List[TextBlock]:
        """Analyze layout and extract text using OpenAI Vision."""
        if not self._api_key:
            raise ValueError("OpenAI API Key is required for Vision processing.")

        try:
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage
        except ImportError as e:
            raise ImportError("langchain-openai required") from e

        base64_image = self._image_to_base64(image)
        
        chat = ChatOpenAI(
            model=self._model,
            api_key=self._api_key,
            temperature=0.0,
            max_tokens=4096,
            model_kwargs={"response_format": {"type": "json_object"}}
        )

        LOGGER.info("Sending image to OpenAI Vision for layout analysis...")
        
        try:
            response = chat.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": self.LAYOUT_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ])
            ])
            
            result = json.loads(response.content)
            raw_blocks = result.get("blocks", [])
            
            text_blocks = []
            img_w, img_h = image.size
            
            LOGGER.info("Image dimensions: %d x %d pixels", img_w, img_h)

            for idx, b in enumerate(raw_blocks):
                # Parse normalized coordinates (0-1000)
                # box_2d: [ymin, xmin, ymax, xmax]
                raw_box = b.get("box_2d", [0, 0, 0, 0])
                text_content = b.get("text", "").strip()
                block_type = b.get("type", "body")
                
                # Debug: Log raw Vision response
                LOGGER.info(
                    "Block %d [%s]: raw_box=%s, text='%s'",
                    idx, block_type, raw_box, text_content[:50] if len(text_content) > 50 else text_content
                )
                
                # Ensure we have exactly 4 values
                if len(raw_box) != 4:
                    LOGGER.warning("Block %d has invalid box_2d length: %s", idx, raw_box)
                    continue
                
                ymin, xmin, ymax, xmax = raw_box
                
                # Convert to pixels
                left = int(xmin * img_w / 1000)
                top = int(ymin * img_h / 1000)
                right = int(xmax * img_w / 1000)
                bottom = int(ymax * img_h / 1000)
                
                # Debug: Log converted pixel coordinates
                LOGGER.info(
                    "Block %d: normalized (ymin=%d, xmin=%d, ymax=%d, xmax=%d) -> pixels (left=%d, top=%d, right=%d, bottom=%d)",
                    idx, ymin, xmin, ymax, xmax, left, top, right, bottom
                )
                
                # Validate dimensions
                width = max(1, right - left)
                height = max(1, bottom - top)
                
                text_blocks.append(TextBlock(
                    text=text_content,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    block_type=block_type,
                    text_color=self._hex_to_rgb(b.get("color", "#000000"))
                ))
            
            LOGGER.info("Vision detected %d semantic text blocks.", len(text_blocks))
            return text_blocks

        except Exception as e:
            LOGGER.error("Vision layout analysis failed: %s", e)
            return []

    def process_pdf(
        self,
        pdf_buffer: io.BytesIO,
        use_openai: bool = True,  # Ignored, always True in this mode
        max_pages: Optional[int] = None,
    ) -> List[PageOCRResult]:
        """Process PDF using pure Vision approach."""
        images = self.convert_pdf_to_images(pdf_buffer, max_pages)
        results = []

        LOGGER.info("Processing %d pages with Vision-First architecture...", len(images))

        for page_num, image in images:
            LOGGER.info("Page %d: Analyzing layout & content...", page_num)
            
            blocks = self._analyze_layout_with_vision(image)
            results.append(PageOCRResult(page_num, image, blocks))

        return results

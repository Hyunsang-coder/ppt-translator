# Refine Text Box Sizing and Color Logic

This plan addresses the issues of conflicting color settings and oversized text boxes observed in test results.

## Key Changes
1.  **Enforce User Color Settings**: Prioritize user-defined background and text colors over Vision-detected colors to ensure consistency and readability (opaque background is critical for translation).
2.  **Optimize Text Box Width**: Implement a heuristic to calculate the expected width of the text based on character count and font size. If the calculated width is significantly smaller than the Vision-detected width (common for short text or titles), shrink the text box width to fit the content, reducing visual clutter.

## Implementation Steps

1.  **Update `PDFToPPTWriter` Color Logic**
    - Modify `_add_text_box` to **ignore** `block.text_color` from Vision.
    - Always use `self._text_style.text_color` and `self._text_style.background_color`.

2.  **Implement Width Optimization Heuristic**
    - In `_add_text_box`, calculate `estimated_text_width`:
        - Formula: `len(block.text) * font_size_in_emu * 0.7` (Hangul/Asian chars are wide, but 0.7 factor averages out width since font size is height).
        - *Note*: This is an estimation. We will use a safe multiplier.
    - Compare `estimated_text_width` with `width` (from Vision).
    - If `estimated_text_width < width`, update `width = max(estimated_text_width, min_width)`.
    - Ensure `word_wrap = True` is preserved for safety.

3.  **Update `tests/test_pdf_to_ppt_layout.py`**
    - Add a test case `test_text_box_width_optimization` to verify that a short text in a wide container gets resized.
    - Add a test case `test_color_enforcement` to verify that user colors are applied regardless of input block color.

4.  **Verification**
    - Run the updated tests.





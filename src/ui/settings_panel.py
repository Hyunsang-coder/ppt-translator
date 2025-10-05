"""Streamlit settings panel for the PPT translator app."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
from streamlit.delta_generator import DeltaGenerator


_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "glossary_template.xlsx"


def _get_glossary_template_bytes() -> bytes:
    """Read the glossary template file into memory.

    Returns:
        Binary content of the template file, or an empty byte string if missing.
    """

    try:
        return _TEMPLATE_PATH.read_bytes()
    except FileNotFoundError:
        st.warning("용어집 템플릿 파일이 아직 생성되지 않았습니다.")
        return b""


def _get_target(container: Optional[DeltaGenerator]) -> DeltaGenerator:
    return container if container is not None else st


def render_settings(container: Optional[DeltaGenerator] = None) -> Dict[str, Any]:
    """Render the translation settings panel and return the user selections."""

    target = _get_target(container)

    target.subheader("⚙️ 번역 설정")
    expander = target.expander("설정 열기", expanded=True)

    col1, col2 = expander.columns(2)

    source_lang = col1.selectbox(
        "소스 언어",
        ["Auto", "한국어", "영어", "일본어", "중국어", "스페인어", "프랑스어", "독일어"],
        index=0,
        help="'Auto'를 선택하면 자동으로 감지합니다",
    )

    target_lang = col2.selectbox(
        "타겟 언어",
        ["Auto", "한국어", "영어", "일본어", "중국어", "스페인어", "프랑스어", "독일어"],
        index=0,
        help="'Auto'를 선택하면 소스 언어의 반대로 추론합니다",
    )

    model = expander.selectbox(
        "모델 선택",
        ["gpt-5", "gpt-5-mini"],
        index=0,
        help="gpt-5: 최고 품질 (느림, 비쌈) | gpt-5-mini: 빠르고 저렴",
    )

    user_prompt = expander.text_area(
        "커스텀 프롬프트 (선택)",
        placeholder="예: 슬라이드 간 일관성 유지",
        height=100,
    )

    expander.markdown("---")

    glossary_file = expander.file_uploader(
        "📊 용어집 업로드 (선택)",
        type=["xlsx"],
        help="A열: 원문, B열: 번역",
    )

    template_bytes = _get_glossary_template_bytes()
    expander.download_button(
        "📥 용어집 템플릿 다운로드",
        data=template_bytes,
        file_name="glossary_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        disabled=not template_bytes,
    )

    return {
        "source_lang": source_lang,
        "target_lang": target_lang,
        "model": model,
        "user_prompt": user_prompt,
        "glossary_file": glossary_file,
    }

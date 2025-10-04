"""Streamlit settings panel for the PPT translator app."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import streamlit as st


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


def render_settings() -> Dict[str, Any]:
    """Render the translation settings panel and return the user selections.

    Returns:
        Dictionary containing the user's language, model, prompt, and glossary
        selections.
    """

    st.subheader("⚙️ 번역 설정")

    with st.expander("설정 열기", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            source_lang = st.selectbox(
                "소스 언어",
                ["Auto", "한국어", "영어", "일본어", "중국어", "스페인어", "프랑스어", "독일어"],
                index=0,
                help="'Auto'를 선택하면 자동으로 감지합니다",
            )

        with col2:
            target_lang = st.selectbox(
                "타겟 언어",
                ["Auto", "한국어", "영어", "일본어", "중국어", "스페인어", "프랑스어", "독일어"],
                index=0,
                help="'Auto'를 선택하면 소스 언어의 반대로 추론합니다",
            )

        model = st.selectbox(
            "모델 선택",
            ["gpt-5", "gpt-5-mini"],
            index=0,
            help="gpt-5: 최고 품질 (느림, 비쌈) | gpt-5-mini: 빠르고 저렴",
        )

        user_prompt = st.text_area(
            "커스텀 프롬프트 (선택)",
            placeholder="예: 전문적이고 격식있는 톤으로 번역해주세요",
            height=100,
        )

        st.markdown("---")

        glossary_file = st.file_uploader(
            "📊 용어집 업로드 (선택)",
            type=["xlsx"],
            help="A열: 원문, B열: 번역",
        )

        template_bytes = _get_glossary_template_bytes()
        st.download_button(
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

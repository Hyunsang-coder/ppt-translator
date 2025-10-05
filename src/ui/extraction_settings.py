"""Sidebar controls for PPT text extraction."""

from __future__ import annotations

from typing import Tuple

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

from src.core.text_extractor import ExtractionOptions


_DEF_FIG_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("생략", "omit"),
    ("플레이스홀더", "placeholder"),
)

_DEF_CHART_OPTIONS: Tuple[Tuple[str, str], ...] = (
    ("제목만", "labels"),
    ("플레이스홀더", "placeholder"),
    ("생략", "omit"),
)


def render_extraction_settings(container: DeltaGenerator | None = None) -> ExtractionOptions:
    """Render extraction options on the provided container."""

    target = container if container is not None else st.sidebar

    target.subheader("📝 텍스트 추출 옵션")

    include_notes = target.checkbox("발표자 노트 포함", value=False)

    figure_labels = [label for label, _ in _DEF_FIG_OPTIONS]
    figure_values = {label: value for label, value in _DEF_FIG_OPTIONS}
    figure_choice = target.selectbox("그림 처리", figure_labels, index=0)

    chart_labels = [label for label, _ in _DEF_CHART_OPTIONS]
    chart_values = {label: value for label, value in _DEF_CHART_OPTIONS}
    chart_choice = target.selectbox("차트 처리", chart_labels, index=0)

    return ExtractionOptions(
        with_notes=include_notes,
        figures=figure_values[figure_choice],
        charts=chart_values[chart_choice],
    )

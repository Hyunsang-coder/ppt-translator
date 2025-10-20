"""File handling helpers to keep uploaded files in memory."""

from __future__ import annotations

import io
from typing import Optional

import streamlit as st


def handle_upload(uploaded_file, max_size_mb: int = 50) -> Optional[io.BytesIO]:
    """Validate and cache the uploaded PPT file in memory.

    Args:
        uploaded_file: Streamlit uploaded file reference.
        max_size_mb: Maximum allowed file size in megabytes.

    Returns:
        BytesIO buffer of the uploaded file or ``None`` when validation fails.
    """

    if uploaded_file is None:
        return None

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > max_size_mb:
        st.error(f"파일 크기가 {max_size_mb}MB를 초과합니다. 더 작은 파일을 업로드해주세요.")
        return None

    st.session_state.pop("uploaded_ppt_bytes", None)
    uploaded_file.seek(0)
    buffer = io.BytesIO(uploaded_file.read())
    buffer.seek(0)
    st.session_state["uploaded_ppt_name"] = uploaded_file.name
    return buffer

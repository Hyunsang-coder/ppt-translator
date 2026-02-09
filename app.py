"""Redirect page for deprecated Streamlit app."""

import streamlit as st

NEW_URL = "https://ppt-translator.vercel.app/translate"

st.set_page_config(page_title="PPT 번역캣 - Moved", page_icon="🐱")

st.markdown(
    f"""
    <div style="display: flex; flex-direction: column; align-items: center;
                justify-content: center; min-height: 60vh; text-align: center;">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🐱 PPT 번역캣</h1>
        <p style="font-size: 1.25rem; color: gray; margin-bottom: 2rem;">
            이 페이지는 더 이상 사용되지 않습니다.<br>
            This app has moved to a new home.
        </p>
        <a href="{NEW_URL}" target="_self"
           style="background-color: #FF6B35; color: white; padding: 0.75rem 2rem;
                  border-radius: 8px; text-decoration: none; font-size: 1.1rem;
                  font-weight: 600;">
            새 사이트로 이동 &rarr;
        </a>
        <p style="margin-top: 1.5rem; color: gray; font-size: 0.9rem;">
            <a href="{NEW_URL}" style="color: #FF6B35;">{NEW_URL}</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

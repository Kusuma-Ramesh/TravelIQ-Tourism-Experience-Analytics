"""
TravelIQ — Tourism Experience Analytics
Classification, Prediction & Recommendation System

Part 1: Frontend foundation & application architecture.
No data cleaning, EDA, or ML happens in this part — every dynamic
value on screen is explicitly-labelled demo data until the later
parts wire in the real pipeline.
"""

from pathlib import Path

import streamlit as st

from components.navigation import NAV_ITEMS
from components.sidebar import render_sidebar

ROOT = Path(__file__).parent

st.set_page_config(
    page_title="TravelIQ — Tourism Experience Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = ROOT / "styles" / "main.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


load_css()

pages = [
    st.Page(item.file, title=item.label, icon=item.icon, default=item.default)
    for item in NAV_ITEMS
]

pg = st.navigation(pages, position="hidden")

render_sidebar()

pg.run()

"""Home — hero, real tourism KPIs, dashboard insights, and world explorer."""

import pandas as pd
import streamlit as st
import textwrap
from components.cards import render_kpi_row, section_title
from components.charts import demo_bar_ranking
from components.world_map import render_explore_the_world

CONSOLIDATED_CSV_PATH = "data/processed/consolidated.csv"


@st.cache_data
def load_home_data():
    return pd.read_csv(CONSOLIDATED_CSV_PATH)


# ---------------------------------------------------------------------------
# Real project data
# ---------------------------------------------------------------------------

df = load_home_data()

total_travelers = df["UserId"].nunique()
average_rating = df["Rating"].mean()
total_attractions = df["AttractionId"].nunique()
total_visits = len(df)

top_destinations = (
    df.groupby("AttractionCityName")
    .size()
    .sort_values(ascending=False)
    .head(5)
    .sort_values()
)

top_attractions = (
    df.groupby(["AttractionId", "Attraction", "AttractionCityName"])
    .size()
    .sort_values(ascending=False)
    .head(5)
    .sort_values()
)

attraction_summary = (
    df.groupby(
        ["AttractionId", "Attraction", "AttractionCityName", "AttractionTypeName"],
        as_index=False,
    )
    .agg(
        average_rating=("Rating", "mean"),
        visit_count=("UserId", "size"),
    )
)

next_adventure = (
    attraction_summary.sort_values(
        ["average_rating", "visit_count"],
        ascending=[False, False],
    )
    .iloc[0]
)


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

HERO_SVG = """
<svg class="tiq-hero-decor" viewBox="0 0 1000 260" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMaxYMid slice">
  <defs>
    <linearGradient id="mtn1" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#8B6FF0" stop-opacity="0.20"/>
      <stop offset="100%" stop-color="#8B6FF0" stop-opacity="0.0"/>
    </linearGradient>
    <linearGradient id="mtn2" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#4CB8F0" stop-opacity="0.18"/>
      <stop offset="100%" stop-color="#4CB8F0" stop-opacity="0.0"/>
    </linearGradient>
  </defs>

  <polygon points="620,260 700,140 760,220 820,110 900,260" fill="url(#mtn2)"/>
  <polygon points="560,260 660,170 730,260" fill="url(#mtn1)"/>

  <g opacity="0.35" fill="#F3F6FB">
    <ellipse cx="760" cy="55" rx="34" ry="13"/>
    <ellipse cx="785" cy="48" rx="26" ry="11"/>
    <ellipse cx="880" cy="90" rx="28" ry="10"/>
    <ellipse cx="900" cy="84" rx="20" ry="9"/>
  </g>

  <path d="M 640 210 Q 760 130 900 60" stroke="#F0A24C" stroke-width="2" stroke-dasharray="6 8" fill="none" opacity="0.65"/>

  <g transform="translate(636,204)">
    <circle r="5" fill="#F2685C"/>
    <circle r="9" fill="none" stroke="#F2685C" stroke-width="1.4" opacity="0.6"/>
  </g>

  <g transform="translate(898,58)">
    <circle r="5" fill="#2FD9A8"/>
    <circle r="9" fill="none" stroke="#2FD9A8" stroke-width="1.4" opacity="0.6"/>
  </g>

  <g transform="translate(770,140) rotate(-28)" fill="#F3F6FB" opacity="0.85">
    <path d="M0 0 L18 2 L26 -1 L18 -4 L0 -6 L-6 -10 L-10 -9 L-7 -4 L-16 -3 L-19 -6 L-22 -5 L-19 0 L-22 5 L-19 6 L-16 3 L-7 4 L-10 9 L-6 10 L0 6 L18 4 L26 1 L18 -2 Z" transform="scale(0.9)"/>
  </g>
</svg>
"""

POLAROID_STACK = """
<div class="tiq-polaroid-stack">
    <div class="tiq-polaroid tiq-polaroid-1">
        <div class="photo" style="background:linear-gradient(140deg,#8B6FF0,#4CB8F0);"></div>
    </div>
    <div class="tiq-polaroid tiq-polaroid-2">
        <div class="photo" style="background:linear-gradient(140deg,#F0A24C,#F2685C);"></div>
    </div>
    <div class="tiq-polaroid tiq-polaroid-3">
        <div class="photo" style="background:linear-gradient(140deg,#2FD9A8,#4CB8F0);"></div>
    </div>
</div>
"""

hero_html = f"""<div class="tiq-hero">
{HERO_SVG}
{POLAROID_STACK}
<div class="tiq-eyebrow">✈️ TravelIQ Dashboard</div>
<div class="tiq-hero-title">Where will your next<br/><span class="accent">adventure</span> take you?</div>
<div class="tiq-subtitle">Discover smarter. Travel better.</div>
</div>"""

st.markdown(hero_html, unsafe_allow_html=True)

col_a, col_b, _ = st.columns([1.1, 1, 2.3])

with col_a:
    if st.button("Discover My Journey →", width="stretch", key="hero_cta_primary"):
        st.switch_page("pages/passport.py")

with col_b:
    st.markdown('<div class="tiq-btn-secondary">', unsafe_allow_html=True)
    if st.button("Explore Analytics", width="stretch", key="hero_cta_secondary"):
        st.switch_page("pages/analytics.py")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")


# ---------------------------------------------------------------------------
# KPI row — real project data
# ---------------------------------------------------------------------------

render_kpi_row(
    [
        {
            "icon": "👥",
            "label": "Total Travelers",
            "value": f"{total_travelers:,}",
            "delta": "Unique travelers in dataset",
            "color": "violet",
        },
        {
            "icon": "⭐",
            "label": "Average Rating",
            "value": f"{average_rating:.2f} / 5",
            "delta": "Average recorded rating",
            "color": "sky",
        },
        {
            "icon": "📍",
            "label": "Attractions",
            "value": f"{total_attractions:,}",
            "delta": "Attractions in dataset",
            "color": "emerald",
        },
        {
            "icon": "🧳",
            "label": "Total Visits",
            "value": f"{total_visits:,}",
            "delta": "Recorded visits",
            "color": "orange",
        },
    ]
)

st.write("")


# ---------------------------------------------------------------------------
# Dashboard insight panels
# ---------------------------------------------------------------------------

col1, col2, col3 = st.columns([1.15, 1.15, 1])

with col1:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("🌐", "Top Destinations")
    st.caption("Ranked by recorded visits")
    demo_bar_ranking(
        labels=top_destinations.index.tolist(),
        values=top_destinations.values.tolist(),
        height=240,
    )
    st.markdown("</div>", unsafe_allow_html=True)


with col2:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("🔥", "Trending Attractions")
    st.caption("Most visited attractions in the dataset")
    demo_bar_ranking(
        labels=[
            f"{attraction} — {city}"
            for attraction_id, attraction, city in top_attractions.index
        ],
        values=top_attractions.values.tolist(),
        height=240,
    )
    st.markdown("</div>", unsafe_allow_html=True)


with col3:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("✨", "Your Next Adventure")
    st.markdown('<span class="tiq-badge">Top Rated Pick</span>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div style="margin-top:0.9rem;">
            <div style="font-weight:800; font-size:1.05rem; color:var(--text-primary);">
                {next_adventure["Attraction"]}
            </div>
            <div style="font-size:0.8rem; color:var(--text-muted); margin-bottom:0.6rem;">
                {next_adventure["AttractionTypeName"]} • {next_adventure["AttractionCityName"]}
            </div>
            <div style="font-size:0.82rem; color:var(--accent-orange); margin-bottom:0.7rem;">
                ★ {next_adventure["average_rating"]:.1f}
                • {int(next_adventure["visit_count"]):,} visits
            </div>
        </div>
        """,
        unsafe_allow_html=True,

    )

    st.markdown(
        f"""
        <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:0.25rem;">
            Dataset Rating
        </div>
        <div class="tiq-progress-track">
            <div class="tiq-progress-fill"
                 style="width:{min(100, (next_adventure['average_rating'] / 5) * 100):.0f}%; background:var(--gradient-primary);">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button(
        "Explore Recommendation →",
        width="stretch",
        key="home_reco_cta",
    ):
        st.switch_page("pages/recommendations.py")

    st.markdown("</div>", unsafe_allow_html=True)


st.write("")


# ---------------------------------------------------------------------------
# Explore the World — interactive map + dataset-driven attraction explorer
# ---------------------------------------------------------------------------

render_explore_the_world()
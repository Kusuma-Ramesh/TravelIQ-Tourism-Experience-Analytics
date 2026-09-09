"""
Tourism Analytics — real dataset-driven analytics.

Every KPI, chart, and insight on this page is read from the project's
own validated, leakage-safe EDA artifacts (docs/eda/eda_summary_stats.json
and docs/eda/tables/*.csv), produced earlier in the pipeline and accepted
at Checkpoint 08. Nothing here is hardcoded demo data, and no EDA is
recomputed on this page -- see components/analytics_data.py for the
(cached, read-only) data access layer.
"""

import streamlit as st

from components.analytics_data import (
    MONTH_NAMES,
    kpis,
    load_summary,
    ordered_month_counts,
    ordered_year_counts,
)
from components.cards import empty_state, glass_card_close, glass_card_open, page_header, render_kpi_row, section_title
from components.charts import demo_bar_ranking, demo_donut, demo_trend_chart

page_header(
    eyebrow="📊 Analytics",
    title="Tourism Analytics",
    subtitle="Trends, demographics, and visit patterns across the dataset.",
)

# ---------------------------------------------------------------------------
# Load real data (cached) -- gracefully degrade if artifacts are missing
# ---------------------------------------------------------------------------
try:
    summary = load_summary()
except FileNotFoundError:
    summary = None

if summary is None:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    empty_state(
        icon="📈",
        title="Analytics data not found",
        subtitle="Couldn't find docs/eda/eda_summary_stats.json. Run the EDA pipeline "
        "(scripts/eda_analysis.py) to generate it, then reload this page.",
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

overview = summary["dataset_overview"]
traveler = summary["traveler_analysis"]
attraction = summary["attraction_analysis"]
temporal = summary["temporal_analysis"]
rating = summary["rating_analysis"]
insights = summary.get("business_insights", [])
k = kpis(summary)

# ---------------------------------------------------------------------------
# 1-4. Headline KPIs
# ---------------------------------------------------------------------------
render_kpi_row(
    [
        {
            "icon": "🧾",
            "label": "Total Transactions",
            "value": f"{k['total_transactions']:,}",
            "delta": f"{k['year_min']}–{k['year_max']}",
            "color": "violet",
        },
        {
            "icon": "👥",
            "label": "Unique Travelers",
            "value": f"{k['unique_travelers']:,}",
            "delta": f"From {k['n_countries']} countries",
            "color": "sky",
        },
        {
            "icon": "📍",
            "label": "Attractions",
            "value": f"{k['n_attractions']}",
            "delta": f"{k['n_attraction_types']} attraction types",
            "color": "emerald",
        },
        {
            "icon": "⭐",
            "label": "Average Rating",
            "value": f"{k['avg_rating']:.2f} / 5",
            "delta": f"{k['pct_4_or_5']:.1f}% rated 4★ or 5★",
            "color": "orange",
        },
    ]
)

st.write("")

# ---------------------------------------------------------------------------
# 5 & 8. Visit-mode distribution + rating distribution
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    glass_card_open()
    section_title("🧳", "Visit Mode Distribution")
    mode_counts = traveler["visit_mode_counts"]
    demo_donut(labels=list(mode_counts.keys()), values=list(mode_counts.values()), height=260)
    glass_card_close()

with col2:
    glass_card_open()
    section_title("⭐", "Rating Distribution")
    rating_items = sorted(rating["rating_counts"].items(), key=lambda kv: int(kv[0]))
    demo_bar_ranking(
        labels=[f"{r}★" for r, _ in rating_items],
        values=[c for _, c in rating_items],
        height=260,
    )
    glass_card_close()

st.write("")

# ---------------------------------------------------------------------------
# 6. Traveler demographics / geography
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    glass_card_open()
    section_title("🌍", "Travelers by Continent")
    continent_counts = traveler["continent_counts"]
    demo_donut(labels=list(continent_counts.keys()), values=list(continent_counts.values()), height=260)
    glass_card_close()

with col2:
    glass_card_open()
    section_title("🏳️", "Top Traveler Countries")
    top_countries = list(traveler["top_15_user_countries"].items())[:5]
    demo_bar_ranking(
        labels=[c for c, _ in top_countries][::-1],
        values=[v for _, v in top_countries][::-1],
        height=260,
    )
    glass_card_close()

st.write("")

# ---------------------------------------------------------------------------
# 7. Attraction popularity
# ---------------------------------------------------------------------------
glass_card_open()
section_title("🔥", "Most Visited Attractions")
top_attractions = list(attraction["top_10_attractions_by_visits"].items())[:5]
st.caption(
    f"Demand is concentrated across only {k['n_attractions']} attractions in the dataset."
)
demo_bar_ranking(
    labels=[a for a, _ in top_attractions][::-1],
    values=[v for _, v in top_attractions][::-1],
    height=260,
)
glass_card_close()

st.write("")

# ---------------------------------------------------------------------------
# 9 & 10. Yearly and monthly visit trends
# ---------------------------------------------------------------------------
peak_month_name = MONTH_NAMES[temporal["peak_month"] - 1]

col1, col2 = st.columns(2)

with col1:
    glass_card_open()
    section_title("📈", "Yearly Visit Trend")
    yr_labels, yr_values = ordered_year_counts(temporal["yearly_visit_counts"])
    st.caption(f"Peak year: {temporal['peak_year']} · Lowest: {temporal['lowest_visit_year']} (COVID-19 disruption)")
    demo_trend_chart(values=yr_values, labels=yr_labels, color="#4CB8F0", height=200)
    glass_card_close()

with col2:
    glass_card_open()
    section_title("🗓️", "Monthly Visit Seasonality")
    mo_labels, mo_values = ordered_month_counts(temporal["monthly_visit_counts"])
    st.caption(f"Peak month: {peak_month_name}")
    demo_trend_chart(values=mo_values, labels=mo_labels, color="#F0A24C", height=200)
    glass_card_close()

st.write("")

# ---------------------------------------------------------------------------
# 11. Additional EDA insights: rating trend over time, top/bottom rated
#     attractions, and the dataset's own business insights.
# ---------------------------------------------------------------------------
glass_card_open()
section_title("💹", "Average Rating Over Time")
yrr_labels, yrr_values = ordered_year_counts(temporal["yearly_avg_rating"])
demo_trend_chart(values=yrr_values, labels=yrr_labels, color="#2FD9A8", height=180)
glass_card_close()

st.write("")

col1, col2 = st.columns(2)

with col1:
    glass_card_open()
    section_title("🏆", "Highest-Rated Attractions")
    st.caption("Minimum sample size applied for a fair ranking.")
    top_rated = list(attraction["top_rated_attractions_min50"].items())[:5]
    for name, score in top_rated:
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; padding:0.45rem 0; '
            f'border-bottom:1px solid rgba(255,255,255,0.06); font-size:0.88rem; color:var(--text-secondary);">'
            f"<span>{name}</span><span style='color:#2FD9A8; font-weight:700;'>★ {score:.2f}</span></div>",
            unsafe_allow_html=True,
        )
    glass_card_close()

with col2:
    glass_card_open()
    section_title("🔧", "Room-to-Improve Attractions")
    st.caption("Minimum sample size applied for a fair ranking.")
    bottom_rated = list(attraction["bottom_rated_attractions_min50"].items())[:5]
    for name, score in bottom_rated:
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; padding:0.45rem 0; '
            f'border-bottom:1px solid rgba(255,255,255,0.06); font-size:0.88rem; color:var(--text-secondary);">'
            f"<span>{name}</span><span style='color:#F2685C; font-weight:700;'>★ {score:.2f}</span></div>",
            unsafe_allow_html=True,
        )
    glass_card_close()

st.write("")

if insights:
    glass_card_open()
    section_title("🧠", "Key Insights")
    bullets_html = "".join(
        f'<li style="margin-bottom:0.6rem; line-height:1.6;">{text}</li>' for text in insights
    )
    st.markdown(
        f'<ul style="color:var(--text-secondary); font-size:0.88rem; padding-left:1.1rem; margin:0;">{bullets_html}</ul>',
        unsafe_allow_html=True,
    )
    glass_card_close()

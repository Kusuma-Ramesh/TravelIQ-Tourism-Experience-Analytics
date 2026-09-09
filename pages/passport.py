"""Memory Passport — the signature scrapbook-style travel history page."""

import calendar

import pandas as pd
import streamlit as st

from components.cards import page_header, section_title
from components.passport import stamp_grid

CONSOLIDATED_CSV_PATH = "data/processed/consolidated.csv"

page_header(
    eyebrow="🛂 Your Journey",
    title="Memory Passport",
)

# Traveler ID input — for now this just captures the ID; no lookups or
# stats are wired to it yet.
traveler_id = st.number_input(
    "Traveler ID",
    min_value=1,
    step=1,
    value=1,
    key="traveler_id",
)

# Load consolidated dataset and filter to the selected traveler.
# Only builds the filtered dataframe for now — not yet wired into the UI.
consolidated_df = pd.read_csv(CONSOLIDATED_CSV_PATH)
traveler_df = consolidated_df[consolidated_df["UserId"] == traveler_id]

if traveler_df.empty:
    st.warning(f"No travel records were found for Traveler ID {traveler_id}.")
    st.stop()

# Calculated stats from traveler_df — not yet wired into the UI.
total_visits = len(traveler_df)
attractions_explored = traveler_df["AttractionId"].nunique()
cities_explored = traveler_df["AttractionCityId"].nunique()
attraction_types_explored = traveler_df["AttractionTypeId"].nunique()
average_rating = traveler_df["Rating"].mean()
most_common_visit_mode = (
    traveler_df["VisitModeName"].mode().iloc[0]
    if not traveler_df["VisitModeName"].mode().empty
    else None
)
years_visited = traveler_df["VisitYear"].nunique()

# Real stamps generated from traveler_df — one stamp per unique
# AttractionTypeName + AttractionCityName combination (not per transaction),
# keeping the first real VisitMonth/VisitYear seen for each combination,
# limited to 8 stamps max.
unique_stamp_rows = traveler_df.drop_duplicates(
    subset=["AttractionTypeName", "AttractionCityName"], keep="first"
).head(8)

traveler_stamps = [
    {
        "country": row["AttractionTypeName"],
        "city": row["AttractionCityName"],
        "status": "visited",
        "date": f"{calendar.month_abbr[int(row['VisitMonth'])]} {int(row['VisitYear'])}",
    }
    for _, row in unique_stamp_rows.iterrows()
]

col_main, col_side = st.columns([2, 1])

with col_main:
    st.markdown('<div class="tiq-passport-shell">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tiq-passport-heading">
            <span class="tiq-passport-title">My Travel Passport</span>
            <span style="font-size:1.4rem;">🧭</span>
        </div>
        <div class="tiq-passport-subtitle">Journey • Explore • Remember</div>
        """,
        unsafe_allow_html=True,
    )
    stamp_grid(traveler_stamps)
    st.markdown(
        '<div class="tiq-passport-quote">"Collect stamps. Cherish memories. Keep exploring."</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col_side:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("📊", "Journey So Far")
    # No real dataset totals exist yet, so these are shown as simple
    # label/value stats (matching the Memories Collected style below)
    # instead of progress_stat's current/total bar, which would require
    # a placeholder denominator.
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.9rem;">
            <span style="color:var(--text-secondary); font-weight:600;">Destinations Explored</span>
            <span style="color:var(--text-primary); font-weight:800;">{cities_explored}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-bottom:0.9rem;">
            <span style="color:var(--text-secondary); font-weight:600;">Attractions Explored</span>
            <span style="color:var(--text-primary); font-weight:800;">{attractions_explored}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-top:0.2rem;">
            <span style="color:var(--text-secondary); font-weight:600;">Memories Collected</span>
            <span style="color:var(--text-primary); font-weight:800;">{total_visits}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-top:0.9rem;">
            <span style="color:var(--text-secondary); font-weight:600;">Average Rating</span>
            <span style="color:var(--text-primary); font-weight:800;">{average_rating:.2f}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-top:0.9rem;">
            <span style="color:var(--text-secondary); font-weight:600;">Favorite Travel Style</span>
            <span style="color:var(--text-primary); font-weight:800;">{most_common_visit_mode}</span>
        </div>
        <div style="display:flex; justify-content:space-between; font-size:0.85rem; margin-top:0.9rem;">
            <span style="color:var(--text-secondary); font-weight:600;">Travel Years</span>
            <span style="color:var(--text-primary); font-weight:800;">{years_visited}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # ---- Easter egg: click the compass a few times ----
    st.markdown('<div class="tiq-card" style="text-align:center;">', unsafe_allow_html=True)
    st.caption("Some emblems are more than decoration.")
    with st.container(key="compass_trigger"):
        clicked = st.button("🧭", key="compass_btn")

    if clicked:
        st.session_state["compass_clicks"] = st.session_state.get("compass_clicks", 0) + 1

    clicks = st.session_state.get("compass_clicks", 0)

    if clicks >= 5:
        st.markdown(
            """
            <div class="tiq-secret-reveal">
                <div class="headline">✨ SECRET DISCOVERED ✨</div>
                <div class="body">Some journeys are planned.<br/>The best ones become memories.</div>
                <div style="margin-top:0.4rem; font-size:1.3rem;">🧭</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif clicks > 0:
        st.caption(f"Keep going… ({clicks}/5)")
    st.markdown("</div>", unsafe_allow_html=True)

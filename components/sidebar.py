"""
Custom dark-glass sidebar: brand header, nav links, traveler footer.

Uses st.page_link so it plugs directly into st.navigation's routing
(no dead links, no manual state juggling) while still letting us
control the visual chrome around it.
"""

import pandas as pd
import streamlit as st
from components.navigation import NAV_ITEMS
from components.passport import sidebar_passport_preview

CONSOLIDATED_CSV_PATH = "data/processed/consolidated.csv"


def render_brand():
    st.markdown(
        """
        <div class="tiq-brand">
            <span class="tiq-brand-icon">✈️</span>
            <div>
                <p class="tiq-brand-name">TravelIQ</p>
                <p class="tiq-brand-sub">Explore • Predict • Recommend</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_nav():
    for item in NAV_ITEMS:
        label = f"{item.icon}  {item.label}"
        st.page_link(item.file, label=label, width="stretch")


def render_footer():
    traveler_id = st.session_state.get("traveler_id")

    if traveler_id is None:
        cities_explored = 0
        attractions_explored = 0
        total_visits = 0
    else:
        consolidated_df = pd.read_csv(CONSOLIDATED_CSV_PATH)
        traveler_df = consolidated_df[consolidated_df["UserId"] == traveler_id]

        cities_explored = traveler_df["AttractionCityId"].nunique()
        attractions_explored = traveler_df["AttractionId"].nunique()
        total_visits = len(traveler_df)

    sidebar_passport_preview(
        countries=cities_explored,
        attractions=attractions_explored,
        memories=total_visits,
    )

    st.markdown(
        """
        <div class="tiq-sidebar-footer">
            <div class="tiq-traveler-card">
                <div class="tiq-traveler-avatar">🧭</div>
                <div>
                    <p class="tiq-traveler-name">Traveler</p>
                    <p class="tiq-traveler-role">Explorer Mode</p>
                </div>
            </div>
            <p style="margin-top:0.6rem; font-size:0.74rem; color:var(--text-muted);">
                <span class="tiq-status-dot"></span>Ready to discover
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        render_brand()
        render_nav()
        render_footer()
"""
"🌍 Explore the World" — interactive map + dataset-driven attraction explorer.

Home-page-only feature. Flow:
    World map -> user clicks a country (or picks one from the dropdown)
    -> attractions for that country are read straight from our own
       processed dataset (data/processed/consolidated.csv) and rendered
       below the map.

No attraction, rating, or popularity figure here is invented -- every
number comes from an aggregation over real transaction rows. Countries
with no attraction data in our dataset show a clean empty state instead
of fabricated content.

The map's geography (country shapes/borders) is drawn from Plotly's
built-in world-atlas basemap, used purely as a geographic reference
layer -- no WorldAtlas articles, images, or text are fetched, scraped,
or reproduced anywhere in this module.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from components.cards import empty_state, section_title
from components.charts import world_choropleth

ROOT = Path(__file__).resolve().parents[1]
CONSOLIDATED_PATH = ROOT / "data" / "processed" / "consolidated.csv"
COUNTRY_PATH = ROOT / "data" / "cleaned" / "country.csv"

_STATE_KEY = "tiq_explore_country"


@st.cache_data(show_spinner=False)
def _load_attraction_data() -> pd.DataFrame:
    """Attraction-level slice of our processed dataset (one row per visit)."""
    cols = [
        "AttractionId",
        "Attraction",
        "AttractionTypeName",
        "AttractionCityName",
        "AttractionCountry",
        "Rating",
    ]
    df = pd.read_csv(CONSOLIDATED_PATH, usecols=cols)
    return df


@st.cache_data(show_spinner=False)
def _load_known_countries() -> list[str]:
    """The full country universe already present in our own dataset
    (data/cleaned/country.csv), used to populate the destination picker
    and the map's base layer -- not an external source."""
    df = pd.read_csv(COUNTRY_PATH)
    df = df[(df["Country"] != "-") & (~df.get("flag_duplicate_name", False))]
    return sorted(df["Country"].dropna().unique().tolist())


def _country_summary(attractions: pd.DataFrame, known_countries: list[str]) -> pd.DataFrame:
    """One row per known country with attraction count / avg rating,
    zero-filled for countries our dataset has no attraction data for."""
    agg = (
        attractions.groupby("AttractionCountry")
        .agg(
            AttractionCount=("AttractionId", "nunique"),
            AvgRating=("Rating", "mean"),
            TotalVisits=("Rating", "count"),
        )
        .reset_index()
        .rename(columns={"AttractionCountry": "Country"})
    )
    base = pd.DataFrame({"Country": known_countries})
    merged = base.merge(agg, on="Country", how="left")
    merged["AttractionCount"] = merged["AttractionCount"].fillna(0).astype(int)
    merged["TotalVisits"] = merged["TotalVisits"].fillna(0).astype(int)

    def _hover(row) -> str:
        if row["AttractionCount"] > 0:
            return (
                f"<b>{row['Country']}</b><br>"
                f"{row['AttractionCount']} attraction(s) in our dataset<br>"
                f"Avg rating: {row['AvgRating']:.2f} ★  ·  {row['TotalVisits']} visits logged"
            )
        return f"<b>{row['Country']}</b><br>No attractions available yet"

    merged["HoverText"] = merged.apply(_hover, axis=1)
    return merged


def _attractions_for_country(attractions: pd.DataFrame, country: str) -> pd.DataFrame:
    subset = attractions[attractions["AttractionCountry"] == country]
    if subset.empty:
        return subset
    summary = (
        subset.groupby(["AttractionId", "Attraction", "AttractionTypeName", "AttractionCityName"])
        .agg(AvgRating=("Rating", "mean"), TotalVisits=("Rating", "count"))
        .reset_index()
        .sort_values("TotalVisits", ascending=False)
    )
    return summary


def _attraction_card_html(row) -> str:
    stars = "★" * round(row.AvgRating) + "☆" * (5 - round(row.AvgRating))
    return f"""
    <div class="tiq-attraction-card">
        <div class="tiq-attraction-type">{row.AttractionTypeName}</div>
        <div class="tiq-attraction-name">{row.Attraction}</div>
        <div class="tiq-attraction-city">📍 {row.AttractionCityName}</div>
        <div class="tiq-attraction-meta">
            <span class="tiq-attraction-rating">{stars} <b>{row.AvgRating:.1f}</b></span>
            <span class="tiq-attraction-visits">🧳 {int(row.TotalVisits)} visits logged</span>
        </div>
    </div>
    """


def render_explore_the_world():
    """Renders the full 'Explore the World' section: map, destination
    picker, and the dataset-driven attraction list for the selection."""
    attractions = _load_attraction_data()
    known_countries = _load_known_countries()
    countries_with_data = sorted(attractions["AttractionCountry"].dropna().unique().tolist())

    if _STATE_KEY not in st.session_state:
        st.session_state[_STATE_KEY] = countries_with_data[0] if countries_with_data else known_countries[0]

    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("🌍", "EXPLORE THE WORLD")
    st.caption("Click a country on the map — or pick one below — to see its attractions from our dataset.")

    country_df = _country_summary(attractions, known_countries)
    fig = world_choropleth(country_df)

    event = st.plotly_chart(
        fig,
        width="stretch",
        key="tiq_world_map_chart",
        on_select="rerun",
        selection_mode="points",
        config={"displayModeBar": False, "scrollZoom": False},
    )

    if event and event.get("selection") and event["selection"].get("points"):
        clicked = event["selection"]["points"][0].get("location")
        if clicked:
            st.session_state[_STATE_KEY] = clicked

    col_pick, col_badge = st.columns([2.2, 1])
    with col_pick:
        options = known_countries
        current = st.session_state[_STATE_KEY]
        idx = options.index(current) if current in options else 0
        chosen = st.selectbox("Or choose a destination", options, index=idx, key="tiq_explore_dropdown")
        if chosen != st.session_state[_STATE_KEY]:
            st.session_state[_STATE_KEY] = chosen
    with col_badge:
        st.write("")
        has_data = st.session_state[_STATE_KEY] in countries_with_data
        badge_text = "📍 Data available" if has_data else "🚧 No data yet"
        st.markdown(f'<span class="tiq-badge">{badge_text}</span>', unsafe_allow_html=True)

    selected_country = st.session_state[_STATE_KEY]
    st.markdown(
        f'<div class="tiq-section-title" style="margin-top:1.1rem;">'
        f"<span>🏝️</span><span>Attractions in {selected_country}</span></div>",
        unsafe_allow_html=True,
    )

    country_attractions = _attractions_for_country(attractions, selected_country)
    if country_attractions.empty:
        empty_state(
            icon="🧭",
            title="No attractions available yet",
            subtitle=f"Our dataset doesn't have attraction records for {selected_country} yet. "
            "Try Indonesia, or explore other destinations on the map.",
        )
    else:
        st.markdown('<div class="tiq-attraction-grid">', unsafe_allow_html=True)
        cards_html = "".join(_attraction_card_html(row) for row in country_attractions.itertuples())
        st.markdown(cards_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

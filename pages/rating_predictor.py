"""Rating Predictor -- connected to the real trained regression model.

Inference is delegated entirely to components/rating_predictor.py
(Checkpoint 10, Part 1) -- this page only collects real, meaningful
user inputs (a real attraction, home continent/region, visit mode,
and a planned month/year) and renders the result. No second
prediction pipeline, no retraining, no hidden statistics asked of
the user.
"""

import streamlit as st

from components.cards import page_header, section_title
from components.rating_predictor import known_options, predict_rating

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

page_header(
    eyebrow="⭐ Predict",
    title="Rating Predictor",
    subtitle="Estimate the rating a traveler is likely to give an attraction.",
)

# ---------------------------------------------------------------------------
# Load real, dataset-sourced options -- gracefully degrade if unavailable
# ---------------------------------------------------------------------------
try:
    options = known_options()
except FileNotFoundError:
    options = None

if not options or not options.get("attractions"):
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("⚠️", "Predictor Unavailable")
    st.markdown(
        '<div class="tiq-empty-state"><div class="tiq-empty-title">Model data not found</div>'
        '<div class="tiq-empty-sub">Couldn\'t load the trained regression model or its supporting '
        "data. Make sure models/regression/best_model.joblib and data/processed/consolidated.csv "
        "are present.</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

attractions = sorted(options["attractions"], key=lambda a: a["Attraction"])
attraction_labels = [f"{a['Attraction']} — {a['AttractionTypeName']} ({a['AttractionCityName']})" for a in attractions]

continents = options["continents"]
continent_to_regions = {
    c: [r for r in regions if r and r != "-"] or regions
    for c, regions in options["continent_to_regions"].items()
}
visit_modes = options["visit_modes"]

col_form, col_result = st.columns([1, 1.1])

with col_form:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("📝", "Trip Details")

    attraction_idx = st.selectbox(
        "Attraction",
        options=range(len(attractions)),
        format_func=lambda i: attraction_labels[i],
        key="rp_attraction_idx",
    )
    selected_attraction = attractions[attraction_idx]

    visit_mode_name = st.selectbox("Visit Mode", visit_modes, key="rp_visit_mode")

    col_month, col_year = st.columns(2)
    with col_month:
        visit_month_name = st.selectbox("Planned Month", MONTH_NAMES, index=0, key="rp_month")
        visit_month = MONTH_NAMES.index(visit_month_name) + 1
    with col_year:
        visit_year = st.number_input("Planned Year", min_value=2013, max_value=2027, value=2026, step=1, key="rp_year")
    if int(visit_year) > 2022:
        st.caption("⚠️ Predictions beyond 2022 are extrapolations beyond the model's historical training period.")

    user_continent = st.selectbox("Traveler's Home Continent", continents, key="rp_continent")
    regions_for_continent = continent_to_regions.get(user_continent, [])
    user_region = st.selectbox("Traveler's Home Region", regions_for_continent, key="rp_region")

    st.markdown('<div class="tiq-btn-secondary">', unsafe_allow_html=True)
    predict_clicked = st.button("Predict Rating", width="stretch", key="rp_predict_btn")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Prediction uses the trained model plus the attraction's real historical track record.")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    try:
        result = predict_rating(
            attraction_id=selected_attraction["AttractionId"],
            visit_year=int(visit_year),
            visit_month=int(visit_month),
            visit_mode_name=visit_mode_name,
            user_id=None,
            user_continent=user_continent,
            user_region=user_region,
        )
        st.session_state["rp_result"] = result
        st.session_state["rp_error"] = None
    except ValueError as e:
        st.session_state["rp_result"] = None
        st.session_state["rp_error"] = str(e)

with col_result:
    st.markdown('<div class="tiq-card" style="height:100%;">', unsafe_allow_html=True)
    section_title("🎯", "Expected Experience")

    error = st.session_state.get("rp_error")
    result = st.session_state.get("rp_result")

    if error:
        st.markdown(
            f"""
            <div class="tiq-empty-state" style="padding:2.4rem 1.5rem;">
                <div class="tiq-empty-title">Couldn't generate a prediction</div>
                <div class="tiq-empty-sub">{error}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif result:
        rating_value = result["display_rating"]
        full_stars = int(round(rating_value))
        stars_html = "★" * full_stars + "☆" * (5 - full_stars)
        st.markdown(
            f"""
            <div class="tiq-empty-state" style="padding:2.4rem 1.5rem;">
                <div class="tiq-eyebrow" style="margin-bottom:0.6rem;">Expected Experience</div>
                <div style="font-size:2.6rem; font-weight:800; color:var(--text-primary); line-height:1;">{rating_value:.2f} <span style="font-size:1.2rem; color:var(--text-muted);">/ 5</span></div>
                <div style="color:var(--accent-orange); font-size:1.1rem; margin-top:0.5rem;">{stars_html}</div>
                <div class="tiq-empty-title" style="margin-top:1rem;">{selected_attraction['Attraction']}</div>
                <div class="tiq-empty-sub">Predicted for a {visit_mode_name.lower()} visit in {visit_month_name} {int(visit_year)}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="tiq-empty-state" style="padding:2.4rem 1.5rem;">
                <div class="tiq-eyebrow" style="margin-bottom:0.6rem;">Expected Experience</div>
                <div style="font-size:2.6rem; font-weight:800; color:var(--text-muted); line-height:1;">— / 5</div>
                <div class="tiq-empty-title" style="margin-top:1rem;">Set your trip details</div>
                <div class="tiq-empty-sub">Choose an attraction and travel context, then hit Predict Rating.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:1.2rem;">
            <span class="tiq-badge">Model: Gradient Boosting Regressor</span>
            <span class="tiq-badge">Test R²: 0.0724</span>
            <span class="tiq-badge">MAE: 0.7752</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

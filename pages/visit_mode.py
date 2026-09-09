"""Visit Mode Predictor -- connected to the real trained classification model.

Inference is delegated entirely to components/visit_mode_predictor.py
(Checkpoint 11, Part 1) -- this page only collects real, meaningful
user inputs (a real attraction, home continent/region, and a planned
month/year) and renders the result. No second prediction pipeline, no
retraining, no hidden statistics asked of the user.

Note: the original placeholder inputs on this page (Age, a generic
Continent list, a generic Attraction Type list, Trip Duration) don't
match the trained model's real feature schema (identified in
Checkpoint 11 Part 1) -- Age and Trip Duration aren't used by the
model at all, and the generic Continent/Attraction-Type lists don't
match the real dataset categories. Consistent with how Rating
Predictor was connected in Checkpoint 10, those inputs are swapped
for real, dataset-sourced equivalents while the card layout, titles,
and styling are left exactly as they were.
"""

import streamlit as st

from components.cards import page_header, section_title
from components.charts import ACCENTS
from components.visit_mode_predictor import known_options, predict_visit_mode

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

page_header(
    eyebrow="🧳 Predict",
    title="Visit Mode Predictor",
    subtitle="Classify how a traveler is most likely to visit — solo, family, couple, friends, or business.",
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
        '<div class="tiq-empty-sub">Couldn\'t load the trained classification model or its supporting '
        "data. Make sure models/classification/best_model.joblib and data/processed/consolidated.csv "
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
class_labels = options["class_labels"]

col_form, col_result = st.columns([1, 1.1])

with col_form:
    st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
    section_title("📝", "Traveler Profile")

    attraction_idx = st.selectbox(
        "Attraction",
        options=range(len(attractions)),
        format_func=lambda i: attraction_labels[i],
        key="vm_attraction_idx",
    )
    selected_attraction = attractions[attraction_idx]

    user_continent = st.selectbox("Traveler's Home Continent", continents, key="vm_continent")
    regions_for_continent = continent_to_regions.get(user_continent, [])
    user_region = st.selectbox("Traveler's Home Region", regions_for_continent, key="vm_region")

    col_month, col_year = st.columns(2)
    with col_month:
        visit_month_name = st.selectbox("Planned Month", MONTH_NAMES, index=0, key="vm_month")
        visit_month = MONTH_NAMES.index(visit_month_name) + 1
    with col_year:
        visit_year = st.number_input("Planned Year", min_value=2013, max_value=2027, value=2026, step=1, key="vm_year")
    if int(visit_year) > 2022:
        st.caption("⚠️ Predictions beyond 2022 are extrapolations beyond the model's historical training period.")

    st.markdown('<div class="tiq-btn-secondary">', unsafe_allow_html=True)
    predict_clicked = st.button("Predict Visit Mode", width="stretch", key="vm_predict_btn")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Prediction uses the trained classifier plus the attraction's real historical track record.")
    st.markdown("</div>", unsafe_allow_html=True)

if predict_clicked:
    try:
        result = predict_visit_mode(
            attraction_id=selected_attraction["AttractionId"],
            visit_year=int(visit_year),
            visit_month=int(visit_month),
            user_id=None,
            user_continent=user_continent,
            user_region=user_region,
        )
        st.session_state["vm_result"] = result
        st.session_state["vm_error"] = None
    except ValueError as e:
        st.session_state["vm_result"] = None
        st.session_state["vm_error"] = str(e)

with col_result:
    st.markdown('<div class="tiq-card" style="height:100%;">', unsafe_allow_html=True)
    section_title("🎯", "Expected Visit Mode")

    error = st.session_state.get("vm_error")
    result = st.session_state.get("vm_result")

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
        predicted_mode = result["predicted_mode"]
        probs = result.get("class_probabilities") or {}
        st.markdown(
            f"""
            <div class="tiq-empty-state" style="padding:2.4rem 1.5rem;">
                <div class="tiq-eyebrow" style="margin-bottom:0.6rem;">Expected Visit Mode</div>
                <div style="font-size:2.6rem; font-weight:800; color:var(--text-primary); line-height:1;">{predicted_mode}</div>
                <div class="tiq-empty-title" style="margin-top:1rem;">{selected_attraction['Attraction']}</div>
                <div class="tiq-empty-sub">Predicted for {visit_month_name} {int(visit_year)}.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if probs:
            st.markdown('<div style="margin-top:0.4rem;">', unsafe_allow_html=True)
            for i, label in enumerate(class_labels):
                pct = round(probs.get(label, 0.0) * 100, 1)
                color = ACCENTS[i % len(ACCENTS)]
                st.markdown(
                    f"""
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:var(--text-secondary); margin-bottom:0.15rem;">
                        <span>{label}</span><span>{pct:.1f}%</span>
                    </div>
                    <div class="tiq-progress-track">
                        <div class="tiq-progress-fill" style="width:{pct}%; background:{color};"></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            """
            <div class="tiq-empty-state" style="padding:2.4rem 1.5rem;">
                <div class="tiq-eyebrow" style="margin-bottom:0.6rem;">Expected Visit Mode</div>
                <div style="font-size:2.6rem; font-weight:800; color:var(--text-muted); line-height:1;">—</div>
                <div class="tiq-empty-title" style="margin-top:1rem;">Set your traveler profile</div>
                <div class="tiq-empty-sub">Choose an attraction and travel context, then hit Predict Visit Mode.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="display:flex; gap:0.6rem; flex-wrap:wrap; margin-top:1.2rem;">
            <span class="tiq-badge">Model: Random Forest Classifier</span>
            <span class="tiq-badge">Test Accuracy: 42.5%</span>
            <span class="tiq-badge">Macro F1: 0.2811</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

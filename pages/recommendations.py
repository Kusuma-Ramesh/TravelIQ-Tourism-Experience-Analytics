"""Attraction Recommendations -- connected to the real, already-trained
content/profile-based recommender.

Inference is delegated entirely to components/recommendations_data.py,
which is a thin wrapper around the EXISTING
scripts/recommender.ContentProfileRecommender -- no new scoring logic,
no retraining, no second recommendation pipeline.

Note on inputs: the original placeholder inputs on this page
("Preferred Region", "Interests") don't correspond to anything the
real recommender accepts -- it has no notion of a home region and
computes personalization purely from a traveler's OWN past rating
history (see scripts/recommender.py: type_affinity is derived from
interactions_df filtered by UserId). Consistent with how Rating
Predictor and Visit Mode Predictor were connected, those two fields
are replaced with the traveler-identity input the model actually
supports (returning traveler by real Traveler ID, or a brand-new
traveler -> the recommender's own built-in cold-start fallback), while
"Number of Recommendations" is kept completely unchanged since it maps
directly to the existing `k` parameter. Card layout, titles, and
styling are otherwise untouched.
"""

import streamlit as st

from components.cards import page_header, section_title
from components.recommendations_data import get_recommendations

page_header(
    eyebrow="🗺️ Recommend",
    title="Attraction Recommendations",
    subtitle="Personalized attraction suggestions based on traveler preferences and history.",
)

st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
section_title("🎛️", "Preferences")
c1, c2, c3 = st.columns(3)
with c1:
    traveler_type = st.selectbox("Traveler Type", ["New Traveler", "Returning Traveler"], key="rec_traveler_type")
with c2:
    if traveler_type == "Returning Traveler":
        user_id = st.number_input(
            "Your Traveler ID", min_value=1, value=26, step=1, key="rec_user_id",
            help="Personalizes picks using that traveler's real rating history, if we have one on file.",
        )
    else:
        user_id = None
        st.caption("We'll show today's most popular, highly-rated attractions.")
with c3:
    rec_count = st.slider("Number of Recommendations", 3, 12, 6, key="rec_count")
get_recs_clicked = st.button("Get Recommendations", width="stretch", key="rec_btn")
st.caption("Recommendations use the trained content/profile-based recommender and each traveler's real visit history.")
st.markdown("</div>", unsafe_allow_html=True)

if get_recs_clicked:
    try:
        recs = get_recommendations(user_id=user_id, k=int(rec_count), exclude_visited=True)
        # Convert straight to plain Python-typed records (str/float/bool) so
        # the render step below never touches a DataFrame/numpy value
        # directly -- the real attraction names, reasons, and scores are
        # preserved exactly, just as native Python types instead of
        # pandas/numpy objects.
        records = [
            {
                "Attraction": str(row["Attraction"]),
                "AttractionTypeName": str(row["AttractionTypeName"]),
                "Reason": str(row["Reason"]),
                "score": float(row["score"]),
                "is_cold_start": bool(row["is_cold_start"]),
            }
            for _, row in recs.iterrows()
        ]
        st.session_state["rec_results"] = records
        st.session_state["rec_error"] = None
    except Exception as e:
        st.session_state["rec_results"] = None
        st.session_state["rec_error"] = (
            f"Something went wrong while generating recommendations ({type(e).__name__}). "
            "Please try again or pick a different traveler."
        )

st.write("")

st.markdown('<div class="tiq-card">', unsafe_allow_html=True)
section_title("✨", "Suggested For You")

error = st.session_state.get("rec_error")
results = st.session_state.get("rec_results")

if error:
    st.markdown(
        f"""
        <div class="tiq-empty-state">
            <div class="tiq-empty-icon">⚠️</div>
            <div class="tiq-empty-title">Couldn't generate recommendations</div>
            <div class="tiq-empty-sub">{error}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
elif results is not None and len(results) > 0:
    is_cold_start = results[0]["is_cold_start"]
    banner_text = (
        "Popular picks — no personal history found for this traveler yet"
        if is_cold_start
        else "Personalized using this traveler's real rating history"
    )
    st.markdown(
        f'<div class="tiq-eyebrow" style="margin-bottom:0.8rem;">{banner_text}</div>',
        unsafe_allow_html=True,
    )
    for row in results:
        st.markdown(
            f"""
            <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:1rem;
                        padding:0.85rem 0; border-bottom:1px solid rgba(255,255,255,0.06);">
                <div style="flex:1;">
                    <div style="font-weight:700; color:var(--text-primary); font-size:0.98rem;">{row['Attraction']}</div>
                    <div style="color:var(--text-muted); font-size:0.8rem; margin:0.15rem 0 0.35rem 0;">{row['AttractionTypeName']}</div>
                    <div style="color:var(--text-secondary); font-size:0.83rem; line-height:1.5;">{row['Reason']}</div>
                </div>
                <div style="text-align:right; white-space:nowrap;">
                    <span class="tiq-badge">Score {row['score']:.2f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
elif results is not None:
    st.markdown(
        """
        <div class="tiq-empty-state">
            <div class="tiq-empty-icon">🗺️</div>
            <div class="tiq-empty-title">No recommendations available</div>
            <div class="tiq-empty-sub">This traveler has already visited every attraction in the dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    from components.cards import empty_state
    empty_state(
        icon="🗺️",
        title="Recommendations will appear here",
        subtitle="Set your traveler type above and click Get Recommendations.",
    )
st.markdown("</div>", unsafe_allow_html=True)

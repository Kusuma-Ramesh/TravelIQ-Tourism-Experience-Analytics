"""
Attraction Recommendations -- inference layer (Checkpoint 12).

This module loads the EXISTING trained recommendation artifacts
(models/recommendation/recommender_artifacts.joblib, which bundles the
attraction content profile in models/recommendation/attraction_profile.csv
plus the scoring weights in models/recommendation/config.json) and calls
the EXISTING `ContentProfileRecommender` class from
scripts/recommender.py exactly as implemented -- no new scoring
formula, no retraining, no second recommendation pipeline.

`scripts/recommender.py` is not a package (no scripts/__init__.py), so
it's imported the same way scripts/train_recommendation_model.py
itself already does: by adding the scripts/ directory to sys.path and
importing the module directly. scripts/recommender.py itself is never
modified.

WHAT THIS MODULE ADDS
----------------------
Just enough glue for a Streamlit page to call the recommender:
  - cached loaders for the saved recommender artifact and the full
    `data/processed/recommendation_interactions.csv` history (the same
    file the production artifact was trained on, per
    models/recommendation/config.json)
  - `get_recommendations(user_id, k, exclude_visited)` -- a thin
    pass-through to `ContentProfileRecommender.recommend(...)`

COLD START
----------
`ContentProfileRecommender.recommend()` already handles cold start on
its own: if `user_id` has no rows in `interactions_df` (a brand-new
traveler ID, or no ID at all), the user's history is naturally empty
and the recommender falls back to its documented non-personalized
score. This module relies on that existing behavior rather than
reimplementing it -- callers can pass a real, existing UserId for a
personalized list, or any UserId not present in the data (this module
uses -1, which never occurs in the real dataset) to get the cold-start
"popular picks" list. The returned DataFrame's `is_cold_start` column
(already produced by `recommend()`) tells the caller which happened.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = str(ROOT / "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from recommender import ContentProfileRecommender  # noqa: E402  (existing, untouched module)

ARTIFACT_PATH = ROOT / "models" / "recommendation" / "recommender_artifacts.joblib"
INTERACTIONS_PATH = ROOT / "data" / "processed" / "recommendation_interactions.csv"

# Sentinel for a brand-new traveler with no UserId in the dataset --
# guaranteed not to collide with a real UserId (all real IDs are positive).
COLD_START_SENTINEL_USER_ID = -1


@st.cache_resource(show_spinner=False)
def load_recommender() -> ContentProfileRecommender:
    """The saved, already-trained ContentProfileRecommender -- loaded via
    its own existing `.load()` classmethod, nothing reimplemented here."""
    return ContentProfileRecommender.load(str(ARTIFACT_PATH))


@st.cache_data(show_spinner=False)
def load_interactions() -> pd.DataFrame:
    """The same full interaction history the production artifact was
    trained on (per models/recommendation/config.json) -- used only to
    look up a given user's own history for personalization / exclusion."""
    return pd.read_csv(INTERACTIONS_PATH)


@st.cache_data(show_spinner=False)
def known_user_ids() -> list[int]:
    """Real UserIds present in the interaction history, for a caller that
    wants to validate or offer a 'try a real traveler ID' example."""
    return sorted(load_interactions()["UserId"].unique().tolist())


def get_recommendations(user_id: int | None, k: int = 6, exclude_visited: bool = True) -> pd.DataFrame:
    """Thin pass-through to the existing recommender. `user_id=None` (or
    any id absent from the interaction history) naturally triggers the
    recommender's own built-in cold-start fallback -- see module
    docstring."""
    model = load_recommender()
    interactions = load_interactions()
    effective_user_id = COLD_START_SENTINEL_USER_ID if user_id is None else user_id
    return model.recommend(
        user_id=effective_user_id,
        interactions_df=interactions,
        k=k,
        exclude_visited=exclude_visited,
    )

"""
Visit Mode Predictor -- inference layer (Checkpoint 11, Part 1).

This module loads the EXISTING trained classification artifact
(models/classification/best_model.joblib, a Random Forest Classifier
inside a sklearn Pipeline whose first step already embeds
preprocessing -- StandardScaler on numeric features, OneHotEncoder
(handle_unknown='ignore') on categorical features) and builds ONLY
the feature row needed to call `.predict()` / `.predict_proba()` on
it. No model is retrained here. The exact feature schema/order comes
from models/classification/preprocessing_config.json, written by
scripts/train_classification_model.py.

WHY data/processed/classification_features.csv IS NEVER READ HERE
-------------------------------------------------------------------
scripts/train_classification_model.py's own module docstring documents
that classification_features.csv contains Rating as an input feature
and full-history (non-temporal) user_*/attraction_* aggregates -- both
unsafe for a pre-visit prediction. The trained model was instead built
from features recomputed straight from data/processed/consolidated.csv
using the SAME leakage-safe temporal helpers already used for
regression (add_temporal_user_history / add_temporal_attraction_history
in scripts/feature_engineering.py). This module mirrors that choice at
inference time and additionally reuses the already-verified historical-
as-of computation from components/rating_predictor.py (imported
read-only, not modified) rather than re-implementing the same math a
second time.

LEAKAGE SAFETY
--------------
  - Rating is never read, computed, or included as a feature.
  - user_*/attraction_* historical aggregates are computed on demand
    for the requested (VisitYear, VisitMonth), using only transaction
    history with period STRICTLY BEFORE that period -- identical
    "strictly-earlier-period" methodology as Rating Predictor and as
    scripts/feature_engineering.py.
  - classification_features.csv is never read.
  - For a brand-new/prospective traveler (no UserId in the dataset),
    callers pass `user_continent` / `user_region` directly and get the
    same fixed cold-start fallback (no history exists) as the trained
    schema defines.

This module does not touch pages/visit_mode.py -- UI integration is
Part 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from components.rating_predictor import (
    _distinct_count_as_of,
    _history_stats_as_of,
    load_attraction_lookup,
    load_history,
    load_user_geo_lookup,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "classification" / "best_model.joblib"
CONFIG_PATH = ROOT / "models" / "classification" / "preprocessing_config.json"


# ---------------------------------------------------------------------------
# Cached artifact loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """The saved sklearn Pipeline. Preprocessing is already its first step --
    nothing else needs to be fit or loaded separately at inference time."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    """feature_columns_in_order / numeric_features / categorical_features /
    class_labels, exactly as written by scripts/train_classification_model.py."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def known_options() -> dict:
    """Real, dataset-sourced option lists for building a prediction form --
    the same attraction/continent/region universe as Rating Predictor, plus
    this model's class labels."""
    geo = load_user_geo_lookup()
    continents = sorted(geo["UserContinent"].dropna().unique().tolist())
    continent_to_regions = (
        geo.groupby("UserContinent")["UserRegion"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )
    attractions = load_attraction_lookup()
    config = load_config()
    return {
        "continents": continents,
        "continent_to_regions": continent_to_regions,
        "attractions": attractions[
            ["AttractionId", "Attraction", "AttractionTypeName", "AttractionCityName"]
        ].to_dict(orient="records"),
        "class_labels": config["class_labels"],
    }


# ---------------------------------------------------------------------------
# Feature row construction + prediction
# ---------------------------------------------------------------------------

def build_feature_row(
    attraction_id: int,
    visit_year: int,
    visit_month: int,
    user_id: int | None = None,
    user_continent: str | None = None,
    user_region: str | None = None,
    is_ambiguous_repeat: bool = False,
) -> pd.DataFrame:
    """Build the single-row DataFrame the saved classification pipeline
    expects -- exactly `feature_columns_in_order` from
    models/classification/preprocessing_config.json. No Rating column,
    no VisitModeName column (that's the target being predicted)."""
    history = load_history()
    attraction_lookup = load_attraction_lookup()
    config = load_config()

    as_of_period = int(visit_year) * 12 + int(visit_month)

    attr_row = attraction_lookup[attraction_lookup["AttractionId"] == attraction_id]
    if attr_row.empty:
        raise ValueError(
            f"Unknown AttractionId: {attraction_id!r}. Must be one of the "
            f"{len(attraction_lookup)} attractions already in the dataset."
        )
    attraction_type_name = attr_row.iloc[0]["AttractionTypeName"]
    attraction_city_name = attr_row.iloc[0]["AttractionCityName"]

    attr_stats = _history_stats_as_of(history, "AttractionId", attraction_id, as_of_period)
    attraction_distinct_users = _distinct_count_as_of(history, "AttractionId", attraction_id, "UserId", as_of_period)

    if user_id is not None:
        geo = load_user_geo_lookup()
        user_row = geo[geo["UserId"] == user_id]
        if user_row.empty:
            raise ValueError(
                f"Unknown UserId: {user_id!r}. Pass user_continent/user_region "
                "instead for a new/prospective traveler."
            )
        resolved_continent = user_row.iloc[0]["UserContinent"]
        resolved_region = user_row.iloc[0]["UserRegion"]
        user_stats = _history_stats_as_of(history, "UserId", user_id, as_of_period)
        user_distinct_attractions = _distinct_count_as_of(history, "UserId", user_id, "AttractionId", as_of_period)
    else:
        if not user_continent or not user_region:
            raise ValueError(
                "For a new/prospective traveler (no user_id), both "
                "user_continent and user_region are required."
            )
        resolved_continent = user_continent
        resolved_region = user_region
        # Cold start by definition -- a user who isn't in the data yet has
        # no strictly-earlier history to aggregate.
        user_stats = {"total_visits": 0, "avg_rating": 3.0, "rating_std": 0.0, "has_history": 0}
        user_distinct_attractions = 0

    row = {
        "VisitYear": int(visit_year),
        "VisitMonth": int(visit_month),
        "user_total_visits": user_stats["total_visits"],
        "user_avg_rating": user_stats["avg_rating"],
        "user_rating_std": user_stats["rating_std"],
        "user_distinct_attractions": user_distinct_attractions,
        "has_user_history": user_stats["has_history"],
        "attraction_total_visits": attr_stats["total_visits"],
        "attraction_avg_rating": attr_stats["avg_rating"],
        "attraction_rating_std": attr_stats["rating_std"],
        "attraction_distinct_users": attraction_distinct_users,
        "has_attraction_history": attr_stats["has_history"],
        "is_ambiguous_repeat": int(bool(is_ambiguous_repeat)),
        "UserContinent": resolved_continent,
        "UserRegion": resolved_region,
        "AttractionTypeName": attraction_type_name,
        "AttractionCityName": attraction_city_name,
    }

    feature_order = config["feature_columns_in_order"]
    return pd.DataFrame([row])[feature_order]


def predict_visit_mode(
    attraction_id: int,
    visit_year: int,
    visit_month: int,
    user_id: int | None = None,
    user_continent: str | None = None,
    user_region: str | None = None,
    is_ambiguous_repeat: bool = False,
) -> dict:
    """Run the existing trained classifier on one prediction context.
    Returns the predicted class label (one of the 5 real VisitModeName
    values) plus per-class probabilities for a richer UI if wanted."""
    model = load_model()
    X = build_feature_row(
        attraction_id=attraction_id,
        visit_year=visit_year,
        visit_month=visit_month,
        user_id=user_id,
        user_continent=user_continent,
        user_region=user_region,
        is_ambiguous_repeat=is_ambiguous_repeat,
    )
    predicted_label = str(model.predict(X)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        probabilities = {str(c): float(p) for c, p in zip(model.classes_, proba)}
    return {
        "predicted_mode": predicted_label,
        "class_probabilities": probabilities,
        "features_used": X.iloc[0].to_dict(),
    }

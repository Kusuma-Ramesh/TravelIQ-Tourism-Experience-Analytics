"""
Rating Predictor -- inference layer (Checkpoint 10, Part 1).

This module loads the EXISTING trained regression artifact
(models/regression/best_model.joblib, a Gradient Boosting Regressor
inside a sklearn Pipeline whose first step already embeds preprocessing
-- StandardScaler on numeric features, OneHotEncoder(handle_unknown=
'ignore') on categorical features) and builds ONLY the feature row
needed to call `.predict()` on it. No model is retrained here, and no
new preprocessing is introduced -- the exact feature schema and
ordering come from models/regression/preprocessing_config.json, which
was written by scripts/train_regression_model.py.

LEAKAGE SAFETY
--------------
The trained model expects two families of "historical-as-of" features
(user_* and attraction_*) that scripts/feature_engineering.py computes
per (entity, calendar-month period) using only periods STRICTLY BEFORE
the row's own period (see add_temporal_user_history /
add_temporal_attraction_history there). Those precomputed values only
exist for periods already present in the historical dataset -- but a
prediction request is for a NEW, not-yet-happened visit, so this module
recomputes the identical "strictly-earlier-period" aggregate on demand,
from the project's own consolidated transaction history
(data/processed/consolidated.csv), for whatever (user/attraction,
VisitYear, VisitMonth) the caller asks about:

  - user_total_visits / user_avg_rating / user_rating_std /
    user_distinct_attractions / has_user_history
  - attraction_total_visits / attraction_avg_rating /
    attraction_rating_std / attraction_distinct_users /
    has_attraction_history

Both use the SAME math as feature_engineering.py (mean / sample std of
Rating over periods < the requested period; cold start -> avg 3.0,
std 0.0, has_history=0). Rating itself, any future period, and
classification_features.csv are never read or used as inputs.

For a brand-new/prospective traveler (no UserId in the dataset yet),
callers pass `user_continent` / `user_region` directly instead of a
`user_id`, and the user-side historical features are the fixed
cold-start fallback (no history can exist for a user who isn't in the
data at all).

This module does not touch pages/rating_predictor.py -- UI integration
is Part 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "regression" / "best_model.joblib"
CONFIG_PATH = ROOT / "models" / "regression" / "preprocessing_config.json"
CONSOLIDATED_PATH = ROOT / "data" / "processed" / "consolidated.csv"

# Matches scripts/feature_engineering.py's cold-start fallback exactly.
_RATING_FALLBACK = 3.0

_HISTORY_COLS = ["UserId", "AttractionId", "VisitYear", "VisitMonth", "Rating"]


# ---------------------------------------------------------------------------
# Cached artifact / data loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_model():
    """The saved sklearn Pipeline. Preprocessing is already its first step --
    nothing else needs to be fit or loaded separately at inference time."""
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    """feature_columns_in_order / numeric_features / categorical_features,
    exactly as written by scripts/train_regression_model.py."""
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_history() -> pd.DataFrame:
    """Minimal real transaction history used to derive historical-as-of
    features on demand. Rating is included ONLY to compute past-period
    aggregates for periods strictly before a requested prediction period --
    never as a feature for the period being predicted."""
    df = pd.read_csv(CONSOLIDATED_PATH, usecols=_HISTORY_COLS)
    df["_period"] = df["VisitYear"] * 12 + df["VisitMonth"]
    return df


@st.cache_data(show_spinner=False)
def load_attraction_lookup() -> pd.DataFrame:
    """One row per AttractionId -> Attraction, AttractionTypeName,
    AttractionCityName (each is constant per attraction in this dataset)."""
    df = pd.read_csv(
        CONSOLIDATED_PATH,
        usecols=["AttractionId", "Attraction", "AttractionTypeName", "AttractionCityName"],
    )
    return df.drop_duplicates(subset="AttractionId").sort_values("AttractionId").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_user_geo_lookup() -> pd.DataFrame:
    """One row per UserId -> UserContinent, UserRegion (each constant per user)."""
    df = pd.read_csv(CONSOLIDATED_PATH, usecols=["UserId", "UserContinent", "UserRegion"])
    return df.drop_duplicates(subset="UserId").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_visit_modes() -> list[str]:
    df = pd.read_csv(CONSOLIDATED_PATH, usecols=["VisitModeName"])
    return sorted(df["VisitModeName"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def known_options() -> dict:
    """Real, dataset-sourced option lists for building a prediction form --
    nothing here is an invented category."""
    geo = load_user_geo_lookup()
    continents = sorted(geo["UserContinent"].dropna().unique().tolist())
    continent_to_regions = (
        geo.groupby("UserContinent")["UserRegion"]
        .apply(lambda s: sorted(s.dropna().unique().tolist()))
        .to_dict()
    )
    attractions = load_attraction_lookup()
    return {
        "visit_modes": load_visit_modes(),
        "continents": continents,
        "continent_to_regions": continent_to_regions,
        "attractions": attractions[["AttractionId", "Attraction", "AttractionTypeName", "AttractionCityName"]]
        .to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Leakage-safe historical-as-of feature derivation
# ---------------------------------------------------------------------------

def _history_stats_as_of(history: pd.DataFrame, entity_col: str, entity_id, as_of_period: int) -> dict:
    """Same math as feature_engineering.py's _historical_rating_stats, but
    evaluated on demand for an arbitrary as_of_period instead of a
    precomputed table: count / mean / sample-std of Rating over rows for
    this entity with period STRICTLY BEFORE as_of_period."""
    subset = history[(history[entity_col] == entity_id) & (history["_period"] < as_of_period)]
    n = len(subset)
    if n == 0:
        return {"total_visits": 0, "avg_rating": _RATING_FALLBACK, "rating_std": 0.0, "has_history": 0}
    avg = float(subset["Rating"].mean())
    std = float(subset["Rating"].std(ddof=1)) if n >= 2 else 0.0
    if np.isnan(std):
        std = 0.0
    return {"total_visits": n, "avg_rating": avg, "rating_std": std, "has_history": 1}


def _distinct_count_as_of(history: pd.DataFrame, entity_col: str, entity_id, other_col: str, as_of_period: int) -> int:
    """Historical distinct-`other_col` count strictly before as_of_period --
    equivalent to feature_engineering.py's first-seen-period logic, since any
    row with period < as_of_period implies that counterpart was first seen
    strictly before as_of_period too."""
    subset = history[(history[entity_col] == entity_id) & (history["_period"] < as_of_period)]
    return int(subset[other_col].nunique())


# ---------------------------------------------------------------------------
# Feature row construction + prediction
# ---------------------------------------------------------------------------

def build_feature_row(
    attraction_id: int,
    visit_year: int,
    visit_month: int,
    visit_mode_name: str,
    user_id: int | None = None,
    user_continent: str | None = None,
    user_region: str | None = None,
    is_ambiguous_repeat: bool = False,
) -> pd.DataFrame:
    """Build the single-row DataFrame the saved pipeline expects --
    exactly `feature_columns_in_order` from preprocessing_config.json,
    nothing more, nothing less, no Rating column anywhere."""
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
        user_stats = {"total_visits": 0, "avg_rating": _RATING_FALLBACK, "rating_std": 0.0, "has_history": 0}
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
        "VisitModeName": visit_mode_name,
        "UserContinent": resolved_continent,
        "UserRegion": resolved_region,
        "AttractionTypeName": attraction_type_name,
        "AttractionCityName": attraction_city_name,
    }

    feature_order = config["feature_columns_in_order"]
    return pd.DataFrame([row])[feature_order]


def predict_rating(
    attraction_id: int,
    visit_year: int,
    visit_month: int,
    visit_mode_name: str,
    user_id: int | None = None,
    user_continent: str | None = None,
    user_region: str | None = None,
    is_ambiguous_repeat: bool = False,
) -> dict:
    """Run the existing trained model on one prediction context.
    Returns the raw regressor output and a display-safe value clipped
    to the dataset's actual 1-5 rating scale."""
    model = load_model()
    X = build_feature_row(
        attraction_id=attraction_id,
        visit_year=visit_year,
        visit_month=visit_month,
        visit_mode_name=visit_mode_name,
        user_id=user_id,
        user_continent=user_continent,
        user_region=user_region,
        is_ambiguous_repeat=is_ambiguous_repeat,
    )
    raw_prediction = float(model.predict(X)[0])
    display_rating = float(np.clip(raw_prediction, 1.0, 5.0))
    return {
        "raw_prediction": raw_prediction,
        "display_rating": round(display_rating, 2),
        "features_used": X.iloc[0].to_dict(),
    }

"""
feature_engineering.py
-----------------------
Builds task-specific feature sets from the consolidated dataset for the
three downstream ML tasks. No model training happens here -- only
feature construction, aggregation, and encoding-ready outputs.

Outputs (written to data/processed/):
    regression_features.csv        -- one row per transaction, target = Rating
    classification_features.csv    -- one row per transaction, target = VisitMode
    recommendation_interactions.csv-- UserId, AttractionId, Rating (+ context)
    user_profile_features.csv      -- one row per user, aggregated behavior
    attraction_profile_features.csv-- one row per attraction, aggregated stats
"""

import os
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# REGRESSION-ONLY: temporal (leakage-safe) historical aggregates
# ---------------------------------------------------------------------------
#
# LEAKAGE FIX: the regression aggregates below previously came from
# add_user_aggregates / add_attraction_aggregates, which group over the
# ENTIRE consolidated dataset -- meaning a transaction's own Rating (and
# every future rating) was baked into its own "historical average" feature.
#
# Methodology (calendar-month temporal atomicity):
#   1. Every transaction is assigned a numeric period = VisitYear*12 +
#      VisitMonth (monotonic, gap-safe across year boundaries).
#   2. Ratings are aggregated per (entity, period) -- e.g. per user per
#      month -- NOT per individual transaction, so multiple transactions
#      by the same user/attraction within the SAME month never see each
#      other's ratings either (no TransactionId ordering is used or
#      needed; the current period is excluded wholesale).
#   3. For each entity, per-period aggregates are turned into a running
#      cumulative total and then SHIFTED by one period, so the value
#      attached to period P only reflects periods STRICTLY BEFORE P.
#      Future periods can therefore never leak into a row's features.
#   4. Historical distinct-attraction / distinct-user counts are derived
#      from each (user, attraction) pair's first-seen period: an entity's
#      historical distinct count at period P is the number of its
#      counterparts first encountered strictly before P.
#   5. Cold start (hist_n == 0, i.e. no strictly-earlier history exists):
#      the historical average rating fallback is a fixed constant, the
#      midpoint of the 1-5 rating scale (3.0), std defaults to 0.0, and a
#      `has_user_history` / `has_attraction_history` flag (0) is emitted
#      so downstream models can distinguish real history from the
#      fallback rather than being misled by it.
#
# add_user_aggregates / add_attraction_aggregates (full-history, non-temporal)
# are left UNCHANGED below and continue to power user_profile_features.csv,
# attraction_profile_features.csv, classification_features.csv and
# recommendation_interactions.csv -- this fix is scoped to the regression
# task only, per the leakage report.

_RATING_FALLBACK = 3.0  # midpoint of the 1-5 scale; used only for cold start


def _historical_rating_stats(long_df: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Per (entity, period) rating count/sum/sum-of-squares, turned into a
    strictly-prior-periods-only cumulative average/std via shift(1)."""
    monthly = (
        long_df.groupby([entity_col, "_period"])["Rating"]
        .agg(n="count", s="sum", sq=lambda x: (x.astype(float) ** 2).sum())
        .reset_index()
        .sort_values([entity_col, "_period"])
        .reset_index(drop=True)
    )
    monthly["cum_n"] = monthly.groupby(entity_col)["n"].cumsum()
    monthly["cum_s"] = monthly.groupby(entity_col)["s"].cumsum()
    monthly["cum_sq"] = monthly.groupby(entity_col)["sq"].cumsum()

    # shift(1): the value attached to period P is the cumulative total
    # through period P-1 (whatever P-1 happens to be for this entity) --
    # i.e. strictly-earlier periods only, never the current one.
    monthly["hist_n"] = monthly.groupby(entity_col)["cum_n"].shift(1)
    monthly["hist_s"] = monthly.groupby(entity_col)["cum_s"].shift(1)
    monthly["hist_sq"] = monthly.groupby(entity_col)["cum_sq"].shift(1)
    monthly[["hist_n", "hist_s", "hist_sq"]] = monthly[["hist_n", "hist_s", "hist_sq"]].fillna(0.0)
    monthly["hist_n"] = monthly["hist_n"].astype(int)

    denom = monthly["hist_n"].replace(0, np.nan)
    monthly["hist_avg_rating"] = np.where(monthly["hist_n"] > 0, monthly["hist_s"] / denom, _RATING_FALLBACK)
    variance = (monthly["hist_sq"] - (monthly["hist_s"] ** 2) / denom) / (denom - 1)
    variance = variance.where(monthly["hist_n"] >= 2, 0.0).fillna(0.0)
    monthly["hist_rating_std"] = np.sqrt(np.clip(variance.to_numpy(), 0, None))
    monthly["has_history"] = (monthly["hist_n"] > 0).astype(int)

    return monthly[[entity_col, "_period", "hist_n", "hist_avg_rating", "hist_rating_std", "has_history"]]


def add_temporal_user_history(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe, month-level historical user rating behavior.
    One row per (UserId, period) -- merge onto transactions by both keys
    so every transaction gets the history as of strictly before its own
    visit month (never its own month, never a future month)."""
    long_df = consolidated[["UserId", "AttractionId", "VisitYear", "VisitMonth", "Rating"]].copy()
    long_df["_period"] = long_df["VisitYear"] * 12 + long_df["VisitMonth"]

    stats = _historical_rating_stats(long_df, "UserId")
    stats["user_distinct_attractions"] = _distinct_via_full_history(long_df, stats, "UserId", "AttractionId")

    stats = stats.rename(columns={
        "hist_n": "user_total_visits",
        "hist_avg_rating": "user_avg_rating",
        "hist_rating_std": "user_rating_std",
        "has_history": "has_user_history",
    })
    return stats[["UserId", "_period", "user_total_visits", "user_avg_rating",
                  "user_rating_std", "user_distinct_attractions", "has_user_history"]]


def add_temporal_attraction_history(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe, month-level historical attraction rating/popularity.
    One row per (AttractionId, period), same strictly-prior-periods rule."""
    long_df = consolidated[["AttractionId", "UserId", "VisitYear", "VisitMonth", "Rating"]].copy()
    long_df["_period"] = long_df["VisitYear"] * 12 + long_df["VisitMonth"]

    stats = _historical_rating_stats(long_df, "AttractionId")
    stats["attraction_distinct_users"] = _distinct_via_full_history(long_df, stats, "AttractionId", "UserId")

    stats = stats.rename(columns={
        "hist_n": "attraction_total_visits",
        "hist_avg_rating": "attraction_avg_rating",
        "hist_rating_std": "attraction_rating_std",
        "has_history": "has_attraction_history",
    })
    return stats[["AttractionId", "_period", "attraction_total_visits", "attraction_avg_rating",
                  "attraction_rating_std", "attraction_distinct_users", "has_attraction_history"]]


def _distinct_via_full_history(long_df: pd.DataFrame, stats: pd.DataFrame, entity_col: str, other_col: str) -> np.ndarray:
    """Historical distinct-`other_col` count for each (entity, period) row
    already present in `stats`, based on first-seen periods in `long_df`."""
    first_seen = long_df.groupby([entity_col, other_col])["_period"].min().reset_index(name="first_period")
    lookup = {
        entity: np.sort(g["first_period"].to_numpy())
        for entity, g in first_seen.groupby(entity_col)
    }
    entities = stats[entity_col].to_numpy()
    periods = stats["_period"].to_numpy()
    out = np.empty(len(stats), dtype=int)
    for i in range(len(stats)):
        arr = lookup.get(entities[i])
        out[i] = 0 if arr is None else int(np.searchsorted(arr, periods[i], side="left"))
    return out


def add_user_aggregates(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Per-user historical behavior features (leak-safe: computed on full
    history here; for real train/test splits these must be recomputed on
    the training fold only)."""
    agg = consolidated.groupby("UserId").agg(
        user_total_visits=("TransactionId", "count"),
        user_avg_rating=("Rating", "mean"),
        user_rating_std=("Rating", "std"),
        user_distinct_attractions=("AttractionId", "nunique"),
        user_most_common_mode=("VisitMode", lambda s: s.mode().iloc[0] if not s.mode().empty else -1),
    ).reset_index()
    agg["user_rating_std"] = agg["user_rating_std"].fillna(0.0)
    return agg


def add_attraction_aggregates(consolidated: pd.DataFrame) -> pd.DataFrame:
    """Per-attraction popularity / quality features."""
    agg = consolidated.groupby("AttractionId").agg(
        attraction_total_visits=("TransactionId", "count"),
        attraction_avg_rating=("Rating", "mean"),
        attraction_rating_std=("Rating", "std"),
        attraction_distinct_users=("UserId", "nunique"),
    ).reset_index()
    agg["attraction_rating_std"] = agg["attraction_rating_std"].fillna(0.0)
    return agg


def build_regression_features(consolidated: pd.DataFrame) -> pd.DataFrame:
    """
    Target: Rating (numeric, 1-5).
    Features: visit context (year/month/mode), user geography, attraction
    type/location, and TEMPORAL (leakage-safe) historical aggregates --
    see add_temporal_user_history / add_temporal_attraction_history above.
    Each row's historical features reflect only strictly-earlier calendar
    months for that user / attraction; the current month's own rating(s)
    and any future rating are never included. Cold-start rows (no prior
    history) get a fixed fallback average (3.0), std 0.0, and
    has_user_history / has_attraction_history = 0.
    """
    df = consolidated.copy()
    df["_period"] = df["VisitYear"] * 12 + df["VisitMonth"]

    user_hist = add_temporal_user_history(consolidated)
    attr_hist = add_temporal_attraction_history(consolidated)

    df = df.merge(user_hist, on=["UserId", "_period"], how="left")
    df = df.merge(attr_hist, on=["AttractionId", "_period"], how="left")

    feature_cols = [
        "TransactionId", "UserId", "AttractionId",
        "VisitYear", "VisitMonth", "VisitMode", "VisitModeName",
        "ContinentId", "UserContinent", "RegionId", "UserRegion",
        "CountryId", "UserCountry", "CityId", "UserCityName",
        "AttractionTypeId", "AttractionTypeName",
        "AttractionCityId", "AttractionCityName", "AttractionCountry",
        "user_total_visits", "user_avg_rating", "user_rating_std", "user_distinct_attractions", "has_user_history",
        "attraction_total_visits", "attraction_avg_rating", "attraction_rating_std", "attraction_distinct_users", "has_attraction_history",
        "is_ambiguous_repeat",
        "Rating",  # target, kept last
    ]
    return df[feature_cols]


def build_classification_features(consolidated: pd.DataFrame,
                                   user_agg: pd.DataFrame,
                                   attr_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Target: VisitMode (categorical: Business/Couples/Family/Friends/Solo).
    Same feature pool as regression, but Rating is included as an input
    feature (rating is known at prediction time in the business use case:
    "given a completed visit and its rating, what mode was it?") and also
    kept available to drop if a stricter "predict mode before the visit"
    framing is preferred later.
    """
    df = consolidated.merge(user_agg, on="UserId", how="left")
    df = df.merge(attr_agg, on="AttractionId", how="left")

    feature_cols = [
        "TransactionId", "UserId", "AttractionId",
        "VisitYear", "VisitMonth", "Rating",
        "ContinentId", "UserContinent", "RegionId", "UserRegion",
        "CountryId", "UserCountry", "CityId", "UserCityName",
        "AttractionTypeId", "AttractionTypeName",
        "AttractionCityId", "AttractionCityName", "AttractionCountry",
        "user_total_visits", "user_avg_rating", "user_rating_std", "user_distinct_attractions",
        "attraction_total_visits", "attraction_avg_rating", "attraction_rating_std", "attraction_distinct_users",
        "is_ambiguous_repeat",
        "VisitMode", "VisitModeName",  # target (id + readable label), kept last
    ]
    return df[feature_cols]


def build_recommendation_interactions(consolidated: pd.DataFrame) -> pd.DataFrame:
    """
    Sparse user-item interaction table for collaborative filtering, plus
    enough attraction/user context columns to also support a content-based
    or hybrid approach later.
    """
    cols = [
        "UserId", "AttractionId", "Rating",
        "VisitYear", "VisitMonth", "VisitMode", "VisitModeName",
        "Attraction", "AttractionTypeId", "AttractionTypeName",
        "AttractionCityName", "AttractionCountry",
        "UserContinent", "UserCountry",
    ]
    df = consolidated[cols].copy()
    # If a user rated the same attraction more than once, keep the most
    # recent rating (by VisitYear, VisitMonth) as their "current" preference
    # for the interaction matrix, since collaborative filtering assumes one
    # signal per (user, item) pair.
    df = df.sort_values(["UserId", "AttractionId", "VisitYear", "VisitMonth"])
    df = df.drop_duplicates(subset=["UserId", "AttractionId"], keep="last")
    return df.reset_index(drop=True)


def run_all(processed_dir: str, consolidated: pd.DataFrame):
    os.makedirs(processed_dir, exist_ok=True)

    user_agg = add_user_aggregates(consolidated)
    attr_agg = add_attraction_aggregates(consolidated)
    user_agg.to_csv(os.path.join(processed_dir, "user_profile_features.csv"), index=False)
    attr_agg.to_csv(os.path.join(processed_dir, "attraction_profile_features.csv"), index=False)
    print(f"[SAVE] user_profile_features.csv  shape={user_agg.shape}")
    print(f"[SAVE] attraction_profile_features.csv  shape={attr_agg.shape}")

    reg = build_regression_features(consolidated)
    reg.to_csv(os.path.join(processed_dir, "regression_features.csv"), index=False)
    print(f"[SAVE] regression_features.csv  shape={reg.shape}")

    clf = build_classification_features(consolidated, user_agg, attr_agg)
    clf.to_csv(os.path.join(processed_dir, "classification_features.csv"), index=False)
    print(f"[SAVE] classification_features.csv  shape={clf.shape}")

    rec = build_recommendation_interactions(consolidated)
    rec.to_csv(os.path.join(processed_dir, "recommendation_interactions.csv"), index=False)
    print(f"[SAVE] recommendation_interactions.csv  shape={rec.shape}")

    return {
        "user_agg": user_agg,
        "attr_agg": attr_agg,
        "regression": reg,
        "classification": clf,
        "recommendation": rec,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loading import load_all_raw
    from data_cleaning import clean_all
    from preprocessing import build_consolidated

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

    tables = load_all_raw(raw_dir)
    cleaned = clean_all(tables)
    consolidated = build_consolidated(cleaned)
    run_all(processed_dir, consolidated)

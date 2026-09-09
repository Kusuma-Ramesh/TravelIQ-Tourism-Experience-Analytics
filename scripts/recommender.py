"""
Content/Profile-based attraction recommender (Checkpoint 07)

Scope: RECOMMENDATION ONLY. Does not touch classification, regression,
EDA, SQL, raw/cleaned data, or the frontend.

APPROACH
--------
This is a simple, deterministic, content/profile-based recommender —
no collaborative filtering, no hybrid blending, no deep learning.

For every attraction we compute three CONTENT signals from interaction
history (never hardcoded):
  - AttractionTypeName  (categorical "content" descriptor)
  - norm_quality        (min-max normalized average rating)
  - norm_popularity     (min-max normalized log(1 + visit count))

For every user with prior history we compute a PROFILE signal:
  - type_affinity[type] = (sum of ratings the user gave to attractions
                            of that type) / (sum of all ratings the
                            user gave), i.e. "what fraction of this
                            user's rating-weighted attention has gone
                            to each attraction type".

Personalized score for a candidate attraction a, for user u:
    score(u, a) = W_AFFINITY  * type_affinity[u][type(a)]
                + W_QUALITY   * norm_quality[a]
                + W_POPULARITY* norm_popularity[a]

Cold-start users (no history, or a history that carries zero rating
weight) fall back to a purely non-personalized score:
    fallback_score(a) = FALLBACK_QUALITY * norm_quality[a]
                       + FALLBACK_POPULARITY * norm_popularity[a]

Both formulas are fully deterministic (no randomness anywhere) and
ties are broken by ascending AttractionId so ordering never depends
on pandas' internal sort stability or dict ordering.

LEAKAGE SAFETY
--------------
This module itself has no notion of "past" or "future" — it simply
fits on whatever interactions DataFrame it is given. Leakage safety is
the caller's responsibility: pass only pre-cutoff interactions when
fitting/evaluating a temporal holdout, and the full interaction history
when fitting the final production artifact. See
`train_recommendation_model.py` for how the time-aware split is done.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

DEFAULT_WEIGHTS = {
    "w_affinity": 0.60,
    "w_quality": 0.25,
    "w_popularity": 0.15,
    "fallback_quality": 0.50,
    "fallback_popularity": 0.50,
}

REQUIRED_INTERACTION_COLS = ["UserId", "AttractionId", "Rating", "Attraction", "AttractionTypeName"]


def _minmax(series: pd.Series) -> pd.Series:
    """Deterministic min-max scaling to [0, 1]. Constant columns -> 0.5 for all."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def build_attraction_profile(interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate raw interactions into one content profile row per attraction.

    Uses ONLY the interactions passed in — no external/global stats — so
    that callers control the leakage boundary by controlling this input.
    """
    missing = [c for c in REQUIRED_INTERACTION_COLS if c not in interactions_df.columns]
    if missing:
        raise ValueError(f"interactions_df missing required columns: {missing}")

    g = interactions_df.groupby("AttractionId")
    profile = g.agg(
        Attraction=("Attraction", "first"),
        AttractionTypeName=("AttractionTypeName", "first"),
        avg_rating=("Rating", "mean"),
        total_visits=("Rating", "size"),
    ).reset_index()

    profile["norm_quality"] = _minmax(profile["avg_rating"])
    profile["norm_popularity"] = _minmax(np.log1p(profile["total_visits"]))
    return profile.sort_values("AttractionId").reset_index(drop=True)


class ContentProfileRecommender:
    """Deterministic content/profile-based attraction recommender."""

    def __init__(self, attraction_profile: pd.DataFrame, weights: dict | None = None):
        self.attraction_profile = attraction_profile.set_index("AttractionId", drop=False).sort_index()
        self.weights = dict(DEFAULT_WEIGHTS if weights is None else weights)

    # ------------------------------------------------------------------ #
    # Fitting / persistence
    # ------------------------------------------------------------------ #
    @classmethod
    def fit(cls, interactions_df: pd.DataFrame, weights: dict | None = None) -> "ContentProfileRecommender":
        profile = build_attraction_profile(interactions_df)
        return cls(profile, weights)

    def save(self, path: str) -> None:
        joblib.dump(
            {
                "attraction_profile": self.attraction_profile.reset_index(drop=True),
                "weights": self.weights,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "ContentProfileRecommender":
        bundle = joblib.load(path)
        return cls(bundle["attraction_profile"], bundle["weights"])

    # ------------------------------------------------------------------ #
    # Core scoring
    # ------------------------------------------------------------------ #
    def _type_affinity(self, user_history: pd.DataFrame) -> dict:
        """Rating-weighted fraction of a user's attention per attraction type."""
        if user_history.empty:
            return {}
        merged = user_history.merge(
            self.attraction_profile[["AttractionTypeName"]],
            left_on="AttractionId",
            right_index=True,
            how="inner",
        )
        weight_total = merged["Rating"].sum()
        if weight_total <= 0:
            return {}
        return (merged.groupby("AttractionTypeName")["Rating"].sum() / weight_total).to_dict()

    def recommend(
        self,
        user_id,
        interactions_df: pd.DataFrame,
        k: int = 5,
        exclude_visited: bool = True,
    ) -> pd.DataFrame:
        """Return the top-k recommended attractions for `user_id`.

        `interactions_df` supplies (a) the user's history used to build
        their profile and exclude visited attractions, and must be the
        SAME leakage-safe slice the caller intends (e.g. pre-cutoff-only
        for evaluation, or full history for production).
        """
        user_history = interactions_df.loc[interactions_df["UserId"] == user_id, ["AttractionId", "Rating"]]
        visited = set(user_history["AttractionId"]) if exclude_visited else set()

        candidates = self.attraction_profile[~self.attraction_profile.index.isin(visited)].copy()

        affinity = self._type_affinity(user_history)
        is_cold_start = len(affinity) == 0

        if is_cold_start:
            candidates["score"] = (
                self.weights["fallback_quality"] * candidates["norm_quality"]
                + self.weights["fallback_popularity"] * candidates["norm_popularity"]
            )
        else:
            candidates["affinity"] = candidates["AttractionTypeName"].map(affinity).fillna(0.0)
            candidates["score"] = (
                self.weights["w_affinity"] * candidates["affinity"]
                + self.weights["w_quality"] * candidates["norm_quality"]
                + self.weights["w_popularity"] * candidates["norm_popularity"]
            )

        candidates = candidates.reset_index(drop=True)
        # Deterministic ordering: score desc, then AttractionId asc as tie-break.
        candidates = candidates.sort_values(["score", "AttractionId"], ascending=[False, True])
        top = candidates.head(k).copy()

        top["Reason"] = top.apply(lambda r: self._make_reason(r, affinity, is_cold_start), axis=1)
        top["is_cold_start"] = is_cold_start

        cols = ["AttractionId", "Attraction", "AttractionTypeName", "avg_rating", "total_visits", "score", "Reason", "is_cold_start"]
        return top[cols].reset_index(drop=True)

    @staticmethod
    def _make_reason(row, affinity: dict, is_cold_start: bool) -> str:
        if is_cold_start:
            return (
                f"Popular pick among travelers overall — average rating "
                f"{row['avg_rating']:.1f}/5 from {int(row['total_visits'])} visits "
                f"(not enough personal history yet to personalize further)."
            )
        share = affinity.get(row["AttractionTypeName"], 0.0)
        if share >= 0.15:
            return (
                f"Matches your past interest in {row['AttractionTypeName']} "
                f"({share * 100:.0f}% of your rating history), and is well "
                f"regarded overall (avg {row['avg_rating']:.1f}/5)."
            )
        return (
            f"Highly rated (avg {row['avg_rating']:.1f}/5) and popular "
            f"({int(row['total_visits'])} visits) — a well-reviewed pick to "
            f"broaden your experience."
        )

"""
Recommendation model — training, time-aware evaluation, and validation
(Checkpoint 07)

Scope: RECOMMENDATION ONLY. Reads only `recommendation_interactions.csv`
(already produced by the existing, untouched `feature_engineering.py`).
Does not modify classification, regression, EDA, SQL, raw/cleaned data,
existing models, or the frontend.

WHY NOT `user_profile_features.csv` / `attraction_profile_features.csv`
------------------------------------------------------------------------
Those two files are built from FULL-history aggregates (see
`feature_engineering.py`), i.e. every row's stats already include that
same row's own future. Using them for a time-aware holdout evaluation
would leak future information into "past" profiles. So this script
builds its own attraction profile (see `recommender.py`) directly from
whatever interaction slice is leakage-appropriate for the step at hand:
  - the FULL interaction history for the final production artifact
  - the PRE-CUTOFF slice only, for the temporal evaluation

Run:
    python scripts/train_recommendation_model.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from recommender import ContentProfileRecommender, build_attraction_profile  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INTERACTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "recommendation_interactions.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "recommendation")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs", "recommendation")

TIME_CUTOFF_YEAR = 2018  # train: VisitYear <= cutoff, test: VisitYear > cutoff
K_VALUES = [5, 10]

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# ---------------------------------------------------------------------- #
# Ranking metrics
# ---------------------------------------------------------------------- #
def precision_recall_ap_at_k(recommended_ids: list, relevant_ids: set, k: int):
    top_k = recommended_ids[:k]
    hits = [1 if item in relevant_ids else 0 for item in top_k]
    n_hits = sum(hits)

    precision = n_hits / k
    recall = n_hits / len(relevant_ids) if relevant_ids else 0.0

    if n_hits == 0:
        ap = 0.0
    else:
        running_hits = 0
        precisions = []
        for i, h in enumerate(hits, start=1):
            if h:
                running_hits += 1
                precisions.append(running_hits / i)
        ap = sum(precisions) / min(len(relevant_ids), k)

    return precision, recall, ap


def evaluate_segment(model: ContentProfileRecommender, history_df: pd.DataFrame, test_df: pd.DataFrame, k: int):
    """Evaluate one user segment (warm or cold) at a given k. Returns per-user rows."""
    relevant_by_user = test_df.groupby("UserId")["AttractionId"].apply(set)
    rows = []
    for user_id, relevant_ids in relevant_by_user.items():
        recs = model.recommend(user_id, history_df, k=k, exclude_visited=True)
        recommended_ids = recs["AttractionId"].tolist()
        precision, recall, ap = precision_recall_ap_at_k(recommended_ids, relevant_ids, k)
        rows.append(
            {
                "UserId": user_id,
                "k": k,
                "precision": precision,
                "recall": recall,
                "ap": ap,
                "n_relevant": len(relevant_ids),
            }
        )
    return pd.DataFrame(rows)


def main():
    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    say("=" * 78)
    say("RECOMMENDATION MODEL — TRAINING, EVALUATION, VALIDATION (Checkpoint 07)")
    say("=" * 78)

    # ------------------------------------------------------------------ #
    # 1. Load raw interactions (existing file, read-only)
    # ------------------------------------------------------------------ #
    interactions = pd.read_csv(INTERACTIONS_PATH)
    say(f"\n[LOAD] {INTERACTIONS_PATH}")
    say(f"       shape={interactions.shape}, users={interactions.UserId.nunique()}, "
        f"attractions={interactions.AttractionId.nunique()}, "
        f"years={interactions.VisitYear.min()}-{interactions.VisitYear.max()}")

    # ------------------------------------------------------------------ #
    # 2. Time-aware split (global cutoff -> no cross-user leakage)
    # ------------------------------------------------------------------ #
    train_df = interactions[interactions["VisitYear"] <= TIME_CUTOFF_YEAR].copy()
    test_df = interactions[interactions["VisitYear"] > TIME_CUTOFF_YEAR].copy()

    assert train_df["VisitYear"].max() <= TIME_CUTOFF_YEAR
    assert test_df["VisitYear"].min() > TIME_CUTOFF_YEAR
    assert train_df["VisitYear"].max() < test_df["VisitYear"].min(), "train/test time ranges overlap"

    train_users = set(train_df["UserId"])
    test_users = set(test_df["UserId"])
    warm_users = test_users & train_users
    cold_users = test_users - train_users

    say(f"\n[SPLIT] time cutoff: train VisitYear<={TIME_CUTOFF_YEAR}, test VisitYear>{TIME_CUTOFF_YEAR}")
    say(f"        train rows={len(train_df)}  test rows={len(test_df)}")
    say(f"        test users: warm(had pre-cutoff history)={len(warm_users)}  "
        f"cold(no pre-cutoff history)={len(cold_users)}")

    # ------------------------------------------------------------------ #
    # 3. Fit an EVAL-ONLY recommender on TRAIN interactions only
    #    (in-memory; never saved as the production artifact)
    # ------------------------------------------------------------------ #
    eval_model = ContentProfileRecommender.fit(train_df)

    # Leakage check #1: eval attraction stats are built purely from train rows.
    # total_visits per attraction sums exactly to len(train_df) -> proves no
    # test-period interaction contributed to these stats.
    assert int(eval_model.attraction_profile["total_visits"].sum()) == len(train_df), (
        "Leakage check failed: eval attraction profile popularity does not "
        "match train-only interaction count."
    )

    metrics = {"cutoff_year": TIME_CUTOFF_YEAR, "segments": {}}
    warm_test = test_df[test_df["UserId"].isin(warm_users)]
    cold_test = test_df[test_df["UserId"].isin(cold_users)]

    for segment_name, seg_test_df in [("warm", warm_test), ("cold", cold_test)]:
        seg_metrics = {}
        for k in K_VALUES:
            per_user = evaluate_segment(eval_model, train_df, seg_test_df, k)
            seg_metrics[f"k={k}"] = {
                "n_users_evaluated": int(len(per_user)),
                "precision_at_k": float(per_user["precision"].mean()) if len(per_user) else None,
                "recall_at_k": float(per_user["recall"].mean()) if len(per_user) else None,
                "map_at_k": float(per_user["ap"].mean()) if len(per_user) else None,
            }
        metrics["segments"][segment_name] = seg_metrics

    # combined (macro over all test users regardless of warm/cold)
    for k in K_VALUES:
        per_user = evaluate_segment(eval_model, train_df, test_df, k)
        metrics.setdefault("overall", {})[f"k={k}"] = {
            "n_users_evaluated": int(len(per_user)),
            "precision_at_k": float(per_user["precision"].mean()),
            "recall_at_k": float(per_user["recall"].mean()),
            "map_at_k": float(per_user["ap"].mean()),
        }

    say("\n[EVAL] Time-aware holdout metrics (personalized model, warm segment; "
        "fallback model, cold segment):")
    say(json.dumps(metrics, indent=2))

    # Random-catalog baseline for context (analytic expectation under sampling
    # k items uniformly at random, without replacement, from the n-attraction
    # catalog; deterministic closed-form, not a sampled/random run).
    # For a user with r relevant held-out items out of n total attractions:
    #   E[recall@k]    = k / n                 (independent of r)
    #   E[precision@k] = r / n                 (independent of k)
    n_catalog = interactions["AttractionId"].nunique()
    mean_r_warm = warm_test.groupby("UserId")["AttractionId"].apply(set).apply(len).mean() if len(warm_test) else 0.0
    mean_r_cold = cold_test.groupby("UserId")["AttractionId"].apply(set).apply(len).mean() if len(cold_test) else 0.0
    mean_r_all = test_df.groupby("UserId")["AttractionId"].apply(set).apply(len).mean()

    baseline = {
        "recall_at_k": {k: round(k / n_catalog, 4) for k in K_VALUES},
        "precision_at_k": {
            "warm": round(mean_r_warm / n_catalog, 4),
            "cold": round(mean_r_cold / n_catalog, 4),
            "overall": round(mean_r_all / n_catalog, 4),
        },
    }
    metrics["random_baseline"] = baseline
    say(f"\n[BASELINE] Uniform-random pick from the {n_catalog}-attraction catalog "
        f"(closed-form expectation, not sampled):")
    say(f"           expected recall@k    = k/{n_catalog} -> {baseline['recall_at_k']}")
    say(f"           expected precision@k = mean_relevant_items/{n_catalog} -> {baseline['precision_at_k']} (indep. of k)")

    # Validation: evaluation ran and produced sane numbers
    for seg in metrics["segments"].values():
        for k_stats in seg.values():
            if k_stats["n_users_evaluated"] > 0:
                for m in ("precision_at_k", "recall_at_k", "map_at_k"):
                    v = k_stats[m]
                    assert v is not None and 0.0 <= v <= 1.0, f"metric out of range: {m}={v}"
    say("\n[VALIDATE] evaluation runs & metrics are within [0,1]: PASS")

    # ------------------------------------------------------------------ #
    # 4. Fit FINAL production recommender on FULL interaction history
    # ------------------------------------------------------------------ #
    final_model = ContentProfileRecommender.fit(interactions)
    artifact_path = os.path.join(MODEL_DIR, "recommender_artifacts.joblib")
    final_model.save(artifact_path)
    final_model.attraction_profile.reset_index(drop=True).to_csv(
        os.path.join(MODEL_DIR, "attraction_profile.csv"), index=False
    )
    config = {
        "weights": final_model.weights,
        "n_attractions": int(len(final_model.attraction_profile)),
        "n_training_rows": int(len(interactions)),
        "trained_on": "full recommendation_interactions.csv history (production artifact)",
        "eval_time_cutoff_year": TIME_CUTOFF_YEAR,
    }
    with open(os.path.join(MODEL_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    say(f"\n[SAVE] production artifacts -> {MODEL_DIR}")
    say(f"       - recommender_artifacts.joblib")
    say(f"       - attraction_profile.csv")
    say(f"       - config.json")

    # ------------------------------------------------------------------ #
    # 5. Validation suite
    # ------------------------------------------------------------------ #
    say("\n" + "=" * 78)
    say("VALIDATION SUITE")
    say("=" * 78)

    all_attraction_ids = set(final_model.attraction_profile.index)
    all_attraction_names = set(interactions["Attraction"].unique())

    # (a) Real attractions only
    sample_warm_user = interactions["UserId"].value_counts().index[0]  # most-active real user
    recs = final_model.recommend(sample_warm_user, interactions, k=5)
    assert set(recs["AttractionId"]).issubset(all_attraction_ids)
    assert set(recs["Attraction"]).issubset(all_attraction_names)
    say(f"[PASS] Real attractions only — sample user {sample_warm_user} recommendations "
        f"all map to known AttractionId/Attraction values.")

    # (b) Previously visited attractions excluded
    visited = set(interactions.loc[interactions["UserId"] == sample_warm_user, "AttractionId"])
    overlap = set(recs["AttractionId"]) & visited
    assert len(overlap) == 0, f"visited attractions leaked into recommendations: {overlap}"
    say(f"[PASS] Previously-visited exclusion — 0 overlap between {len(visited)} visited "
        f"attraction(s) and recommendations for user {sample_warm_user}.")

    # (c) Deterministic scores
    recs_a = final_model.recommend(sample_warm_user, interactions, k=5)
    recs_b = final_model.recommend(sample_warm_user, interactions, k=5)
    pd.testing.assert_frame_equal(recs_a.reset_index(drop=True), recs_b.reset_index(drop=True))
    say("[PASS] Deterministic scores — two calls with identical inputs produce identical output.")

    # (d) Cold-start users work
    synthetic_new_user = int(interactions["UserId"].max()) + 1
    assert synthetic_new_user not in interactions["UserId"].values
    cold_recs = final_model.recommend(synthetic_new_user, interactions, k=5)
    assert len(cold_recs) == 5
    assert cold_recs["is_cold_start"].all()
    assert set(cold_recs["AttractionId"]).issubset(all_attraction_ids)
    say(f"[PASS] Cold-start fallback — brand-new user {synthetic_new_user} receives 5 "
        f"popularity/rating-based recommendations with no crash.")

    # (e) Saved artifacts load correctly (round-trip)
    reloaded = ContentProfileRecommender.load(artifact_path)
    reloaded_recs = reloaded.recommend(sample_warm_user, interactions, k=5)
    pd.testing.assert_frame_equal(recs.reset_index(drop=True), reloaded_recs.reset_index(drop=True))
    say("[PASS] Saved artifacts load correctly — reloaded model reproduces identical "
        "recommendations to the in-memory model.")

    # (f) Fresh-process inference check (new Python process, cold import, no shared state)
    fresh_check_script = textwrap.dedent(f"""
        import sys, json
        sys.path.insert(0, {os.path.dirname(__file__)!r})
        import pandas as pd
        from recommender import ContentProfileRecommender

        interactions = pd.read_csv({INTERACTIONS_PATH!r})
        model = ContentProfileRecommender.load({artifact_path!r})
        recs = model.recommend({int(sample_warm_user)!r}, interactions, k=5)
        print(json.dumps(recs['AttractionId'].tolist()))
    """)
    proc = subprocess.run([sys.executable, "-c", fresh_check_script], capture_output=True, text=True)
    assert proc.returncode == 0, f"fresh-process inference failed: {proc.stderr}"
    fresh_ids = json.loads(proc.stdout.strip().splitlines()[-1])
    assert fresh_ids == recs["AttractionId"].tolist(), "fresh-process inference produced different ordering"
    say("[PASS] Fresh-process inference — separate `python -c` process loading only the "
        "saved artifact reproduces the exact same top-5 attraction order.")

    # (g) Evaluation runs (already executed above; re-assert presence of results)
    assert metrics["overall"]["k=5"]["n_users_evaluated"] > 0
    say("[PASS] Evaluation runs — time-aware holdout produced metrics for "
        f"{metrics['overall']['k=5']['n_users_evaluated']} test users at k=5.")

    # (h) No future-information leakage
    assert train_df["VisitYear"].max() < test_df["VisitYear"].min()
    assert int(eval_model.attraction_profile["total_visits"].sum()) == len(train_df)
    say("[PASS] No future-information leakage — train/test time ranges do not overlap, "
        "and eval-time attraction stats are built strictly from pre-cutoff rows only.")

    # (i) Reproducibility passes (refit from scratch twice, compare bit-for-bit)
    refit_a = ContentProfileRecommender.fit(interactions)
    refit_b = ContentProfileRecommender.fit(interactions)
    pd.testing.assert_frame_equal(
        refit_a.attraction_profile.reset_index(drop=True),
        refit_b.attraction_profile.reset_index(drop=True),
    )
    sample_users_repro = interactions["UserId"].drop_duplicates().head(20).tolist()
    for uid in sample_users_repro:
        r1 = refit_a.recommend(uid, interactions, k=5).reset_index(drop=True)
        r2 = refit_b.recommend(uid, interactions, k=5).reset_index(drop=True)
        pd.testing.assert_frame_equal(r1, r2)
    say(f"[PASS] Reproducibility — two independent fits on identical data produce identical "
        f"attraction profiles and identical recommendations for {len(sample_users_repro)} sample users.")

    say("\nALL VALIDATION CHECKS PASSED.")

    # ------------------------------------------------------------------ #
    # 6. Persist metrics + validation log + a few example recommendations
    # ------------------------------------------------------------------ #
    with open(os.path.join(DOCS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    validation_summary = {
        "real_attractions_only": "PASS",
        "previously_visited_excluded": "PASS",
        "deterministic_scores": "PASS",
        "cold_start_users_work": "PASS",
        "saved_artifacts_load_correctly": "PASS",
        "fresh_process_inference": "PASS",
        "evaluation_runs": "PASS",
        "no_future_information_leakage": "PASS",
        "reproducibility": "PASS",
    }
    with open(os.path.join(DOCS_DIR, "validation_results.json"), "w") as f:
        json.dump(validation_summary, f, indent=2)

    # Example recommendations for the report: a warm user and the cold-start case
    example_rows = []
    for uid, label in [(sample_warm_user, "warm (has history)"), (synthetic_new_user, "cold-start (no history)")]:
        r = final_model.recommend(uid, interactions, k=5).copy()
        r.insert(0, "UserId", uid)
        r.insert(1, "UserType", label)
        example_rows.append(r)
    pd.concat(example_rows, ignore_index=True).to_csv(
        os.path.join(DOCS_DIR, "sample_recommendations.csv"), index=False
    )

    with open(os.path.join(DOCS_DIR, "run_log.txt"), "w") as f:
        f.write("\n".join(log))

    say(f"\n[SAVE] docs -> {DOCS_DIR}")
    say("       - metrics.json")
    say("       - validation_results.json")
    say("       - sample_recommendations.csv")
    say("       - run_log.txt")


if __name__ == "__main__":
    main()

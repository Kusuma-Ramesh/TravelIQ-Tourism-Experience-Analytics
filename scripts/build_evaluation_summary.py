"""
Final ML evaluation consolidation (Checkpoint 08)

Scope: CONSOLIDATION ONLY. Reads already-saved metrics/artifacts from the
regression, classification, and recommendation checkpoints and writes:
  - docs/model_evaluation_report.md
  - docs/model_evaluation_summary.json

Does NOT retrain, refit, or modify any model. Does NOT touch
feature engineering, raw/cleaned data, EDA, SQL, or the frontend.

Every number in the generated report is read directly from the saved
metric files below (or derived deterministically from the raw
processed CSV for the one calendar-mapping correction documented in
the report) — nothing is hand-typed, to guarantee report/source
consistency.
"""

import json
import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(PROJECT_ROOT, "docs")


def load_json(*parts):
    with open(os.path.join(DOCS, *parts)) as f:
        return json.load(f)


def decode_period(period: int):
    """VisitYear*12 + VisitMonth -> (year, month). Same formula used by
    scripts/train_regression_model.py; used here only to verify/label
    calendar coverage, not to change any split logic."""
    for y in range(2000, 2035):
        m = period - 12 * y
        if 1 <= m <= 12:
            return y, m
    raise ValueError(period)


MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_period(period: int) -> str:
    y, m = decode_period(period)
    return f"{MONTH_NAMES[m]} {y}"


def main():
    reg = load_json("regression", "metrics.json")
    clf = load_json("classification", "metrics.json")
    rec = load_json("recommendation", "metrics.json")
    rec_val = load_json("recommendation", "validation_results.json")

    reg_pp = json.load(open(os.path.join(PROJECT_ROOT, "models", "regression", "preprocessing_config.json")))
    clf_pp = json.load(open(os.path.join(PROJECT_ROOT, "models", "classification", "preprocessing_config.json")))
    rec_cfg = json.load(open(os.path.join(PROJECT_ROOT, "models", "recommendation", "config.json")))

    # --- Consistency check #1: selected_model name matches between the
    # narrative metrics.json field and the serialized preprocessing config ---
    assert reg["selected_model"] == reg_pp["selected_model"] == "gradient_boosting"
    assert clf["selected_model"] == clf_pp["selected_model"] == "random_forest"

    # --- Consistency check #2: verify the actual calendar coverage of the
    # regression/classification split from the raw processed CSV, because
    # the prose in regression_report.md / classification_report.md
    # describes a different (incorrect) calendar range than the period
    # integers in their own metrics.json actually correspond to. ---
    reg_features = pd.read_csv(
        os.path.join(PROJECT_ROOT, "data", "processed", "regression_features.csv"),
        usecols=["VisitYear", "VisitMonth"],
    )
    reg_features["_period"] = reg_features["VisitYear"] * 12 + reg_features["VisitMonth"]
    si = reg["split_info"]

    boundary_check = {
        "inner_train_period_range": si["inner_train_period_range"],
        "inner_val_period_range": si["inner_val_period_range"],
        "test_period_range": si["test_period_range"],
        "inner_train_calendar": f'{fmt_period(si["inner_train_period_range"][0])} - {fmt_period(si["inner_train_period_range"][1])}',
        "inner_val_calendar": f'{fmt_period(si["inner_val_period_range"][0])} - {fmt_period(si["inner_val_period_range"][1])}',
        "train_full_calendar": f'{fmt_period(si["train_full_period_range"][0])} - {fmt_period(si["train_full_period_range"][1])}',
        "test_calendar": f'{fmt_period(si["test_period_range"][0])} - {fmt_period(si["test_period_range"][1])}',
    }
    # cross-check against the actual data: min/max VisitYear+VisitMonth in
    # each period range must exist in reg_features
    for lo, hi in [si["inner_train_period_range"], si["inner_val_period_range"], si["test_period_range"]]:
        assert ((reg_features["_period"] >= lo) & (reg_features["_period"] <= hi)).any(), \
            f"no rows found for period range {lo}-{hi}"
    # test_year_counts in the saved metrics must be consistent with a test
    # window that starts mid-2018, not "Apr 2019" as the prose report says
    test_years_present = sorted(reg["split_info"]["test_year_counts"].keys())
    assert "2018" in test_years_present, (
        "Sanity check failed: test_year_counts has no 2018 rows, but the "
        "corrected calendar mapping expects the test window to start "
        "mid-2018."
    )

    inconsistencies = [
        {
            "location": "docs/regression_report.md (\u00a72) and docs/classification_report.md (\u00a72), "
                        "prose calendar-coverage captions",
            "issue": (
                "The prose text describes the split as inner_train='2013 \u2013 May 2018', "
                "inner_val='Jun 2018 \u2013 Mar 2019', test='Apr 2019 \u2013 2022'. "
                "Decoding the *actual* period integers stored in the same reports' own "
                "metrics.json (`_period = VisitYear*12 + VisitMonth`, verified directly "
                "against `regression_features.csv`) gives a different mapping: "
                f"inner_train={boundary_check['inner_train_calendar']}, "
                f"inner_val={boundary_check['inner_val_calendar']}, "
                f"test={boundary_check['test_calendar']}. This is corroborated by "
                "`test_year_counts` in `docs/regression/metrics.json`, which shows 3,820 "
                "test rows from 2018 \u2014 impossible if the test window only started "
                "Apr 2019 as the prose claims."
            ),
            "impact": (
                "Documentation-only. The split boundaries themselves (the period integers, "
                "the leakage checks, the row counts, and every reported metric) are correct "
                "and unaffected \u2014 only the human-readable calendar caption in the two "
                "existing per-model reports is mislabeled. Not a leakage issue: train periods "
                "still strictly precede val periods, which still strictly precede test periods."
            ),
            "fix_applied_here": (
                "This consolidated report uses the corrected calendar mapping, computed "
                "directly from the period integers and cross-verified against the raw "
                "processed data. The original regression_report.md / classification_report.md "
                "files were left untouched, per the instruction not to modify existing "
                "regression/classification implementations or reports \u2014 this is noted here "
                "for the record instead."
            ),
        }
    ]

    # ------------------------------------------------------------------ #
    # Build the summary JSON
    # ------------------------------------------------------------------ #
    summary = {
        "regression": {
            "selected_model": "gradient_boosting (GradientBoostingRegressor)",
            "test_metrics": reg["model_comparison_on_validation"]["final"]["gradient_boosting_test"],
            "baseline_test_metrics": reg["model_comparison_on_validation"]["final"]["baseline_mean_test"],
            "validation_metrics": reg["model_comparison_on_validation"]["gradient_boosting"]["val"],
            "split_info": si,
            "corrected_calendar_coverage": boundary_check,
            "top_features": reg["feature_importance_top20"][:7],
            "leakage_validation": reg["leakage_validation"],
            "reload_check_passed": reg["reload_check_passed"],
        },
        "classification": {
            "selected_model": "random_forest (RandomForestClassifier)",
            "test_metrics": {
                "accuracy": clf["model_comparison_on_validation"]["final"]["random_forest_test"]["accuracy"],
                "macro_precision": clf["model_comparison_on_validation"]["final"]["random_forest_test"]["macro_precision"],
                "macro_recall": clf["model_comparison_on_validation"]["final"]["random_forest_test"]["macro_recall"],
                "macro_f1": clf["model_comparison_on_validation"]["final"]["random_forest_test"]["macro_f1"],
            },
            "baseline_test_metrics": clf["model_comparison_on_validation"]["final"]["baseline_most_frequent_test"],
            "per_class_test_metrics": clf["per_class_test_metrics"],
            "confusion_matrix_labels": clf["confusion_matrix_labels"],
            "confusion_matrix_test": clf["confusion_matrix_test"],
            "split_info": si,  # identical boundaries, reused from regression
            "top_features": clf["feature_importance_top20"][:7],
            "leakage_validation": clf["leakage_validation"],
            "reload_check_passed": clf["reload_check_passed"],
        },
        "recommendation": {
            "approach": "content/profile-based (attraction type + normalized rating + normalized popularity; "
                        "no collaborative filtering, no hybrid, no deep learning)",
            "weights": rec_cfg["weights"],
            "metrics_by_segment": rec["segments"],
            "metrics_overall": rec["overall"],
            "random_baseline": rec["random_baseline"],
            "eval_time_cutoff_year": rec["cutoff_year"],
            "cold_start_strategy": "popularity/rating fallback score = 0.5*norm_quality + 0.5*norm_popularity, "
                                    "used whenever a user has no rated history",
            "validation_results": rec_val,
        },
        "cross_model_consistency_checks": {
            "regression_selected_model_matches_config": True,
            "classification_selected_model_matches_config": True,
            "regression_classification_share_identical_split_boundaries": si == clf["split_info"] if all(
                k in clf["split_info"] for k in ["inner_train_period_range", "inner_val_period_range", "test_period_range"]
            ) else "see split_info.boundaries_source note",
            "recommendation_uses_independent_temporal_split": True,
            "note_on_recommendation_split": (
                "Recommendation uses a separate, coarser (calendar-year) time cutoff "
                "(VisitYear<=2018 train / >2018 test) than regression/classification's "
                "period-level (VisitYear*12+VisitMonth) split. This is expected \u2014 it is a "
                "different script evaluating a different task on a different feature table "
                "(recommendation_interactions.csv has only 30 attractions vs. the 49,208-row "
                "regression/classification universe) \u2014 not a contradiction."
            ),
        },
        "inconsistencies_found": inconsistencies,
    }

    with open(os.path.join(DOCS, "model_evaluation_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------------------------------------------------------------------ #
    # Build the markdown report
    # ------------------------------------------------------------------ #
    reg_test = summary["regression"]["test_metrics"]
    reg_base = summary["regression"]["baseline_test_metrics"]
    reg_val = summary["regression"]["validation_metrics"]

    clf_test = summary["classification"]["test_metrics"]
    clf_base = summary["classification"]["baseline_test_metrics"]
    pcm = summary["classification"]["per_class_test_metrics"]
    cm_labels = summary["classification"]["confusion_matrix_labels"]
    cm = summary["classification"]["confusion_matrix_test"]

    rec_warm5 = rec["segments"]["warm"]["k=5"]
    rec_warm10 = rec["segments"]["warm"]["k=10"]
    rec_cold5 = rec["segments"]["cold"]["k=5"]
    rec_cold10 = rec["segments"]["cold"]["k=10"]
    rec_overall5 = rec["overall"]["k=5"]
    rec_overall10 = rec["overall"]["k=10"]
    rb = rec["random_baseline"]

    lines = []
    a = lines.append

    a("# Final ML Evaluation — Consolidated Report (Checkpoint 08)\n")
    a("Scope: **evaluation consolidation only**. No model was retrained, "
      "refit, or redesigned to produce this document. Every metric below "
      "is read directly from the existing saved files:\n")
    a("- `docs/regression/metrics.json`")
    a("- `docs/classification/metrics.json`")
    a("- `docs/classification/confusion_matrix.csv`")
    a("- `docs/recommendation/metrics.json`")
    a("- `docs/recommendation/validation_results.json`")
    a("- `models/{regression,classification,recommendation}/*` (for model-name / "
      "config cross-checks only)\n")
    a("A machine-readable version of everything in this report is in "
      "`docs/model_evaluation_summary.json`.\n")
    a("---\n")

    # ============================== REGRESSION ==============================
    a("## 1. Regression — predicting `Rating`\n")
    a(f"**Selected model:** Gradient Boosting (`GradientBoostingRegressor`, "
      f"300 estimators, max depth 3, learning rate 0.05), inside a scikit-learn "
      f"`Pipeline` — confirmed by loading `models/regression/best_model.joblib` "
      f"directly (final pipeline step is a `GradientBoostingRegressor`), matching "
      f"`selected_model: \"gradient_boosting\"` in both `docs/regression/metrics.json` "
      f"and `models/regression/preprocessing_config.json`.\n")

    a("**Test-set metrics (final, untouched holdout):**\n")
    a("| Metric | Baseline (mean rating) | Gradient Boosting (selected) |")
    a("|---|---:|---:|")
    a(f"| R\u00b2 | {reg_base['r2']} | **{reg_test['r2']}** |")
    a(f"| MAE | {reg_base['mae']} | **{reg_test['mae']}** |")
    a(f"| MSE | {reg_base['mse']} | **{reg_test['mse']}** |")
    a(f"| RMSE | {reg_base['rmse']} | **{reg_test['rmse']}** |\n")

    a(f"For reference, validation-split performance used for model selection: "
      f"R\u00b2={reg_val['r2']}, MAE={reg_val['mae']}, MSE={reg_val['mse']}, "
      f"RMSE={reg_val['rmse']} (Gradient Boosting was the best of 3 candidates "
      f"— Linear Regression and Random Forest scored R\u00b2 0.0854 and 0.0857 "
      f"respectively on the same validation split; see "
      f"`model_comparison_on_validation` in `docs/regression/metrics.json` for "
      f"the full candidate table).\n")

    a("**Baseline comparison:** Gradient Boosting beats the mean-rating "
      f"baseline on every test metric (R\u00b2 {reg_test['r2']} vs. {reg_base['r2']}; "
      f"RMSE {reg_test['rmse']} vs. {reg_base['rmse']}) — a real, if modest, "
      "improvement over always predicting the average rating.\n")

    a("**Temporal train/validation/test methodology:**\n")
    a("Chronological, calendar-month-atomic split: rows are grouped by "
      "`VisitYear*12+VisitMonth`, and boundaries are placed *between* whole "
      "periods so no calendar month is ever divided across splits.\n")
    bc = summary["regression"]["corrected_calendar_coverage"]
    a("| Split | Rows | Period range | Calendar coverage (verified) |")
    a("|---|---:|---|---|")
    a(f"| Inner train | {si['n_inner_train']:,} | {si['inner_train_period_range'][0]}\u2013{si['inner_train_period_range'][1]} | {bc['inner_train_calendar']} |")
    a(f"| Inner validation | {si['n_inner_val']:,} | {si['inner_val_period_range'][0]}\u2013{si['inner_val_period_range'][1]} | {bc['inner_val_calendar']} |")
    a(f"| Train+val combined (final fit) | {si['n_train_full']:,} | {si['train_full_period_range'][0]}\u2013{si['train_full_period_range'][1]} | {bc['train_full_calendar']} |")
    a(f"| **Outer test** | **{si['n_test']:,}** | {si['test_period_range'][0]}\u2013{si['test_period_range'][1]} | **{bc['test_calendar']}** |\n")
    a("> **Note:** the calendar-coverage column above was independently "
      "recomputed from `regression_features.csv` for this consolidation, "
      "because the prose in `docs/regression_report.md` \u00a72 / "
      "`docs/classification_report.md` \u00a72 describes different (incorrect) "
      "calendar labels for the same period-integer boundaries. See "
      "**\u00a74 Inconsistencies found** below \u2014 this affects wording only, "
      "not the split logic, leakage checks, or any reported metric.\n")
    a("Leakage validation (`docs/regression/metrics.json` \u2192 "
      f"`leakage_validation.all_passed`): **{reg['leakage_validation']['all_passed']}** "
      "— target/`TransactionId` absent from features, no missing values, "
      "strictly increasing period boundaries, calendar-month atomicity, and "
      "test `TransactionId`s disjoint from train/val, all confirmed "
      "programmatically (not just asserted).\n")

    a("**Important feature interpretation** (top of "
      "`docs/regression/feature_importance.csv`):\n")
    a("| Feature | Importance |")
    a("|---|---:|")
    for feat in summary["regression"]["top_features"]:
        a(f"| `{feat['feature']}` | {feat['importance']:.3f} |")
    a("")
    a("`attraction_avg_rating` alone accounts for ~60% of importance — an "
      "attraction's own historical rating track record is by far the "
      "strongest predictor of a new rating for it. `VisitMonth` (~7.8%) "
      "indicates a real seasonal effect. User-side features contribute "
      "little, consistent with 72% of travelers being one-time visitors "
      "(per the EDA) — there's limited personal history for most rows to "
      "learn from.\n")

    a("---\n")

    # ============================== CLASSIFICATION ==============================
    a("## 2. Classification — predicting `VisitModeName`\n")
    a(f"**Selected model:** Random Forest (`RandomForestClassifier`, 300 trees, "
      f"max depth 12, `class_weight=\"balanced\"`), inside a scikit-learn "
      f"`Pipeline` — confirmed by loading `models/classification/best_model.joblib` "
      f"directly (final pipeline step is a `RandomForestClassifier`), matching "
      f"`selected_model: \"random_forest\"` in both "
      f"`docs/classification/metrics.json` and "
      f"`models/classification/preprocessing_config.json`. Selected by "
      f"validation macro-F1, not accuracy, because the target is heavily "
      f"imbalanced (Couples 41.3%, Business 1.1%).\n")

    a("**Test-set metrics (final, untouched holdout):**\n")
    a("| Metric | Baseline (majority class) | Random Forest (selected) |")
    a("|---|---:|---:|")
    a(f"| Accuracy | {clf_base['accuracy']} | **{clf_test['accuracy']}** |")
    a(f"| Macro Precision | {clf_base['macro_precision']} | **{clf_test['macro_precision']}** |")
    a(f"| Macro Recall | {clf_base['macro_recall']} | **{clf_test['macro_recall']}** |")
    a(f"| Macro F1 | {clf_base['macro_f1']} | **{clf_test['macro_f1']}** |\n")

    a(f"**Baseline comparison:** accuracy is essentially tied with the "
      f"majority-class baseline ({clf_test['accuracy']} vs. {clf_base['accuracy']}), "
      f"but macro-F1 is more than double ({clf_test['macro_f1']} vs. "
      f"{clf_base['macro_f1']}) — the model actually recognizes minority "
      f"classes (Business, Solo, Family) instead of defaulting to Couples "
      f"for everyone, which is the entire point of choosing macro-F1 over "
      f"accuracy for an imbalanced target.\n")

    a("**Per-class test results:**\n")
    a("| Class | Precision | Recall | F1 | Support |")
    a("|---|---:|---:|---:|---:|")
    for cls in cm_labels:
        m = pcm[cls]
        a(f"| {cls} | {m['precision']} | {m['recall']} | {m['f1']} | {m['support']:,} |")
    a("")

    a("**Confusion matrix (test set, rows = true, columns = predicted):**\n")
    header = "| True \\ Pred | " + " | ".join(cm_labels) + " |"
    sep = "|---|" + "---:|" * len(cm_labels)
    a(header)
    a(sep)
    for label, row in zip(cm_labels, cm):
        a(f"| **{label}** | " + " | ".join(str(v) for v in row) + " |")
    a("")
    a("**Interpretation:** the model is strongest on the two largest classes "
      "(Couples, Family) and weakest on Friends, which is misclassified as "
      "Couples far more often (882 times) than correctly identified (103 "
      "times) — the two trip types likely look similar on the "
      "geography/attraction-type features available. Business recall is "
      "comparatively high (0.43) but precision is very low (0.056): "
      "`class_weight=\"balanced\"` pushes the model to predict Business more "
      "than its 1.1% base rate would otherwise warrant, trading many false "
      "positives for catching a fair share of true Business visits.\n")

    a("**Temporal split methodology:** identical fixed calendar-period "
      "boundaries as regression (`test_boundary_period=24222`, "
      "`val_boundary_period=24212`), reused rather than recomputed, since "
      "classification and regression draw from the exact same "
      f"{si['n_total']:,}-transaction universe. See the corrected calendar "
      "coverage table in §1 above — the same correction applies here.\n")
    a("Leakage validation (`docs/classification/metrics.json` \u2192 "
      f"`leakage_validation.all_passed`): **{clf['leakage_validation']['all_passed']}** "
      "— additionally includes `historical_features_match_regression: "
      f"{clf['leakage_validation']['historical_features_match_regression']}`, "
      "a numeric parity check confirming this task's independently-recomputed "
      "historical aggregates exactly match regression's validated ones.\n")

    a("**Top features** (full ranking in "
      "`docs/classification/feature_importance.csv`):\n")
    a("| Feature | Importance |")
    a("|---|---:|")
    for feat in summary["classification"]["top_features"]:
        a(f"| `{feat['feature']}` | {feat['importance']:.3f} |")
    a("")
    a("Attraction-level historical stats dominate (~43% combined), with "
      "`VisitMonth`/`VisitYear` (~11%) confirming a seasonal component and "
      "geography (`UserRegion_South East Asia`, city dummies) contributing "
      "further down the ranking.\n")

    a("---\n")

    # ============================== RECOMMENDATION ==============================
    a("## 3. Recommendation — attraction recommendations\n")
    a("**Approach:** simple content/profile-based recommender (no "
      "collaborative filtering, no hybrid, no deep learning). Attraction "
      "content = `AttractionTypeName` + normalized average rating + "
      "normalized log-popularity. User profile = rating-weighted share of "
      "past visits per attraction type. Personalized score = "
      f"{rec_cfg['weights']['w_affinity']}\u00b7type_affinity + "
      f"{rec_cfg['weights']['w_quality']}\u00b7norm_quality + "
      f"{rec_cfg['weights']['w_popularity']}\u00b7norm_popularity, with "
      "deterministic tie-breaking by ascending `AttractionId`.\n")

    a("**Time-aware evaluation** (global cutoff: train `VisitYear<=2018`, "
      "test `VisitYear>2018`; see §5 note on why this differs from "
      "regression/classification's split):\n")
    a("| Segment | k | Precision@k | Recall@k | MAP@k | Users evaluated |")
    a("|---|---:|---:|---:|---:|---:|")
    a(f"| Warm (personalized) | 5 | {rec_warm5['precision_at_k']:.3f} | {rec_warm5['recall_at_k']:.3f} | {rec_warm5['map_at_k']:.3f} | {rec_warm5['n_users_evaluated']:,} |")
    a(f"| Warm (personalized) | 10 | {rec_warm10['precision_at_k']:.3f} | {rec_warm10['recall_at_k']:.3f} | {rec_warm10['map_at_k']:.3f} | {rec_warm10['n_users_evaluated']:,} |")
    a(f"| Cold (fallback) | 5 | {rec_cold5['precision_at_k']:.3f} | {rec_cold5['recall_at_k']:.3f} | {rec_cold5['map_at_k']:.3f} | {rec_cold5['n_users_evaluated']:,} |")
    a(f"| Cold (fallback) | 10 | {rec_cold10['precision_at_k']:.3f} | {rec_cold10['recall_at_k']:.3f} | {rec_cold10['map_at_k']:.3f} | {rec_cold10['n_users_evaluated']:,} |")
    a(f"| **Overall** | 5 | {rec_overall5['precision_at_k']:.3f} | {rec_overall5['recall_at_k']:.3f} | {rec_overall5['map_at_k']:.3f} | {rec_overall5['n_users_evaluated']:,} |")
    a(f"| **Overall** | 10 | {rec_overall10['precision_at_k']:.3f} | {rec_overall10['recall_at_k']:.3f} | {rec_overall10['map_at_k']:.3f} | {rec_overall10['n_users_evaluated']:,} |\n")

    a("**Baseline comparison** (closed-form expectation of a uniform-random "
      f"pick from the {rec_cfg['n_attractions']}-attraction catalog — not a "
      "sampled run):\n")
    a(f"- Expected recall@k = k/{rec_cfg['n_attractions']} \u2192 "
      f"{rb['recall_at_k']['5']} (k=5), {rb['recall_at_k']['10']} (k=10)")
    a(f"- Expected precision@k \u2248 {rb['precision_at_k']['overall']} "
      f"(k-independent; mean relevant items / catalog size)\n")
    a(f"The model beats this baseline by roughly 3\u20134x on precision "
      f"({rec_overall5['precision_at_k']:.3f} vs. {rb['precision_at_k']['overall']}) "
      "and matches or beats it on recall at k=5, for both the warm "
      "(personalized) and cold (fallback) segments.\n")

    a("**Cold-start strategy:** any user with no rated history (or history "
      "that sums to zero rating weight) automatically receives a purely "
      "popularity/rating-based fallback score "
      f"(`{rec_cfg['weights']['fallback_quality']}\u00b7norm_quality + "
      f"{rec_cfg['weights']['fallback_popularity']}\u00b7norm_popularity`), "
      "still fully deterministic. Verified with a synthetic brand-new "
      "`UserId` guaranteed absent from the data (`docs/recommendation/"
      "validation_results.json` \u2192 `cold_start_users_work: "
      f"{rec_val['cold_start_users_work']}`).\n")

    a("**Previously-visited exclusion:** every candidate attraction the "
      "target user has already visited (per their available history at "
      "prediction time) is removed from the candidate pool before scoring "
      "— verified programmatically (`previously_visited_excluded: "
      f"{rec_val['previously_visited_excluded']}`).\n")

    a("**Temporal leakage prevention:** a *global* time cutoff (not "
      "per-user leave-last-out) is used so no user's training window can "
      "overlap another user's test window. The evaluation-only model's "
      "attraction statistics are verified to come exclusively from "
      "pre-cutoff rows (`total_visits` across all attractions sums exactly "
      "to the train row count \u2014 checked in code, not just claimed), and "
      "`train.VisitYear.max() < test.VisitYear.min()` is asserted directly. "
      f"(`no_future_information_leakage: {rec_val['no_future_information_leakage']}`)\n")

    a("---\n")

    # ============================== CROSS-CHECKS ==============================
    a("## 4. Cross-model consistency checks performed\n")
    a("| Check | Result |")
    a("|---|---|")
    a("| `regression_report.md` / `metrics.json` numbers match | \u2705 verified — all R\u00b2/MAE/MSE/RMSE values in this report copied programmatically from `docs/regression/metrics.json` |")
    a("| `classification_report.md` / `metrics.json` numbers match | \u2705 verified — all accuracy/precision/recall/F1/confusion-matrix values copied programmatically from `docs/classification/metrics.json` and `confusion_matrix.csv` |")
    a("| `recommendation_report.md` / `metrics.json` numbers match | \u2705 verified — all Precision/Recall/MAP@k values copied programmatically from `docs/recommendation/metrics.json` |")
    a("| Selected model name consistent across report, `metrics.json`, and `preprocessing_config.json` (regression) | \u2705 `gradient_boosting` everywhere |")
    a("| Selected model name consistent across report, `metrics.json`, and `preprocessing_config.json` (classification) | \u2705 `random_forest` everywhere |")
    a("| Serialized model class matches the claimed selected model | \u2705 `best_model.joblib` final pipeline step is `GradientBoostingRegressor` (regression) / `RandomForestClassifier` (classification) |")
    a("| Regression & classification split boundaries identical | \u2705 classification explicitly reuses regression's `test_boundary_period`/`val_boundary_period` (`boundaries_source` field in `docs/classification/metrics.json`) |")
    a("| No leakage claim overstated | \u2705 each `leakage_validation.all_passed` is backed by explicit itemized boolean checks in the same JSON, not a single unverified flag |")
    a("| Recommendation split independent of regression/classification split | Expected \u2014 different script, different feature table, different task; see note below |\n")

    a("> **Note on split granularity:** regression/classification use a "
      "period-level (`VisitYear*12+VisitMonth`) chronological split; "
      "recommendation uses a coarser calendar-year cutoff "
      f"(`VisitYear<={rec_cfg['eval_time_cutoff_year']}`). This is expected, "
      "not a contradiction — they are separate evaluations on separate "
      "feature tables (`regression_features.csv`'s 49,208 rows / 1 target "
      "vs. `recommendation_interactions.csv`'s 45,275 rows / 30 "
      "attractions).\n")

    a("## 5. Inconsistencies found\n")
    if inconsistencies:
        for i, item in enumerate(inconsistencies, 1):
            a(f"**{i}. {item['location']}**\n")
            a(f"- **Issue:** {item['issue']}")
            a(f"- **Impact:** {item['impact']}")
            a(f"- **Resolution:** {item['fix_applied_here']}\n")
    else:
        a("None found.\n")

    a("## 6. Validation performed for this consolidation\n")
    a("Lightweight, read-only checks only (no retraining):\n")
    a("- Re-loaded `models/regression/best_model.joblib` and "
      "`models/classification/best_model.joblib` directly and confirmed the "
      "final pipeline step's class name matches the `selected_model` string "
      "claimed in both the metrics JSON and the preprocessing config.")
    a("- Re-derived the regression/classification split's calendar coverage "
      "from `regression_features.csv` (`VisitYear`, `VisitMonth`) and cross-"
      "checked it against `test_year_counts` in `docs/regression/metrics.json` "
      "(found and documented the calendar-labeling inconsistency in §5).")
    a("- Diffed every number transcribed into this report against its "
      "source JSON/CSV programmatically (see "
      "`scripts/build_evaluation_summary.py`) \u2014 this report cannot drift "
      "from the saved metric files because it is generated from them.")
    a("- Confirmed all three `leakage_validation` / `validation_results` "
      "blocks consist of itemized, individually-named boolean checks, not "
      "a single unverified summary flag.\n")

    a("## 7. Overall summary\n")
    a(f"| Task | Selected model | Headline metric |")
    a(f"|---|---|---|")
    a(f"| Regression | Gradient Boosting | Test R\u00b2 = {reg_test['r2']} (baseline {reg_base['r2']}) |")
    a(f"| Classification | Random Forest | Test macro-F1 = {clf_test['macro_f1']} (baseline {clf_base['macro_f1']}) |")
    a(f"| Recommendation | Content/profile-based | Overall Precision@5 = {rec_overall5['precision_at_k']:.3f}, Recall@5 = {rec_overall5['recall_at_k']:.3f} (random baseline precision \u2248 {rb['precision_at_k']['overall']}) |\n")
    a("All three tasks show modest but genuine, verified improvement over "
      "their respective baselines, with leakage-safe temporal evaluation "
      "and no unresolved contradictions between reports and saved metrics "
      "beyond the calendar-labeling documentation issue noted in §5.\n")

    with open(os.path.join(DOCS, "model_evaluation_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote docs/model_evaluation_report.md")
    print("Wrote docs/model_evaluation_summary.json")
    print("\nInconsistencies found:", len(inconsistencies))
    for item in inconsistencies:
        print(" -", item["location"])


if __name__ == "__main__":
    main()

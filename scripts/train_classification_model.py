"""
Classification model — predict VisitMode (Checkpoint 06)

Scope: CLASSIFICATION ONLY. Trains and evaluates models to predict
`VisitModeName` (Business / Couples / Family / Friends / Solo).

--------------------------------------------------------------------------
AUDIT FINDING (this checkpoint) — read before trusting any prior summary
of "current state" for classification:
--------------------------------------------------------------------------
Verification against the delivered `data/processed/classification_features.csv`
and `scripts/feature_engineering.py` found TWO real discrepancies versus
what had been assumed complete:

  1. `Rating` IS present as a column in `classification_features.csv`,
     and `build_classification_features()` in `feature_engineering.py`
     deliberately includes it as an input feature (see that function's
     docstring). It is NOT "confirmed absent" in the delivered file.

  2. The historical aggregates baked into `classification_features.csv`
     (`user_avg_rating`, `user_rating_std`, `user_distinct_attractions`,
     `attraction_avg_rating`, `attraction_rating_std`,
     `attraction_distinct_users`) come from `add_user_aggregates()` /
     `add_attraction_aggregates()` — full-history, NON-temporal
     aggregates computed over the ENTIRE dataset (the same class of bug
     that was found and fixed for regression in checkpoint 05 — see the
     "LEAKAGE FIX" comment block at the top of `feature_engineering.py`).
     Used as-is with a chronological split, these would leak each
     transaction's own future ratings into its own features. They do
     NOT match regression's leakage-safe temporal aggregates, and the
     two required flag columns (`has_user_history`, `has_attraction_history`)
     are missing entirely from `classification_features.csv`.

Neither of these is a "rebuild for no reason" — per the task brief,
"do not rebuild the classification pipeline unless verification finds
an actual error", and this is exactly that: an unresolved leakage risk,
of the same kind already fixed for regression.

FIX APPLIED (additive only — `classification_features.csv` and
`feature_engineering.py` are left completely untouched on disk):
This script independently recomputes the classification feature set
from `data/processed/consolidated.csv`, reusing the SAME leakage-safe
temporal helper functions already used by the regression pipeline
(`add_temporal_user_history` / `add_temporal_attraction_history` from
`feature_engineering.py`, imported read-only) so the historical
features used here are byte-for-byte the same methodology as
`regression_features.csv` — verified explicitly in
`validate_no_leakage()` below (`historical_features_match_regression`).

`Rating` is EXCLUDED from the feature set in this script (see
EXCLUDED_COLUMNS below) — a deliberate choice made and documented here,
not inherited from the delivered file, on the grounds that visit mode
is a property of the visit itself (planned/known before or at the time
of visiting), while `Rating` is typically given afterward; predicting
"who visited" from "how they rated it" mixes cause and effect for the
intended use case (the frontend's Visit Mode Predictor takes trip
context, not a rating, as input).

Train/validation/test use the IDENTICAL fixed calendar-month period
boundaries already established for regression (`test_boundary_period =
24222`, `val_boundary_period = 24212`, from
`docs/regression/metrics.json` -> `split_info`), not a freshly computed
boundary — this was verified to be safe: classification and regression
draw from the exact same 49,208-transaction universe, so the same
period boundaries partition both tables identically.

Run:
    python scripts/train_classification_model.py
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONSOLIDATED_PATH = os.path.join(BASE_DIR, "data", "processed", "consolidated.csv")
REGRESSION_FEATURES_PATH = os.path.join(BASE_DIR, "data", "processed", "regression_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "classification")
DOC_DIR = os.path.join(BASE_DIR, "docs", "classification")

TARGET = "VisitModeName"
RANDOM_STATE = 42

# Fixed period boundaries — reused from the already-validated regression
# split (docs/regression/metrics.json -> split_info), not recomputed.
TEST_BOUNDARY_PERIOD = 24222
VAL_BOUNDARY_PERIOD = 24212

# Same exclusion rationale as regression (see train_regression_model.py),
# plus Rating (see module docstring for why it's excluded here).
ID_COLUMNS = [
    "TransactionId", "UserId", "AttractionId", "ContinentId", "RegionId",
    "CountryId", "CityId", "AttractionTypeId", "AttractionCityId",
]
DROPPED_HIGH_CARDINALITY = ["UserCountry", "UserCityName"]
DROPPED_CONSTANT = ["AttractionCountry"]
EXCLUDED_TARGET_LEAKAGE = ["Rating"]

NUMERIC_FEATURES = [
    "VisitYear",
    "VisitMonth",
    "user_total_visits",
    "user_avg_rating",
    "user_rating_std",
    "user_distinct_attractions",
    "has_user_history",
    "attraction_total_visits",
    "attraction_avg_rating",
    "attraction_rating_std",
    "attraction_distinct_users",
    "has_attraction_history",
    "is_ambiguous_repeat",
]
CATEGORICAL_FEATURES = [
    "UserContinent",
    "UserRegion",
    "AttractionTypeName",
    "AttractionCityName",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _import_feature_engineering():
    """Read-only import of feature_engineering.py's temporal helper
    functions, without executing its __main__ block or writing anything."""
    fe_path = os.path.join(BASE_DIR, "scripts", "feature_engineering.py")
    spec = importlib.util.spec_from_file_location("feature_engineering", fe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# Data loading — rebuild leakage-safe classification features from scratch
# --------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    fe = _import_feature_engineering()
    consolidated = pd.read_csv(CONSOLIDATED_PATH)
    consolidated["is_ambiguous_repeat"] = consolidated["is_ambiguous_repeat"].astype(int)

    df = consolidated.copy()
    df["_period"] = df["VisitYear"] * 12 + df["VisitMonth"]

    user_hist = fe.add_temporal_user_history(consolidated)
    attr_hist = fe.add_temporal_attraction_history(consolidated)

    df = df.merge(user_hist, on=["UserId", "_period"], how="left")
    df = df.merge(attr_hist, on=["AttractionId", "_period"], how="left")

    keep_cols = [
        "TransactionId", "UserId", "AttractionId", "VisitYear", "VisitMonth", "_period",
        "ContinentId", "UserContinent", "RegionId", "UserRegion",
        "CountryId", "UserCountry", "CityId", "UserCityName",
        "AttractionTypeId", "AttractionTypeName",
        "AttractionCityId", "AttractionCityName", "AttractionCountry",
        "user_total_visits", "user_avg_rating", "user_rating_std", "user_distinct_attractions", "has_user_history",
        "attraction_total_visits", "attraction_avg_rating", "attraction_rating_std", "attraction_distinct_users", "has_attraction_history",
        "is_ambiguous_repeat", "Rating", "VisitMode", "VisitModeName",
    ]
    return df[keep_cols]


def chronological_split_fixed(df: pd.DataFrame):
    """Same calendar-month-atomic chronological split as regression, but
    using regression's ALREADY-CHOSEN fixed period boundaries rather than
    recomputing them, per the task brief."""
    d = df.copy()

    train_full = d[d["_period"] < TEST_BOUNDARY_PERIOD].reset_index(drop=True)
    test = d[d["_period"] >= TEST_BOUNDARY_PERIOD].reset_index(drop=True)

    inner_train = train_full[train_full["_period"] < VAL_BOUNDARY_PERIOD].reset_index(drop=True)
    inner_val = train_full[train_full["_period"] >= VAL_BOUNDARY_PERIOD].reset_index(drop=True)

    split_info = {
        "n_total": len(d),
        "n_inner_train": len(inner_train),
        "n_inner_val": len(inner_val),
        "n_train_full": len(train_full),
        "n_test": len(test),
        "test_boundary_period": TEST_BOUNDARY_PERIOD,
        "val_boundary_period": VAL_BOUNDARY_PERIOD,
        "inner_train_period_range": [int(inner_train["_period"].min()), int(inner_train["_period"].max())],
        "inner_val_period_range": [int(inner_val["_period"].min()), int(inner_val["_period"].max())],
        "train_full_period_range": [int(train_full["_period"].min()), int(train_full["_period"].max())],
        "test_period_range": [int(test["_period"].min()), int(test["_period"].max())],
        "boundaries_source": "reused from docs/regression/metrics.json split_info (not recomputed)",
    }
    return inner_train, inner_val, train_full, test, split_info


# --------------------------------------------------------------------------
# Preprocessing + models
# --------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=5,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=RANDOM_STATE,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_recall": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def per_class_report(y_true, y_pred, labels) -> dict:
    precisions = precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    recalls = recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    f1s = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(labels).fillna(0).astype(int)
    return {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(support[label]),
        }
        for label, p, r, f in zip(labels, precisions, recalls, f1s)
    }


# --------------------------------------------------------------------------
# Leakage validation
# --------------------------------------------------------------------------

def validate_no_leakage(df: pd.DataFrame, inner_train, inner_val, test) -> dict:
    train_periods = set(inner_train["_period"])
    val_periods = set(inner_val["_period"])
    test_periods = set(test["_period"])

    # Parity check: this script's independently-recomputed temporal
    # historical features must exactly match regression_features.csv's,
    # per-TransactionId, proving the "matches regression 100%" claim
    # rather than just asserting it.
    reg = pd.read_csv(REGRESSION_FEATURES_PATH)
    compare_cols = [
        "user_total_visits", "user_avg_rating", "user_rating_std", "user_distinct_attractions", "has_user_history",
        "attraction_total_visits", "attraction_avg_rating", "attraction_rating_std", "attraction_distinct_users", "has_attraction_history",
    ]
    merged = df[["TransactionId"] + compare_cols].merge(
        reg[["TransactionId"] + compare_cols], on="TransactionId", suffixes=("_clf", "_reg")
    )
    historical_match = all(
        np.allclose(merged[f"{c}_clf"], merged[f"{c}_reg"], equal_nan=True) for c in compare_cols
    )

    checks = {
        "rating_not_in_feature_columns": "Rating" not in FEATURE_COLUMNS,
        "target_not_in_feature_columns": TARGET not in FEATURE_COLUMNS,
        "transaction_id_not_in_feature_columns": "TransactionId" not in FEATURE_COLUMNS,
        "no_missing_values_in_features": not df[FEATURE_COLUMNS].isna().any().any(),
        "train_periods_strictly_precede_val_periods": max(train_periods) < min(val_periods),
        "train_val_periods_strictly_precede_test_periods": max(train_periods | val_periods) < min(test_periods),
        "each_calendar_period_in_exactly_one_split": (
            len(train_periods & val_periods) == 0
            and len(train_periods & test_periods) == 0
            and len(val_periods & test_periods) == 0
        ),
        "test_transaction_ids_disjoint_from_train_val": len(
            set(test["TransactionId"]) & (set(inner_train["TransactionId"]) | set(inner_val["TransactionId"]))
        ) == 0,
        "historical_features_match_regression": historical_match,
        "row_count_matches_regression": len(df) == len(reg),
    }
    checks["all_passed"] = all(bool(v) for v in checks.values())
    return checks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(DOC_DIR, exist_ok=True)

    df = load_data()
    inner_train, inner_val, train_full, test, split_info = chronological_split_fixed(df)
    leakage_checks = validate_no_leakage(df, inner_train, inner_val, test)

    labels = sorted(df[TARGET].unique().tolist())

    X_inner_train, y_inner_train = inner_train[FEATURE_COLUMNS], inner_train[TARGET]
    X_inner_val, y_inner_val = inner_val[FEATURE_COLUMNS], inner_val[TARGET]
    X_train_full, y_train_full = train_full[FEATURE_COLUMNS], train_full[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    # --- baseline: always predict the (inner-)train majority class ---
    baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    baseline.fit(X_inner_train, y_inner_train)
    baseline_val_metrics = evaluate(y_inner_val, baseline.predict(X_inner_val))

    # --- candidate models, selected on inner validation macro-F1 ---
    model_results = {"baseline_most_frequent": {"val": baseline_val_metrics}}
    for name, estimator in candidate_models().items():
        pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_inner_train, y_inner_train)
        val_metrics = evaluate(y_inner_val, pipe.predict(X_inner_val))
        model_results[name] = {"val": val_metrics}
        print(f"[{name}] validation: {val_metrics}")

    best_name = max(
        (n for n in model_results if n != "baseline_most_frequent"),
        key=lambda n: model_results[n]["val"]["macro_f1"],
    )
    print(f"Selected model: {best_name}")

    # --- retrain selected model type on train+val, evaluate once on test ---
    final_pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", candidate_models()[best_name])])
    final_pipe.fit(X_train_full, y_train_full)
    test_preds = final_pipe.predict(X_test)
    test_metrics = evaluate(y_test, test_preds)
    train_full_metrics = evaluate(y_train_full, final_pipe.predict(X_train_full))

    final_baseline = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
    final_baseline.fit(X_train_full, y_train_full)
    baseline_test_metrics = evaluate(y_test, final_baseline.predict(X_test))

    cm = confusion_matrix(y_test, test_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels], columns=[f"pred_{l}" for l in labels])
    cm_df.to_csv(os.path.join(DOC_DIR, "confusion_matrix.csv"))

    per_class = per_class_report(y_test, test_preds, labels)

    model_results["selected_model"] = best_name
    model_results["final"] = {
        "baseline_most_frequent_test": baseline_test_metrics,
        f"{best_name}_train_full": train_full_metrics,
        f"{best_name}_test": test_metrics,
    }

    # --- feature importance (tree-based models only) ---
    feature_importance = None
    if best_name in ("random_forest", "gradient_boosting"):
        ohe = final_pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERIC_FEATURES + cat_feature_names
        importances = final_pipe.named_steps["model"].feature_importances_
        fi_df = (
            pd.DataFrame({"feature": all_feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        fi_df.to_csv(os.path.join(DOC_DIR, "feature_importance.csv"), index=False)
        feature_importance = fi_df.head(20).to_dict(orient="records")
    elif best_name == "logistic_regression":
        # Coefficients aren't directly comparable to tree importances, but
        # save them for transparency since this model has no .feature_importances_.
        ohe = final_pipe.named_steps["preprocess"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERIC_FEATURES + cat_feature_names
        coef_df = pd.DataFrame(
            final_pipe.named_steps["model"].coef_, columns=all_feature_names, index=labels
        ).T
        coef_df.to_csv(os.path.join(DOC_DIR, "feature_importance.csv"))
        feature_importance = "logistic_regression coefficients saved (per-class), not a single importance ranking"

    # --- save model + inference config ---
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(final_pipe, model_path)

    inference_config = {
        "target": TARGET,
        "feature_columns_in_order": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "class_labels": labels,
        "excluded_columns": {
            "id_columns": ID_COLUMNS,
            "dropped_high_cardinality": DROPPED_HIGH_CARDINALITY,
            "dropped_constant": DROPPED_CONSTANT,
            "target_leakage_risk": EXCLUDED_TARGET_LEAKAGE,
        },
        "selected_model": best_name,
        "notes": (
            "Pass a DataFrame containing exactly `feature_columns_in_order` "
            "to model.predict(). Preprocessing (StandardScaler on numeric, "
            "OneHotEncoder(handle_unknown='ignore') on categorical) is "
            "already embedded as the first step of the saved sklearn "
            "Pipeline in best_model.joblib. Historical aggregate columns "
            "(user_*/attraction_*/has_*_history) must be computed the same "
            "leakage-safe temporal way as in this script -- do not source "
            "them from classification_features.csv directly, since that "
            "file's versions are non-temporal (full-history) and unsafe "
            "for point-in-time inference."
        ),
    }
    with open(os.path.join(MODEL_DIR, "preprocessing_config.json"), "w") as f:
        json.dump(inference_config, f, indent=2)

    # --- reload check ---
    reloaded = joblib.load(model_path)
    reload_preds = reloaded.predict(X_test)
    reload_check = bool((reload_preds == final_pipe.predict(X_test)).all())

    summary = {
        "split_info": split_info,
        "leakage_validation": leakage_checks,
        "class_distribution_full_dataset": df[TARGET].value_counts().to_dict(),
        "model_comparison_on_validation": model_results,
        "selected_model": best_name,
        "confusion_matrix_labels": labels,
        "confusion_matrix_test": cm.tolist(),
        "per_class_test_metrics": per_class,
        "feature_importance_top20": feature_importance if isinstance(feature_importance, list) else str(feature_importance),
        "reload_check_passed": reload_check,
    }
    with open(os.path.join(DOC_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Leakage validation passed: {leakage_checks['all_passed']}")
    print(f"Final test metrics ({best_name}): {test_metrics}")
    print(f"Baseline test metrics: {baseline_test_metrics}")
    print(f"Reload check passed: {reload_check}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()

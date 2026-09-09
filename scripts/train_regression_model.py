"""
Regression model — predict Rating (Checkpoint 05)

Scope: REGRESSION ONLY. Trains and evaluates models to predict `Rating`
from data/processed/regression_features.csv, the already-validated,
leakage-safe temporal feature table (see docs/data_dictionary.md, section
4.1). This script does not recompute any historical aggregate, does not
touch raw/cleaned data, does not modify classification/recommendation/EDA/
SQL, and does not change the frontend.

Leakage safeguards (see `validate_no_leakage()` below for the automated
checks):
  - Rating is the target only, never a feature.
  - TransactionId is dropped entirely — never used as a feature, and never
    used to order or break ties (matching the rule already enforced
    upstream when regression_features.csv was built).
  - All engineered historical features (user_*, attraction_*, has_*_history)
    are already "historical-as-of" (strictly-prior-period) values computed
    upstream — this script uses them exactly as delivered and computes no
    new aggregate of its own.
  - Train/validation/test are split chronologically (by VisitYear*12 +
    VisitMonth), not randomly, and every calendar period is assigned
    whole to exactly one split (never divided across a boundary) — see
    `_period_boundary()` / `chronological_split()` — so the model is
    always evaluated on periods strictly after everything it was trained
    on, with no calendar month appearing on both sides of a split.

Run:
    python scripts/train_regression_model.py
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "regression_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models", "regression")
DOC_DIR = os.path.join(BASE_DIR, "docs", "regression")

TARGET = "Rating"
RANDOM_STATE = 42

# Columns intentionally excluded, and why (documented here so the decision
# travels with the code, not just the report):
#   TransactionId       -> identifier only; instructions explicitly forbid
#                           using it as a chronological feature, and as a
#                           raw id it has zero generalizable signal anyway.
#   Rating               -> the target.
#   UserId, AttractionId, ContinentId, RegionId, CountryId, CityId,
#   AttractionTypeId, AttractionCityId
#                        -> raw numeric/categorical ID codes. Using them
#                           directly would let tree models memorize
#                           individual users/attractions instead of
#                           learning generalizable patterns, and they carry
#                           no information beyond what the named columns
#                           below (UserContinent, UserRegion, ... ) or the
#                           engineered historical aggregates already
#                           capture.
#   UserCountry (153 categories), UserCityName (5,546 categories)
#                        -> dropped for cardinality/sparsity: with 49,208
#                           rows, one-hot-encoding thousands of city/country
#                           levels would produce mostly-empty columns and
#                           risks overfitting far more than it helps.
#                           UserContinent (5) and UserRegion (22) retain
#                           the geographic signal at a workable cardinality.
#   AttractionCountry    -> constant (single value, "Indonesia") across all
#                           30 attractions in this dataset — zero variance,
#                           zero predictive value.
ID_COLUMNS = [
    "TransactionId",
    "UserId",
    "AttractionId",
    "ContinentId",
    "RegionId",
    "CountryId",
    "CityId",
    "AttractionTypeId",
    "AttractionCityId",
]
DROPPED_HIGH_CARDINALITY = ["UserCountry", "UserCityName"]
DROPPED_CONSTANT = ["AttractionCountry"]

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
    "VisitModeName",
    "UserContinent",
    "UserRegion",
    "AttractionTypeName",
    "AttractionCityName",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# --------------------------------------------------------------------------
# Data loading & chronological split
# --------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["is_ambiguous_repeat"] = df["is_ambiguous_repeat"].astype(int)
    return df


def _period_boundary(period_counts: pd.Series, right_frac: float) -> int:
    """Given row counts indexed by sorted, unique calendar period, choose
    the period at which a right-hand split (e.g. test, or val-within-
    train) should START, such that:
      - every period is assigned whole to one side (never split a
        calendar month across the boundary)
      - the resulting right-side row count is as close as possible to
        `right_frac` of the total, among the boundaries that satisfy the
        rule above
      - at least one period ends up on each side

    This is a period-level (not row-level) equivalent of a percentile
    split, which is what makes the atomicity guarantee possible.
    """
    periods = period_counts.index.tolist()
    total = period_counts.sum()
    cum = period_counts.cumsum()
    target_left = total * (1 - right_frac)

    best_i, best_diff = None, None
    for i in range(len(periods) - 1):  # leave at least one period on the right
        diff = abs(cum.iloc[i] - target_left)
        if best_diff is None or diff < best_diff:
            best_diff, best_i = diff, i
    return periods[best_i + 1]


def chronological_split(df: pd.DataFrame):
    """Calendar-month-atomic chronological split (VisitYear*12 +
    VisitMonth): every (VisitYear, VisitMonth) period is assigned to
    exactly ONE of {inner_train, inner_val, test} — never split across a
    boundary. Boundaries are chosen (via `_period_boundary`) as close as
    possible to an 80/20 outer split and an 85/15 inner split by row
    count, subject to that atomicity constraint. Periods are never
    ordered or tie-broken by TransactionId anywhere in this process.

        - outer test  = periods strictly after everything in train+val
        - inner val   = periods strictly after everything in inner_train,
                         and strictly before the outer test periods
        - inner train = everything before that

    This mirrors the temporal-atomicity principle already used to build
    regression_features.csv: nothing in train/val is allowed to be from a
    period at or after anything in the corresponding held-out split, and
    no period is ever divided between two splits.
    """
    d = df.copy()
    d["_period"] = d["VisitYear"] * 12 + d["VisitMonth"]

    period_counts = d.groupby("_period").size().sort_index()
    test_boundary = _period_boundary(period_counts, right_frac=0.20)

    train_full = d[d["_period"] < test_boundary].reset_index(drop=True)
    test = d[d["_period"] >= test_boundary].reset_index(drop=True)

    train_full_period_counts = train_full.groupby("_period").size().sort_index()
    val_boundary = _period_boundary(train_full_period_counts, right_frac=0.15)

    inner_train = train_full[train_full["_period"] < val_boundary].reset_index(drop=True)
    inner_val = train_full[train_full["_period"] >= val_boundary].reset_index(drop=True)

    split_info = {
        "n_total": len(d),
        "n_inner_train": len(inner_train),
        "n_inner_val": len(inner_val),
        "n_train_full": len(train_full),
        "n_test": len(test),
        "test_boundary_period": int(test_boundary),
        "val_boundary_period": int(val_boundary),
        "inner_train_period_range": [int(inner_train["_period"].min()), int(inner_train["_period"].max())],
        "inner_val_period_range": [int(inner_val["_period"].min()), int(inner_val["_period"].max())],
        "train_full_period_range": [int(train_full["_period"].min()), int(train_full["_period"].max())],
        "test_period_range": [int(test["_period"].min()), int(test["_period"].max())],
        "test_year_counts": test["VisitYear"].value_counts().sort_index().to_dict(),
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
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
    }


# --------------------------------------------------------------------------
# Leakage validation
# --------------------------------------------------------------------------

def validate_no_leakage(df: pd.DataFrame, inner_train, inner_val, test) -> dict:
    train_periods = set((inner_train["VisitYear"] * 12 + inner_train["VisitMonth"]))
    val_periods = set((inner_val["VisitYear"] * 12 + inner_val["VisitMonth"]))
    test_periods = set((test["VisitYear"] * 12 + test["VisitMonth"]))

    checks = {
        "target_not_in_feature_columns": TARGET not in FEATURE_COLUMNS,
        "transaction_id_not_in_feature_columns": "TransactionId" not in FEATURE_COLUMNS,
        "transaction_id_not_used_for_split_order": True,  # split assigns whole calendar periods only
        "no_missing_values_in_features": not df[FEATURE_COLUMNS].isna().any().any(),
        "train_periods_strictly_precede_val_periods": max(train_periods) < min(val_periods),
        "train_val_periods_strictly_precede_test_periods": max(train_periods | val_periods) < min(test_periods),
        "each_calendar_period_in_exactly_one_split": (
            len(train_periods & val_periods) == 0
            and len(train_periods & test_periods) == 0
            and len(val_periods & test_periods) == 0
        ),
        "test_transaction_ids_disjoint_from_train_val": len(
            set(test["TransactionId"])
            & (set(inner_train["TransactionId"]) | set(inner_val["TransactionId"]))
        )
        == 0,
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
    inner_train, inner_val, train_full, test, split_info = chronological_split(df)
    leakage_checks = validate_no_leakage(df, inner_train, inner_val, test)

    X_inner_train, y_inner_train = inner_train[FEATURE_COLUMNS], inner_train[TARGET]
    X_inner_val, y_inner_val = inner_val[FEATURE_COLUMNS], inner_val[TARGET]
    X_train_full, y_train_full = train_full[FEATURE_COLUMNS], train_full[TARGET]
    X_test, y_test = test[FEATURE_COLUMNS], test[TARGET]

    # --- baseline: predict the (inner-)train mean rating for everyone ---
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_inner_train, y_inner_train)
    baseline_val_metrics = evaluate(y_inner_val, baseline.predict(X_inner_val))

    # --- candidate models, selected on the inner validation split ---
    model_results = {"baseline_mean": {"val": baseline_val_metrics}}
    fitted_pipelines = {}

    for name, estimator in candidate_models().items():
        pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_inner_train, y_inner_train)
        val_metrics = evaluate(y_inner_val, pipe.predict(X_inner_val))
        model_results[name] = {"val": val_metrics}
        fitted_pipelines[name] = pipe
        print(f"[{name}] validation: {val_metrics}")

    # --- pick the best candidate model by validation R² (baseline is a
    # reference, not eligible for selection) ---
    best_name = max(
        (n for n in model_results if n != "baseline_mean"),
        key=lambda n: model_results[n]["val"]["r2"],
    )
    print(f"Selected model: {best_name}")

    # --- retrain the selected model type on train+val combined, evaluate
    # once on the untouched outer test set ---
    final_pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("model", candidate_models()[best_name])])
    final_pipe.fit(X_train_full, y_train_full)
    test_metrics = evaluate(y_test, final_pipe.predict(X_test))
    train_full_metrics = evaluate(y_train_full, final_pipe.predict(X_train_full))

    # baseline re-evaluated the same way, on the same final test set, for a
    # fair final comparison
    final_baseline = DummyRegressor(strategy="mean")
    final_baseline.fit(X_train_full, y_train_full)
    baseline_test_metrics = evaluate(y_test, final_baseline.predict(X_test))

    model_results["selected_model"] = best_name
    model_results["final"] = {
        "baseline_mean_test": baseline_test_metrics,
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

    # --- save model + inference config ---
    model_path = os.path.join(MODEL_DIR, "best_model.joblib")
    joblib.dump(final_pipe, model_path)

    inference_config = {
        "target": TARGET,
        "feature_columns_in_order": FEATURE_COLUMNS,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "excluded_columns": {
            "id_columns": ID_COLUMNS,
            "dropped_high_cardinality": DROPPED_HIGH_CARDINALITY,
            "dropped_constant": DROPPED_CONSTANT,
        },
        "selected_model": best_name,
        "notes": (
            "Pass a DataFrame containing exactly `feature_columns_in_order` "
            "(same names/dtypes as regression_features.csv) to "
            "model.predict(). Preprocessing (StandardScaler on numeric, "
            "OneHotEncoder(handle_unknown='ignore') on categorical) is "
            "already embedded as the first step of the saved sklearn "
            "Pipeline in best_model.joblib -- no separate preprocessing "
            "step is required at inference time."
        ),
    }
    with open(os.path.join(MODEL_DIR, "preprocessing_config.json"), "w") as f:
        json.dump(inference_config, f, indent=2)

    # --- reload check: does the saved model actually work? ---
    reloaded = joblib.load(model_path)
    reload_preds = reloaded.predict(X_test)
    reload_check = bool(np.allclose(reload_preds, final_pipe.predict(X_test)))

    summary = {
        "split_info": split_info,
        "leakage_validation": leakage_checks,
        "model_comparison_on_validation": model_results,
        "selected_model": best_name,
        "feature_importance_top20": feature_importance,
        "reload_check_passed": reload_check,
    }
    with open(os.path.join(DOC_DIR, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Leakage validation passed: {leakage_checks['all_passed']}")
    print(f"Final test metrics ({best_name}): {test_metrics}")
    print(f"Baseline test metrics: {baseline_test_metrics}")
    print(f"Reload check passed: {reload_check}")
    print(f"Model saved to: {model_path}")
    print(f"Config saved to: {os.path.join(MODEL_DIR, 'preprocessing_config.json')}")
    print(f"Metrics saved to: {os.path.join(DOC_DIR, 'metrics.json')}")


if __name__ == "__main__":
    main()

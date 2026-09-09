# Regression Model Report
**Tourism Experience Analytics — Checkpoint 05 (Regression only)**

Target: `Rating`. Source: `data/processed/regression_features.csv`
(49,208 rows), the already-validated, leakage-safe temporal feature table
(see `docs/data_dictionary.md` §4.1). No historical aggregate was
recalculated — every `user_*` / `attraction_*` / `has_*_history` column is
used exactly as delivered by the locked feature-engineering pipeline.

> **Audit fix (this version):** the original checkpoint's chronological
> split allowed the same calendar month to land on both sides of a split
> boundary (e.g. `inner_train` ending at period 24212 while `inner_val`
> also started at 24212). `chronological_split()` in
> `scripts/train_regression_model.py` was rewritten so every
> `(VisitYear, VisitMonth)` period is assigned whole to exactly one split
> — see **§2 "Train/validation/test split"** and **§5 Validation** below
> for the fix and its verification. Nothing else changed: same features,
> same `regression_features.csv`, same three candidate models, same
> hyperparameters, same metrics, same artifact/documentation structure.
> Metrics moved only slightly (e.g. selected-model test R² 0.0722 →
> 0.0724) since the corrected boundary shifts a handful of rows between
> splits.

---

## 1. Files changed

All new, purely additive, except one intentional one-line-per-package
addition to `requirements.txt` (new dependencies needed for this task),
plus the split-logic fix inside the one script from this checkpoint.
Nothing else was modified (verified with a recursive diff against
`tourism_checkpoint_04_sql.zip` for the original files, and against the
prior `tourism_checkpoint_05_regression.zip` for this audit fix — see
**Validation**).

| File | Purpose |
|---|---|
| `scripts/train_regression_model.py` | Full training/evaluation pipeline: calendar-month-atomic chronological split (fixed this version), baseline, 3 candidate models, model selection, final test evaluation, leakage checks, artifact saving. |
| `models/regression/best_model.joblib` | The selected, trained model — a single scikit-learn `Pipeline` (preprocessing + model) ready for `.predict()`. Regenerated with the corrected split. |
| `models/regression/preprocessing_config.json` | Exact feature list/order/types required for inference, and why each excluded column was excluded. Unchanged content. |
| `docs/regression/metrics.json` | Every metric, the leakage validation results, and the split details. Regenerated with the corrected split. |
| `docs/regression/feature_importance.csv` | Full feature importance ranking for the selected model. Regenerated with the corrected split. |
| `docs/regression_report.md` | This report, updated with the corrected split and results. |
| `requirements.txt` | Added `scikit-learn>=1.4` and `joblib>=1.3` (only lines added; nothing else touched; unchanged from the prior checkpoint). |

## 2. Methodology

**Features used (18 total, from `regression_features.csv` only):**

- Numeric (13): `VisitYear`, `VisitMonth`, `user_total_visits`,
  `user_avg_rating`, `user_rating_std`, `user_distinct_attractions`,
  `has_user_history`, `attraction_total_visits`, `attraction_avg_rating`,
  `attraction_rating_std`, `attraction_distinct_users`,
  `has_attraction_history`, `is_ambiguous_repeat`.
- Categorical, one-hot encoded (5): `VisitModeName`, `UserContinent`,
  `UserRegion`, `AttractionTypeName`, `AttractionCityName`.

**Excluded, and why** (also documented inline in the script):
- `TransactionId` — an identifier only; explicitly disallowed as a
  chronological feature by the task brief, and has no generalizable
  signal as a raw ID either.
- `UserId`, `AttractionId`, `ContinentId`, `RegionId`, `CountryId`,
  `CityId`, `AttractionTypeId`, `AttractionCityId` — raw ID codes. Using
  them directly risks memorizing specific users/attractions instead of
  learning generalizable patterns; the named columns (`UserContinent`,
  `AttractionTypeName`, ...) and the engineered historical aggregates
  already carry their useful signal.
- `UserCountry` (153 levels), `UserCityName` (5,546 levels) — dropped for
  cardinality: one-hot-encoding thousands of near-empty categories would
  add noise and overfitting risk far more than signal, on a 49k-row
  dataset. `UserContinent` (5) and `UserRegion` (22) retain the
  geographic signal at a workable resolution.
- `AttractionCountry` — constant (single value, "Indonesia") across all
  30 attractions in this dataset; zero variance, zero predictive value.
- `Rating` — the target; never used as an input.

**Train/validation/test split — chronological, and calendar-month-atomic:**
Rows are grouped by calendar period (`VisitYear*12 + VisitMonth`). A split
boundary is chosen only *between* periods — as close as possible to an
80/20 outer split and an 85/15 inner split by row count — so that **every
period is assigned whole to exactly one split**; no calendar month is
ever divided across a boundary, and `TransactionId` is never used to
order or tie-break anything, matching the same rule already enforced
when `regression_features.csv` itself was built (`docs/data_dictionary.md`
§4.1: *"TransactionId is never used to order or break ties"*).

| Split | Rows | Period range (`VisitYear*12+VisitMonth`) | Calendar coverage |
|---|---:|---|---|
| Inner train (model fitting) | 33,388 | 24157 – 24211 | 2013 – May 2018 |
| Inner validation (model selection) | 5,816 | 24212 – 24221 | Jun 2018 – Mar 2019 |
| Train+val combined (final fit) | 39,204 | 24157 – 24221 | 2013 – Mar 2019 |
| **Outer test (final, untouched holdout)** | **10,004** | 24222 – 24274 | Apr 2019 – 2022 |

Boundaries: `max(inner_train period) = 24211 < min(inner_val period) =
24212` and `max(train+val period) = 24221 < min(test period) = 24222` —
strictly increasing, with no period appearing on both sides (see §5 for
the automated checks).

The outer test set is a genuinely future, never-seen-during-selection
period — it includes the 2020–2021 COVID collapse and the 2022 partial
recovery, so the final metrics reflect the model's ability to generalize
to a materially different period than it was trained/selected on, not
just a random held-out sample from the same era.

**Models compared** (on the inner validation split): a mean-rating
baseline (`DummyRegressor`), `LinearRegression`, `RandomForestRegressor`
(300 trees, max depth 12), `GradientBoostingRegressor` (300 estimators,
max depth 3, learning rate 0.05). The best model by validation R² was
retrained on train+val combined and evaluated once on the outer test set.

## 3. Models tested & metrics

**Validation-split comparison** (used only for model selection):

| Model | R² | MAE | MSE | RMSE |
|---|---:|---:|---:|---:|
| Baseline (mean rating) | -0.0199 | 0.836 | 1.192 | 1.092 |
| Linear Regression | 0.0854 | 0.803 | 1.069 | 1.034 |
| Random Forest | 0.0857 | 0.804 | 1.069 | 1.034 |
| **Gradient Boosting** | **0.1066** | **0.800** | **1.044** | **1.022** |

**Selected model: Gradient Boosting** (highest validation R², lowest
error across all four metrics).

**Final outer-test evaluation** (retrained on train+val, evaluated once
on the untouched future-period holdout):

| Model | R² | MAE | MSE | RMSE |
|---|---:|---:|---:|---:|
| Baseline (mean rating) | -0.0033 | 0.804 | 1.023 | 1.011 |
| **Gradient Boosting (selected)** | **0.0724** | **0.775** | **0.946** | **0.973** |
| Gradient Boosting (train+val, for reference) | 0.1471 | 0.704 | 0.797 | 0.893 |

Gradient Boosting beats the baseline on every metric on the true holdout
period, and the train-vs-test gap (R² 0.147 → 0.072) is a normal, modest
amount of overfitting for a boosted-tree model on this feature set — not
a red flag.

## 4. Feature importance & interpretation

Top predictors for the selected Gradient Boosting model
(`docs/regression/feature_importance.csv` has the full ranking):

| Feature | Importance |
|---|---:|
| `attraction_avg_rating` | 0.600 |
| `attraction_rating_std` | 0.083 |
| `VisitMonth` | 0.078 |
| `attraction_distinct_users` | 0.041 |
| `attraction_total_visits` | 0.032 |
| `is_ambiguous_repeat` | 0.022 |
| `user_avg_rating` | 0.016 |

**Interpretation:**
- **An attraction's own historical average rating dominates the model**
  (60% of total importance) — unsurprising and reassuring: a site that
  has historically satisfied visitors keeps doing so, and this is exactly
  the kind of signal the leakage-safe "historical-as-of" feature was
  designed to capture safely.
- **`VisitMonth` matters more than most other features** (7.8%) — some
  seasonal effect on satisfaction exists independent of which attraction
  or user is involved (e.g. crowding or weather in certain months).
- **Attraction popularity/consistency** (`attraction_distinct_users`,
  `attraction_total_visits`, `attraction_rating_std`) contribute
  meaningfully but far less than the average rating itself — a
  well-established, popular attraction is a weaker satisfaction signal
  than its own track record of ratings.
- **User-side features contribute little** (`user_avg_rating` at only
  1.7%, `user_total_visits`/`user_distinct_attractions` further down the
  ranking) — this lines up directly with the EDA/SQL finding that 72% of
  travelers are one-time visitors: `has_user_history` is 0 for 83.4% of
  rows, so for most transactions there simply is no real user history to
  learn from, capping how much this feature family can help.

## 5. Validation

- **Reproducible training:** the script was deleted and re-run from
  scratch (`rm -rf models/regression docs/regression && python
  scripts/train_regression_model.py`); it produced byte-identical metrics
  both times (fixed `random_state=42` throughout; the split itself is now
  fully deterministic period-boundary logic with no randomness involved
  at all, having previously used a random tiebreak only for within-period
  row order, which the fix removes entirely since periods are no longer
  divided).
- **No target leakage — automated checks, all passed**
  (`docs/regression/metrics.json` → `leakage_validation.all_passed:
  true`):
  - `Rating` is confirmed absent from the feature list.
  - `TransactionId` is confirmed absent from the feature list and
    confirmed not used to order/tie-break the split.
  - No missing values in any feature used for training.
  - `max(inner_train period) = 24211 < min(inner_val period) = 24212`
    — **strictly** less, not less-or-equal.
  - `max(train+val period) = 24221 < min(test period) = 24222` —
    **strictly** less, not less-or-equal.
  - **Calendar-month atomicity, explicitly checked:** the set of periods
    in `inner_train`, the set in `inner_val`, and the set in `test` are
    pairwise disjoint — no `(VisitYear, VisitMonth)` appears in more than
    one split.
  - Outer test `TransactionId`s are confirmed disjoint from every
    train/val `TransactionId`.
- **Metrics generated successfully:** all four required metrics (R², MAE,
  MSE, RMSE) computed for the baseline and all three candidate models on
  validation, and for the baseline and selected model on the final test
  set.
- **Saved model loads and predicts successfully:** the script itself
  reloads `best_model.joblib` immediately after saving and confirms its
  predictions exactly match the in-memory model
  (`reload_check_passed: true`); this was independently re-confirmed in a
  completely fresh Python process (new interpreter, model loaded from
  disk, predictions produced without error).
- **No regressions:** a recursive diff of this checkpoint against the
  prior `tourism_checkpoint_05_regression.zip` confirms the only change
  is inside `scripts/train_regression_model.py` (the split fix) plus the
  regenerated `models/regression/` and `docs/regression*` artifacts —
  `requirements.txt`, every other script, and all data files are
  byte-identical. Transitively, a diff against
  `tourism_checkpoint_04_sql.zip` still confirms no raw/cleaned/processed
  data file was modified (`regression_features.csv` byte-identical), no
  feature-engineering script was touched (`scripts/feature_engineering.py`
  byte-identical), and no SQL/EDA/frontend file was touched.

## 6. Limitations

- **Predictive power is modest** (test R² ≈ 0.07). This is a property of
  the data, not an implementation gap: ratings in this dataset skew
  heavily positive and narrow (79% are 4–5★, std ≈ 0.98 on a 1–5 scale),
  and — per §4 — the single strongest lever (an attraction's own
  historical rating) is unavailable or weak for the 83% of transactions
  where `has_user_history` is 0 and for the small share of attractions
  with limited history. There is a real ceiling on how well `Rating` can
  be predicted from what this feature set alone can see.
- The outer test set's later years are small in absolute terms (2020:
  491 rows, 2021: 34, 2022: 239, out of 10,004 test rows total) — the
  test metrics are still dominated by 2019 volume, so they should not be
  read as a robust measure of performance specifically on the COVID-era
  rows.
- Hyperparameters for Random Forest / Gradient Boosting were set to
  reasonable, fixed values rather than tuned via grid/random search —
  in scope terms this is a model *comparison*, not a hyperparameter
  optimization exercise; further tuning of the selected Gradient
  Boosting model is a natural next step but was not attempted here.
- Categorical geography features were capped at `UserContinent` (5) and
  `UserRegion` (22) to control dimensionality (see §2); a
  higher-cardinality encoding of `UserCountry` (e.g. target/frequency
  encoding instead of one-hot) might recover some additional signal but
  was out of scope for this checkpoint.
- This checkpoint does not touch or re-validate the classification or
  recommendation tasks, or the leakage-safe feature engineering itself —
  it only consumes `regression_features.csv` as delivered.

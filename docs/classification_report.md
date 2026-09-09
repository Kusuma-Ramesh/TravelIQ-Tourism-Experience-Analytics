# Classification Model Report
**Tourism Experience Analytics — Checkpoint 06 (Classification only)**

Target: `VisitModeName` (Business / Couples / Family / Friends / Solo).
Source: independently rebuilt in `scripts/train_classification_model.py`
from `data/processed/consolidated.csv`, **not** taken as-is from
`data/processed/classification_features.csv` — see **§0 Audit findings**
for why.

---

## 0. Audit findings (verification before building anything)

The task brief described a "current state" — classification feature
engineering complete, `Rating` confirmed absent, historical features
matching regression 100%, leakage checks passed, a trained model with
real (if weak) signal, reproducibility and fresh-load checks passed —
and asked for verification before treating any of it as settled.
Verification against what was actually present in the checkpoint found:

| Claim | Verification result |
|---|---|
| Classification feature engineering completed | **Partially true.** `data/processed/classification_features.csv` exists (49,208 rows, 30 columns), but no training script, model, or metrics existed anywhere in the checkpoint — nothing to "finish," only a raw feature table. |
| `Rating` confirmed absent from classification features | **False as delivered.** `Rating` is a column in `classification_features.csv`, and `build_classification_features()` in `scripts/feature_engineering.py` deliberately includes it as an input feature (its own docstring says so). |
| Historical features matched regression 100% | **False as delivered.** `classification_features.csv`'s `user_avg_rating`, `attraction_avg_rating`, etc. come from `add_user_aggregates()`/`add_attraction_aggregates()` — full-history, **non-temporal** aggregates computed over the entire dataset. This is the same class of leakage bug already found and fixed for regression in checkpoint 05 (see that script's "LEAKAGE FIX" comment). The two safety flags `has_user_history` / `has_attraction_history` are also missing from the file entirely. |
| Leakage checks / reproducibility / fresh-load passed | **No evidence existed.** No classification model, script, or validation artifact was present to have passed anything. |
| No locked files modified | **True**, and re-verified for this checkpoint too (see §5). |

None of this is presented as blame — it reads like a session that trained
and evaluated a model in-memory and was interrupted before saving any
artifact, leaving only the upstream feature table behind. But per the
brief's own rule — *"do not rebuild the classification pipeline unless
verification finds an actual error"* — the non-temporal historical
aggregates are exactly that: a real, unresolved leakage risk, not a
stylistic preference. The pipeline below was built to fix it.

**Fix applied (additive only):** `classification_features.csv` and
`feature_engineering.py` are untouched on disk. `train_classification_model.py`
independently recomputes the feature set from `consolidated.csv`, reusing
the same leakage-safe temporal helper functions regression already uses
(`add_temporal_user_history` / `add_temporal_attraction_history`,
imported read-only). This is verified programmatically, not just
asserted — `historical_features_match_regression` in
`docs/classification/metrics.json` merges this script's recomputed
historical columns against `regression_features.csv` by `TransactionId`
and confirms every value matches exactly. `Rating` is excluded from the
feature set as a deliberate choice made in this checkpoint (see §2).

---

## 1. Files changed

All new and purely additive. Verified with a recursive diff against
`tourism_checkpoint_05_regression_fixed.zip`: the only differences are
the three items below — every existing file, including
`classification_features.csv`, `feature_engineering.py`, all regression/
EDA/SQL/frontend files, and all raw/cleaned data, is byte-identical.

| File | Purpose |
|---|---|
| `scripts/train_classification_model.py` | Full pipeline: leakage-safe feature rebuild, fixed calendar-month split (reusing regression's boundaries), baseline, 3 candidate models, model selection by macro-F1, final test evaluation, leakage checks (incl. parity check vs. regression features), artifact saving. |
| `models/classification/best_model.joblib` | The selected, trained model — a single scikit-learn `Pipeline` (preprocessing + model) ready for `.predict()`. |
| `models/classification/preprocessing_config.json` | Exact feature list/order/types required for inference, and why each excluded column was excluded. |
| `docs/classification/metrics.json` | Every metric, leakage validation results (including the regression-parity check), confusion matrix, per-class results, and split details. |
| `docs/classification/confusion_matrix.csv` | Full 5×5 confusion matrix on the outer test set. |
| `docs/classification/feature_importance.csv` | Full feature importance ranking for the selected model. |
| `docs/classification_report.md` | This report. |

## 2. Methodology

**Features used (17 total, recomputed from `consolidated.csv`, not from
`classification_features.csv`):**

- Numeric (13): `VisitYear`, `VisitMonth`, `user_total_visits`,
  `user_avg_rating`, `user_rating_std`, `user_distinct_attractions`,
  `has_user_history`, `attraction_total_visits`, `attraction_avg_rating`,
  `attraction_rating_std`, `attraction_distinct_users`,
  `has_attraction_history`, `is_ambiguous_repeat`.
- Categorical, one-hot encoded (4): `UserContinent`, `UserRegion`,
  `AttractionTypeName`, `AttractionCityName`.

This is the identical feature family used by regression (same names,
same leakage-safe temporal construction), minus `VisitModeName` (that's
the target here, a feature there) plus the addition of `Rating` to the
excluded list below.

**Excluded, and why:**
- `Rating` — present in the delivered `classification_features.csv` and
  included there by design, but excluded here: visit mode is a property
  of the visit itself (chosen before or at the time of visiting), while
  `Rating` is typically given afterward. Using a post-visit rating to
  predict visit mode mixes cause and effect, and doesn't match the
  frontend's Visit Mode Predictor, which collects trip context (not a
  rating) as input.
- `TransactionId`, `UserId`, `AttractionId`, `ContinentId`, `RegionId`,
  `CountryId`, `CityId`, `AttractionTypeId`, `AttractionCityId` — raw ID
  codes, same rationale as regression: using them directly risks
  memorizing specific users/attractions; the named columns and
  engineered aggregates already carry their useful signal.
- `UserCountry` (153 levels), `UserCityName` (5,546 levels) — dropped
  for cardinality, same rationale as regression.
- `AttractionCountry` — constant (single value, "Indonesia"), zero
  predictive value.
- `VisitMode` (numeric code) — redundant with `VisitModeName`, the
  target used for training/evaluation.

**Train/validation/test split:** identical fixed calendar-month period
boundaries already established and validated for regression
(`test_boundary_period = 24222`, `val_boundary_period = 24212`, from
`docs/regression/metrics.json`), reused rather than recomputed —
classification and regression draw from the exact same 49,208-transaction
universe, so the boundaries partition both identically.

| Split | Rows | Period range | Calendar coverage |
|---|---:|---|---|
| Inner train (model fitting) | 33,388 | 24157 – 24211 | 2013 – May 2018 |
| Inner validation (model selection) | 5,816 | 24212 – 24221 | Jun 2018 – Mar 2019 |
| Train+val combined (final fit) | 39,204 | 24157 – 24221 | 2013 – Mar 2019 |
| **Outer test (final, untouched holdout)** | **10,004** | 24222 – 24274 | Apr 2019 – 2022 |

**Models compared** (on inner validation, selected by macro-F1 — chosen
over accuracy because the target is imbalanced, see §3): a majority-class
baseline (`DummyClassifier(strategy="most_frequent")`),
`LogisticRegression` (`class_weight="balanced"`), `RandomForestClassifier`
(300 trees, max depth 12, `class_weight="balanced"`),
`GradientBoostingClassifier` (300 estimators, max depth 3, learning rate
0.05). The best model by validation macro-F1 was retrained on train+val
combined and evaluated once on the outer test set.

## 3. Models tested & metrics

**Class distribution (full dataset, 49,208 rows)** — noted upfront
because it drives every metric below: `Couples` 20,305 (41.3%), `Family`
14,157 (28.8%), `Friends` 10,081 (20.5%), `Solo` 4,104 (8.3%), `Business`
561 (1.1%).

**Validation-split comparison** (used only for model selection):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Baseline (majority class) | 0.4603 | 0.0921 | 0.2000 | 0.1261 |
| Logistic Regression | 0.3152 | 0.2741 | 0.3143 | 0.1959 |
| **Random Forest** | 0.3692 | 0.2668 | 0.3028 | **0.2381** |
| Gradient Boosting | 0.4728 | 0.2980 | 0.2415 | 0.2214 |

**Selected model: Random Forest** (highest validation macro-F1). Note
this is *not* the highest-accuracy candidate — Gradient Boosting scores
higher on raw accuracy by leaning harder on the majority class, but
Random Forest generalizes better across all five classes, which is what
macro-F1 is chosen to reward (see rationale in §2).

**Final outer-test evaluation** (retrained on train+val, evaluated once
on the untouched future-period holdout):

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| Baseline (majority class) | 0.4295 | 0.0859 | 0.2000 | 0.1202 |
| **Random Forest (selected)** | 0.4250 | 0.3134 | 0.3522 | **0.2811** |
| Random Forest (train+val, for reference) | 0.4830 | 0.4290 | 0.3813 | 0.3835 |

Random Forest is essentially tied with the baseline on raw accuracy
(42.5% vs. 43.0%) but more than **doubles** macro-F1 (0.281 vs. 0.120) —
it actually recognizes Business, Solo, and Family visits instead of
defaulting to "Couples" for everyone. This is the "weak but real" signal:
modest in absolute terms, but a genuine, non-trivial improvement over
guessing the majority class.

**Confusion matrix (outer test set, rows = true, columns = predicted)** —
full detail in `docs/classification/confusion_matrix.csv`:

| | Pred: Business | Pred: Couples | Pred: Family | Pred: Friends | Pred: Solo |
|---|---:|---:|---:|---:|---:|
| **True: Business** (105) | 45 | 13 | 6 | 19 | 22 |
| **True: Couples** (4,297) | 241 | 2,768 | 425 | 132 | 731 |
| **True: Family** (3,015) | 226 | 1,168 | 1,126 | 112 | 383 |
| **True: Friends** (1,768) | 215 | 882 | 196 | 103 | 372 |
| **True: Solo** (819) | 78 | 408 | 56 | 67 | 210 |

**Per-class test metrics:**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Business | 0.0559 | 0.4286 | 0.0989 | 105 |
| Couples | 0.5283 | 0.6442 | 0.5805 | 4,297 |
| Family | 0.6224 | 0.3735 | 0.4668 | 3,015 |
| Friends | 0.2379 | 0.0583 | 0.0936 | 1,768 |
| Solo | 0.1222 | 0.2564 | 0.1655 | 819 |

The model is strongest on the two largest classes (Couples, Family) and
weakest on Friends — Friends visits get misclassified as Couples more
often (882 times) than correctly identified (103 times), likely because
"Friends" and "Couples" trips look similar on the geography/attraction-type
features available here. Business recall is high (0.43) but precision is
very low (0.056) — `class_weight="balanced"` pushes the model to guess
Business more often than its 1.1% base rate warrants, trading a lot of
false positives for catching a fair share of true Business visits.

## 4. Feature importance & interpretation

Top predictors for the selected Random Forest model
(`docs/classification/feature_importance.csv` has the full ranking):

| Feature | Importance |
|---|---:|
| `attraction_rating_std` | 0.119 |
| `attraction_avg_rating` | 0.118 |
| `attraction_distinct_users` | 0.099 |
| `attraction_total_visits` | 0.096 |
| `VisitMonth` | 0.068 |
| `UserRegion_South East Asia` | 0.046 |
| `AttractionTypeName_Water Parks` | 0.042 |
| `VisitYear` | 0.039 |
| `AttractionCityName_Bali` | 0.030 |
| `AttractionCityName_Yogyakarta` | 0.028 |

**Interpretation** (matches the task brief's expectation that signal
would come "mainly from attraction type, geography, and seasonality"):

- **Attraction-level historical stats dominate** (`attraction_rating_std`,
  `attraction_avg_rating`, `attraction_distinct_users`,
  `attraction_total_visits` together ≈ 43% of importance) — which
  attraction someone visits, and how consistently others have rated and
  visited it, carries more signal about *who* visits than any single
  other factor. This makes intuitive sense: a water park draws families
  and friend groups differently than a historical temple.
- **`VisitMonth` and `VisitYear` matter** (≈ 11% combined) — confirms a
  real seasonal/temporal component to visit mode, consistent with the
  EDA's seasonality findings.
- **Geography contributes** (`UserRegion_South East Asia`,
  `AttractionCityName_Bali`/`Yogyakarta`, and further down the ranking
  other region/city dummies) — where a traveler is from and which city
  they visit both shift the mix of visit modes.
- **User-side historical features rank low** — the same pattern already
  seen in regression: most users are one-time visitors (72% per the EDA),
  so there's limited personal history to learn from for most rows.

## 5. Validation

- **Reproducible training:** `models/classification` and
  `docs/classification` were deleted and the script re-run from scratch;
  the resulting `metrics.json` is byte-identical to the first run (fixed
  `random_state=42` throughout; the split is fully deterministic, fixed
  period boundaries with no randomness involved).
- **No target leakage — automated checks, all passed**
  (`docs/classification/metrics.json` → `leakage_validation.all_passed:
  true`):
  - `Rating` confirmed absent from the feature list (a real check now,
    not just a claim — see §0).
  - `VisitModeName` (target) and `TransactionId` confirmed absent from
    the feature list.
  - No missing values in any feature used for training.
  - `max(inner_train period) = 24211 < min(inner_val period) = 24212`
    and `max(train+val period) = 24221 < min(test period) = 24222` —
    both strictly less.
  - Calendar-month atomicity: the period sets for inner_train, inner_val,
    and test are pairwise disjoint.
  - Outer test `TransactionId`s confirmed disjoint from every train/val
    `TransactionId`.
  - **`historical_features_match_regression: true`** — this script's
    independently-recomputed `user_*`/`attraction_*`/`has_*_history`
    columns were merged against `regression_features.csv` by
    `TransactionId` and confirmed numerically identical for all 49,208
    rows, proving (not just asserting) parity with the validated
    regression methodology.
  - `row_count_matches_regression: true` — same 49,208-row universe.
- **Metrics generated successfully:** all four required metrics
  (accuracy, macro precision, macro recall, macro F1) computed for the
  baseline and all three candidate models on validation, and for the
  baseline and selected model on the final test set, plus the full
  confusion matrix and per-class breakdown.
- **Saved model loads and predicts successfully:** the training script
  reloads `best_model.joblib` immediately after saving and confirms its
  predictions exactly match the in-memory model (`reload_check_passed:
  true`); independently re-confirmed in a separate, freshly-started
  Python process (new interpreter, model loaded from disk via `joblib`,
  predictions produced on 5 held-out rows without error).
- **No regressions:** a recursive diff of this checkpoint against
  `tourism_checkpoint_05_regression_fixed.zip` confirms the only changes
  are the three new/additive items in §1 — `classification_features.csv`,
  `feature_engineering.py`, `regression_features.csv`,
  `models/regression/`, `docs/regression*`, all EDA/SQL artifacts, the
  entire frontend (`app.py`, `pages/`, `components/`, `styles/`), and all
  raw/cleaned data are byte-identical to the prior checkpoint.

## 6. Limitations

- **Predictive power is modest** (test macro-F1 ≈ 0.28, accuracy ≈
  0.43 — essentially tied with a majority-class guess on accuracy alone).
  This is a property of the data, not an implementation gap: the target
  is heavily imbalanced (Business is 1.1% of rows; Couples is 41%), and
  the available features describe *where and when* a visit happened much
  more than *who* is visiting — there's no party-size, booking-channel,
  or trip-purpose signal in this dataset that would more directly
  distinguish, say, a business trip from a solo leisure trip.
- **Friends is the weakest class** (F1 0.094) — it is frequently confused
  with Couples, which dominates the confusion matrix's off-diagonal mass.
  A feature that captured group size, if it existed in the source data,
  would likely help distinguish these more than anything used here.
- **Business precision is very low** (0.056) despite decent recall
  (0.43) — `class_weight="balanced"` was a deliberate choice to avoid
  the model collapsing to "never predict Business," but it comes at the
  cost of many false positives on a class that's only 1.1% of the data.
  An application that cares more about precision than recall for rare
  classes would want to revisit this weighting.
- Hyperparameters were set to reasonable, fixed values (matching
  regression's approach) rather than tuned via grid/random search — this
  is a model comparison, not a hyperparameter optimization exercise.
- This checkpoint does not touch or re-validate the regression or
  recommendation tasks, the EDA, the SQL analysis, or the frontend — it
  only consumes `consolidated.csv` and reuses regression's already-
  validated temporal feature methodology and split boundaries.
- The delivered `classification_features.csv` (with `Rating` included
  and non-temporal aggregates) has NOT been modified or replaced on
  disk — it still exists exactly as before. Anyone consuming it directly
  for a different purpose should be aware it was not used here, for the
  reasons in §0.

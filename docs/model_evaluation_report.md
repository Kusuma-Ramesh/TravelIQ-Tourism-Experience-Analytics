# Final ML Evaluation — Consolidated Report (Checkpoint 08)

Scope: **evaluation consolidation only**. No model was retrained, refit, or redesigned to produce this document. Every metric below is read directly from the existing saved files:

- `docs/regression/metrics.json`
- `docs/classification/metrics.json`
- `docs/classification/confusion_matrix.csv`
- `docs/recommendation/metrics.json`
- `docs/recommendation/validation_results.json`
- `models/{regression,classification,recommendation}/*` (for model-name / config cross-checks only)

A machine-readable version of everything in this report is in `docs/model_evaluation_summary.json`.

---

## 1. Regression — predicting `Rating`

**Selected model:** Gradient Boosting (`GradientBoostingRegressor`, 300 estimators, max depth 3, learning rate 0.05), inside a scikit-learn `Pipeline` — confirmed by loading `models/regression/best_model.joblib` directly (final pipeline step is a `GradientBoostingRegressor`), matching `selected_model: "gradient_boosting"` in both `docs/regression/metrics.json` and `models/regression/preprocessing_config.json`.

**Test-set metrics (final, untouched holdout):**

| Metric | Baseline (mean rating) | Gradient Boosting (selected) |
|---|---:|---:|
| R² | -0.0033 | **0.0724** |
| MAE | 0.8043 | **0.7752** |
| MSE | 1.023 | **0.9459** |
| RMSE | 1.0114 | **0.9725** |

For reference, validation-split performance used for model selection: R²=0.1066, MAE=0.7999, MSE=1.044, RMSE=1.0218 (Gradient Boosting was the best of 3 candidates — Linear Regression and Random Forest scored R² 0.0854 and 0.0857 respectively on the same validation split; see `model_comparison_on_validation` in `docs/regression/metrics.json` for the full candidate table).

**Baseline comparison:** Gradient Boosting beats the mean-rating baseline on every test metric (R² 0.0724 vs. -0.0033; RMSE 0.9725 vs. 1.0114) — a real, if modest, improvement over always predicting the average rating.

**Temporal train/validation/test methodology:**

Chronological, calendar-month-atomic split: rows are grouped by `VisitYear*12+VisitMonth`, and boundaries are placed *between* whole periods so no calendar month is ever divided across splits.

| Split | Rows | Period range | Calendar coverage (verified) |
|---|---:|---|---|
| Inner train | 33,388 | 24157–24211 | Jan 2013 - Jul 2017 |
| Inner validation | 5,816 | 24212–24221 | Aug 2017 - May 2018 |
| Train+val combined (final fit) | 39,204 | 24157–24221 | Jan 2013 - May 2018 |
| **Outer test** | **10,004** | 24222–24274 | **Jun 2018 - Oct 2022** |

> **Note:** the calendar-coverage column above was independently recomputed from `regression_features.csv` for this consolidation, because the prose in `docs/regression_report.md` §2 / `docs/classification_report.md` §2 describes different (incorrect) calendar labels for the same period-integer boundaries. See **§4 Inconsistencies found** below — this affects wording only, not the split logic, leakage checks, or any reported metric.

Leakage validation (`docs/regression/metrics.json` → `leakage_validation.all_passed`): **True** — target/`TransactionId` absent from features, no missing values, strictly increasing period boundaries, calendar-month atomicity, and test `TransactionId`s disjoint from train/val, all confirmed programmatically (not just asserted).

**Important feature interpretation** (top of `docs/regression/feature_importance.csv`):

| Feature | Importance |
|---|---:|
| `attraction_avg_rating` | 0.600 |
| `attraction_rating_std` | 0.083 |
| `VisitMonth` | 0.078 |
| `attraction_distinct_users` | 0.041 |
| `attraction_total_visits` | 0.032 |
| `is_ambiguous_repeat` | 0.022 |
| `user_avg_rating` | 0.016 |

`attraction_avg_rating` alone accounts for ~60% of importance — an attraction's own historical rating track record is by far the strongest predictor of a new rating for it. `VisitMonth` (~7.8%) indicates a real seasonal effect. User-side features contribute little, consistent with 72% of travelers being one-time visitors (per the EDA) — there's limited personal history for most rows to learn from.

---

## 2. Classification — predicting `VisitModeName`

**Selected model:** Random Forest (`RandomForestClassifier`, 300 trees, max depth 12, `class_weight="balanced"`), inside a scikit-learn `Pipeline` — confirmed by loading `models/classification/best_model.joblib` directly (final pipeline step is a `RandomForestClassifier`), matching `selected_model: "random_forest"` in both `docs/classification/metrics.json` and `models/classification/preprocessing_config.json`. Selected by validation macro-F1, not accuracy, because the target is heavily imbalanced (Couples 41.3%, Business 1.1%).

**Test-set metrics (final, untouched holdout):**

| Metric | Baseline (majority class) | Random Forest (selected) |
|---|---:|---:|
| Accuracy | 0.4295 | **0.425** |
| Macro Precision | 0.0859 | **0.3134** |
| Macro Recall | 0.2 | **0.3522** |
| Macro F1 | 0.1202 | **0.2811** |

**Baseline comparison:** accuracy is essentially tied with the majority-class baseline (0.425 vs. 0.4295), but macro-F1 is more than double (0.2811 vs. 0.1202) — the model actually recognizes minority classes (Business, Solo, Family) instead of defaulting to Couples for everyone, which is the entire point of choosing macro-F1 over accuracy for an imbalanced target.

**Per-class test results:**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Business | 0.0559 | 0.4286 | 0.0989 | 105 |
| Couples | 0.5283 | 0.6442 | 0.5805 | 4,297 |
| Family | 0.6224 | 0.3735 | 0.4668 | 3,015 |
| Friends | 0.2379 | 0.0583 | 0.0936 | 1,768 |
| Solo | 0.1222 | 0.2564 | 0.1655 | 819 |

**Confusion matrix (test set, rows = true, columns = predicted):**

| True \ Pred | Business | Couples | Family | Friends | Solo |
|---|---:|---:|---:|---:|---:|
| **Business** | 45 | 13 | 6 | 19 | 22 |
| **Couples** | 241 | 2768 | 425 | 132 | 731 |
| **Family** | 226 | 1168 | 1126 | 112 | 383 |
| **Friends** | 215 | 882 | 196 | 103 | 372 |
| **Solo** | 78 | 408 | 56 | 67 | 210 |

**Interpretation:** the model is strongest on the two largest classes (Couples, Family) and weakest on Friends, which is misclassified as Couples far more often (882 times) than correctly identified (103 times) — the two trip types likely look similar on the geography/attraction-type features available. Business recall is comparatively high (0.43) but precision is very low (0.056): `class_weight="balanced"` pushes the model to predict Business more than its 1.1% base rate would otherwise warrant, trading many false positives for catching a fair share of true Business visits.

**Temporal split methodology:** identical fixed calendar-period boundaries as regression (`test_boundary_period=24222`, `val_boundary_period=24212`), reused rather than recomputed, since classification and regression draw from the exact same 49,208-transaction universe. See the corrected calendar coverage table in §1 above — the same correction applies here.

Leakage validation (`docs/classification/metrics.json` → `leakage_validation.all_passed`): **True** — additionally includes `historical_features_match_regression: True`, a numeric parity check confirming this task's independently-recomputed historical aggregates exactly match regression's validated ones.

**Top features** (full ranking in `docs/classification/feature_importance.csv`):

| Feature | Importance |
|---|---:|
| `attraction_rating_std` | 0.119 |
| `attraction_avg_rating` | 0.118 |
| `attraction_distinct_users` | 0.099 |
| `attraction_total_visits` | 0.096 |
| `VisitMonth` | 0.068 |
| `UserRegion_South East Asia` | 0.046 |
| `AttractionTypeName_Water Parks` | 0.042 |

Attraction-level historical stats dominate (~43% combined), with `VisitMonth`/`VisitYear` (~11%) confirming a seasonal component and geography (`UserRegion_South East Asia`, city dummies) contributing further down the ranking.

---

## 3. Recommendation — attraction recommendations

**Approach:** simple content/profile-based recommender (no collaborative filtering, no hybrid, no deep learning). Attraction content = `AttractionTypeName` + normalized average rating + normalized log-popularity. User profile = rating-weighted share of past visits per attraction type. Personalized score = 0.6·type_affinity + 0.25·norm_quality + 0.15·norm_popularity, with deterministic tie-breaking by ascending `AttractionId`.

**Time-aware evaluation** (global cutoff: train `VisitYear<=2018`, test `VisitYear>2018`; see §5 note on why this differs from regression/classification's split):

| Segment | k | Precision@k | Recall@k | MAP@k | Users evaluated |
|---|---:|---:|---:|---:|---:|
| Warm (personalized) | 5 | 0.114 | 0.454 | 0.206 | 726 |
| Warm (personalized) | 10 | 0.097 | 0.757 | 0.255 | 726 |
| Cold (fallback) | 5 | 0.174 | 0.711 | 0.458 | 4,106 |
| Cold (fallback) | 10 | 0.099 | 0.815 | 0.473 | 4,106 |
| **Overall** | 5 | 0.165 | 0.672 | 0.420 | 4,832 |
| **Overall** | 10 | 0.099 | 0.806 | 0.440 | 4,832 |

**Baseline comparison** (closed-form expectation of a uniform-random pick from the 30-attraction catalog — not a sampled run):

- Expected recall@k = k/30 → 0.1667 (k=5), 0.3333 (k=10)
- Expected precision@k ≈ 0.0415 (k-independent; mean relevant items / catalog size)

The model beats this baseline by roughly 3–4x on precision (0.165 vs. 0.0415) and matches or beats it on recall at k=5, for both the warm (personalized) and cold (fallback) segments.

**Cold-start strategy:** any user with no rated history (or history that sums to zero rating weight) automatically receives a purely popularity/rating-based fallback score (`0.5·norm_quality + 0.5·norm_popularity`), still fully deterministic. Verified with a synthetic brand-new `UserId` guaranteed absent from the data (`docs/recommendation/validation_results.json` → `cold_start_users_work: PASS`).

**Previously-visited exclusion:** every candidate attraction the target user has already visited (per their available history at prediction time) is removed from the candidate pool before scoring — verified programmatically (`previously_visited_excluded: PASS`).

**Temporal leakage prevention:** a *global* time cutoff (not per-user leave-last-out) is used so no user's training window can overlap another user's test window. The evaluation-only model's attraction statistics are verified to come exclusively from pre-cutoff rows (`total_visits` across all attractions sums exactly to the train row count — checked in code, not just claimed), and `train.VisitYear.max() < test.VisitYear.min()` is asserted directly. (`no_future_information_leakage: PASS`)

---

## 4. Cross-model consistency checks performed

| Check | Result |
|---|---|
| `regression_report.md` / `metrics.json` numbers match | ✅ verified — all R²/MAE/MSE/RMSE values in this report copied programmatically from `docs/regression/metrics.json` |
| `classification_report.md` / `metrics.json` numbers match | ✅ verified — all accuracy/precision/recall/F1/confusion-matrix values copied programmatically from `docs/classification/metrics.json` and `confusion_matrix.csv` |
| `recommendation_report.md` / `metrics.json` numbers match | ✅ verified — all Precision/Recall/MAP@k values copied programmatically from `docs/recommendation/metrics.json` |
| Selected model name consistent across report, `metrics.json`, and `preprocessing_config.json` (regression) | ✅ `gradient_boosting` everywhere |
| Selected model name consistent across report, `metrics.json`, and `preprocessing_config.json` (classification) | ✅ `random_forest` everywhere |
| Serialized model class matches the claimed selected model | ✅ `best_model.joblib` final pipeline step is `GradientBoostingRegressor` (regression) / `RandomForestClassifier` (classification) |
| Regression & classification split boundaries identical | ✅ classification explicitly reuses regression's `test_boundary_period`/`val_boundary_period` (`boundaries_source` field in `docs/classification/metrics.json`) |
| No leakage claim overstated | ✅ each `leakage_validation.all_passed` is backed by explicit itemized boolean checks in the same JSON, not a single unverified flag |
| Recommendation split independent of regression/classification split | Expected — different script, different feature table, different task; see note below |

> **Note on split granularity:** regression/classification use a period-level (`VisitYear*12+VisitMonth`) chronological split; recommendation uses a coarser calendar-year cutoff (`VisitYear<=2018`). This is expected, not a contradiction — they are separate evaluations on separate feature tables (`regression_features.csv`'s 49,208 rows / 1 target vs. `recommendation_interactions.csv`'s 45,275 rows / 30 attractions).

## 5. Inconsistencies found

**1. docs/regression_report.md (§2) and docs/classification_report.md (§2), prose calendar-coverage captions**

- **Issue:** The prose text describes the split as inner_train='2013 – May 2018', inner_val='Jun 2018 – Mar 2019', test='Apr 2019 – 2022'. Decoding the *actual* period integers stored in the same reports' own metrics.json (`_period = VisitYear*12 + VisitMonth`, verified directly against `regression_features.csv`) gives a different mapping: inner_train=Jan 2013 - Jul 2017, inner_val=Aug 2017 - May 2018, test=Jun 2018 - Oct 2022. This is corroborated by `test_year_counts` in `docs/regression/metrics.json`, which shows 3,820 test rows from 2018 — impossible if the test window only started Apr 2019 as the prose claims.
- **Impact:** Documentation-only. The split boundaries themselves (the period integers, the leakage checks, the row counts, and every reported metric) are correct and unaffected — only the human-readable calendar caption in the two existing per-model reports is mislabeled. Not a leakage issue: train periods still strictly precede val periods, which still strictly precede test periods.
- **Resolution:** This consolidated report uses the corrected calendar mapping, computed directly from the period integers and cross-verified against the raw processed data. The original regression_report.md / classification_report.md files were left untouched, per the instruction not to modify existing regression/classification implementations or reports — this is noted here for the record instead.

## 6. Validation performed for this consolidation

Lightweight, read-only checks only (no retraining):

- Re-loaded `models/regression/best_model.joblib` and `models/classification/best_model.joblib` directly and confirmed the final pipeline step's class name matches the `selected_model` string claimed in both the metrics JSON and the preprocessing config.
- Re-derived the regression/classification split's calendar coverage from `regression_features.csv` (`VisitYear`, `VisitMonth`) and cross-checked it against `test_year_counts` in `docs/regression/metrics.json` (found and documented the calendar-labeling inconsistency in §5).
- Diffed every number transcribed into this report against its source JSON/CSV programmatically (see `scripts/build_evaluation_summary.py`) — this report cannot drift from the saved metric files because it is generated from them.
- Confirmed all three `leakage_validation` / `validation_results` blocks consist of itemized, individually-named boolean checks, not a single unverified summary flag.

## 7. Overall summary

| Task | Selected model | Headline metric |
|---|---|---|
| Regression | Gradient Boosting | Test R² = 0.0724 (baseline -0.0033) |
| Classification | Random Forest | Test macro-F1 = 0.2811 (baseline 0.1202) |
| Recommendation | Content/profile-based | Overall Precision@5 = 0.165, Recall@5 = 0.672 (random baseline precision ≈ 0.0415) |

All three tasks show modest but genuine, verified improvement over their respective baselines, with leakage-safe temporal evaluation and no unresolved contradictions between reports and saved metrics beyond the calendar-labeling documentation issue noted in §5.

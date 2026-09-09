# Recommendation System Report (Checkpoint 07)

**Scope:** recommendation system only. Built fresh on top of the verified
`tourism_checkpoint_06_classification.zip`. Classification, regression,
EDA, SQL, raw/cleaned data, and the frontend/UI/map/passport were **not
modified**.

## 1. Why this was rebuilt from scratch

A previous session reportedly completed a full recommender (module,
training/eval, cold-start, leakage checks, reproducibility, final
validation). That work-in-progress checkpoint could not be recovered and
was **not** present in `tourism_checkpoint_06_classification.zip` — that
zip's own `pages/recommendations.py` is still the disabled UI stub, and
no recommender module, model artifacts, or report existed anywhere in
it. Rather than assume or fabricate that prior state, the recommender
was built new from this verified checkpoint, per instruction.

## 2. Files added (all additive — nothing existing was edited)

| File | Purpose |
|---|---|
| `scripts/recommender.py` | Reusable `ContentProfileRecommender` class: fit, save/load, recommend, reasons |
| `scripts/train_recommendation_model.py` | Time-aware split, evaluation, final training, full validation suite |
| `models/recommendation/recommender_artifacts.joblib` | Saved production model (attraction profile + weights) |
| `models/recommendation/attraction_profile.csv` | Human-readable version of the same profile |
| `models/recommendation/config.json` | Weights + build metadata |
| `docs/recommendation/metrics.json` | Full evaluation metrics (warm/cold/overall, k=5 and k=10) |
| `docs/recommendation/validation_results.json` | Pass/fail summary of all validation checks |
| `docs/recommendation/sample_recommendations.csv` | Example output for one warm user and one cold-start user |
| `docs/recommendation/run_log.txt` | Full console log of the training run |
| `docs/recommendation_report.md` | This report |

Nothing in `models/classification`, `models/regression`, `docs/*` (other
reports), `pages/`, `components/`, `app.py`, `styles/`, `database/`, or
`data/raw` / `data/cleaned` was touched.

## 3. Approach

Simple **content/profile-based** recommender — no collaborative
filtering, no hybrid blending, no deep learning, as requested.

For each attraction, three content signals are computed directly from
interaction history (never hardcoded):
- `AttractionTypeName` — categorical descriptor (e.g. Beaches, Volcanos)
- `norm_quality` — min-max normalized average rating
- `norm_popularity` — min-max normalized `log(1 + visit count)`

For each user with prior history, a profile signal is computed:
- `type_affinity[type]` = rating-weighted share of the user's past
  visits that went to each attraction type (e.g. "38% of your
  rating-weighted history is Beaches")

**Personalized score:**
`score = 0.60·type_affinity + 0.25·norm_quality + 0.15·norm_popularity`

**Cold-start fallback** (no history, or history with zero rating
weight): `score = 0.50·norm_quality + 0.50·norm_popularity` — a purely
popularity/rating-based ranking, still fully deterministic.

Ties at any step are broken by ascending `AttractionId`, so ordering
never depends on hash/dict iteration order — the model is fully
deterministic end-to-end.

Every recommendation carries a short interpretable reason string, e.g.:
- *"Matches your past interest in Beaches (38% of your rating history), and is well regarded overall (avg 4.1/5)."*
- *"Popular pick among travelers overall — average rating 4.6/5 from 5605 visits (not enough personal history yet to personalize further)."*

## 4. Signals / features used

| Signal | Source | Leakage-safe? |
|---|---|---|
| `AttractionTypeName` | `recommendation_interactions.csv` | Yes — static attribute |
| Attraction avg rating / visit count | Computed fresh from the interaction slice passed to `.fit()` | Yes — see §6 |
| User type affinity | Computed fresh from the user's interaction slice passed to `.recommend()` | Yes — see §6 |

`user_profile_features.csv` and `attraction_profile_features.csv` (the
pre-existing processed files) were deliberately **not** used for
evaluation — they're built by `feature_engineering.py` from full-history
aggregates, so every row's stats already include that row's own future.
Using them under a chronological holdout would leak future information.
The recommender instead computes its own attraction/user statistics
directly from `recommendation_interactions.csv`, scoped to whatever time
slice the caller passes in.

## 5. Evaluation — time-aware holdout

**Split:** global time cutoff at `VisitYear <= 2018` (train, 39,266
rows) vs. `VisitYear > 2018` (test, 6,009 rows). A global cutoff (rather
than per-user leave-last-out) was used specifically to avoid a subtler
leak: one user's "training" data being drawn from a time after another
user's held-out test point.

Test users split into:
- **Warm** (726 users) — had pre-cutoff history → evaluated with the
  personalized model
- **Cold** (4,106 users) — no pre-cutoff history → evaluated with the
  popularity/rating fallback, exactly as they'd be served in production

| Segment | k | Precision@k | Recall@k | MAP@k | n users |
|---|---|---|---|---|---|
| Warm (personalized) | 5 | 0.114 | 0.454 | 0.206 | 726 |
| Warm (personalized) | 10 | 0.097 | 0.757 | 0.255 | 726 |
| Cold (fallback) | 5 | 0.174 | 0.711 | 0.458 | 4,106 |
| Cold (fallback) | 10 | 0.099 | 0.815 | 0.473 | 4,106 |
| Overall | 5 | 0.165 | 0.672 | 0.420 | 4,832 |
| Overall | 10 | 0.099 | 0.806 | 0.440 | 4,832 |

**Random-catalog baseline** (closed-form expectation of picking k of the
30 attractions uniformly at random, for context — not a sampled run):
- Expected recall@k = k/30 → 0.167 (k=5), 0.333 (k=10)
- Expected precision@k = mean(relevant items)/30 ≈ 0.041–0.043 (k-independent)

Both segments clearly beat the random baseline on precision (≈3–4x) and
match or beat it on recall at k=5. Precision naturally drops at k=10
because the catalog only has 30 attractions total, so a longer list
dilutes precision while mechanically raising recall — expected, not a bug.

## 6. Leakage prevention

- **Global time cutoff**, not per-user leave-last-out, so no user's
  training window overlaps another user's test window.
- The evaluation model's attraction statistics were fit **only** on the
  train split; verified programmatically (`total_visits` across all
  attractions in the eval profile sums exactly to the train row count —
  asserted in the script, not just eyeballed).
- User history used for scoring/exclusion during evaluation is the
  pre-cutoff slice only — a warm user's post-cutoff (test) visits are
  invisible to the model at prediction time.
- `train.VisitYear.max() < test.VisitYear.min()` is asserted directly.

## 7. Cold-start strategy

Any user with no rated history (or, as an edge case, history that sums
to zero rating weight) automatically falls back to a deterministic
popularity/rating-based ranking (`0.5·quality + 0.5·popularity`) — no
personalization is attempted, and reasons are labeled accordingly.
Verified with a synthetic brand-new `UserId` guaranteed absent from the
data.

## 8. Validation results (all passed)

| Check | Result |
|---|---|
| Real attractions only | PASS |
| Previously-visited attractions excluded | PASS |
| Scores deterministic (repeat calls identical) | PASS |
| Cold-start users work | PASS |
| Saved artifacts load correctly | PASS |
| Fresh-process inference (separate `python -c` process, cold import) | PASS |
| Evaluation runs, metrics in [0,1] | PASS |
| No future-information leakage | PASS |
| Reproducibility (two independent fits, 20 sample users) | PASS |

Full console output: `docs/recommendation/run_log.txt`. Machine-readable
summary: `docs/recommendation/validation_results.json`.

## 9. Limitations

- The interaction dataset used here (`recommendation_interactions.csv`)
  only covers **30 distinct attractions**, versus ~1,700 in the raw item
  master (`data/cleaned/updated_item.csv`). This is a pre-existing
  property of the processed file (built by the untouched
  `feature_engineering.py`), not something introduced here — but it
  caps both what can be recommended and how meaningful precision/recall
  look, since a small catalog inflates baseline hit rates.
- Content signals are coarse: attraction type + aggregate rating +
  popularity. There's no use of city/country, price, season, or text
  descriptions — kept simple by design, per the task's constraint against
  a hybrid/deep model.
- Data is heavily skewed toward 2015–2018 (2020–2022 combined are <2% of
  rows), so the post-2018 test set, while real, is thinner than the
  training data — evaluation numbers should be read as directional
  rather than a large-sample guarantee.
- Type affinity is based only on rating-weighted history within these 30
  attractions/17 types; users who only ever visited one type will show
  maximal, possibly overconfident affinity toward that single type.
- No frontend wiring was done (`pages/recommendations.py` remains the
  disabled stub) — out of scope per instructions.

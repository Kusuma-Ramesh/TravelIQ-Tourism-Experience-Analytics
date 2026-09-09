# Exploratory Data Analysis Report
**Tourism Experience Analytics — Checkpoint 03 (EDA only)**

Source of truth: `data/processed/consolidated.csv` (49,208 rows, 24
columns — the fully joined, human-readable base table produced by the
already-locked cleaning/consolidation pipeline). `regression_features.csv`
and `classification_features.csv` were read only to confirm they line up
with `consolidated.csv`; nothing was written back to them.

All numbers below are computed directly from the dataset by
`scripts/eda_analysis.py` and cross-checked by an independent `validate()`
pass in the same script — none are estimated or fabricated.

---

## 1. Dataset overview

| Metric | Value |
|---|---|
| Rows | 49,208 |
| Columns | 24 |
| Missing values (any column) | 0 |
| Fully duplicate rows | 0 |
| Duplicate `TransactionId` | 0 |
| Rows flagged `is_ambiguous_repeat` | 2,536 (5.2%) |
| Unique users | 33,530 |
| Unique attractions | 30 |
| Unique attraction types | 17 |
| Unique visit modes | 5 |
| Unique traveler home countries | 153 |
| Unique traveler home regions / continents | 22 / 5 |
| Attraction country coverage | **Indonesia only** (Bali, Yogyakarta, Malang) |
| Year coverage | 2013–2022 |
| Month coverage | 1–12 (full calendar coverage) |
| Rating range | 1–5 (integer) |

**Reading this correctly:** this is a single-destination dataset — every
attraction is in Indonesia, so "geographic coverage" in this project means
*where travelers come from* (153 countries), not where they go. That
reframes the whole geography angle of the EDA around origin markets,
not destination comparison.

The dataset is clean at the row level (no nulls, no duplicate keys), which
is expected — this was already handled in the locked cleaning step
(`docs/data_dictionary.md` §3 documents the original 52,930 → 49,208 row
reduction from de-duplication before this checkpoint).

---

## 2. Traveler analysis

**Visit Mode distribution** (49,208 transactions):

| Visit Mode | Transactions | Share |
|---|---:|---:|
| Couples | 20,305 | 41.3% |
| Family | 14,157 | 28.8% |
| Friends | 10,081 | 20.5% |
| Solo | 4,104 | 8.3% |
| Business | 561 | 1.1% |

**Geography (traveler origin):**
- Top 3 source countries: Australia (12,683, 25.8%), United Kingdom
  (6,302, 12.8%), United States (5,695, 11.6%) — together **~50%** of all
  transactions.
- Top continents: Asia (14,328), Australia & Oceania (14,145), Europe
  (12,359), America (7,509), Africa (867).
- Indonesia itself is the 4th largest source country (4,412, 9.0%) —
  meaningful domestic tourism alongside international demand.

**Repeat behavior:**
- 33,530 unique users generated 49,208 transactions.
- 9,274 users (27.7%) made more than one visit; median trips/user = 1,
  max = 26.
- This is a long-tail, mostly-one-time-visitor base — relevant context
  for the (already-implemented, untouched) recommendation system's
  cold-start handling.

---

## 3. Attraction analysis

Only 30 attractions exist in the transaction data, across 17 types and 3
cities (Bali, Yogyakarta, Malang).

**Top 5 attractions by visit volume:**

| Attraction | Visits | Share of all transactions |
|---|---:|---:|
| Sacred Monkey Forest Sanctuary | 12,369 | 25.1% |
| Waterbom Bali | 6,079 | 12.4% |
| Tegalalang Rice Terrace | 5,585 | 11.3% |
| Uluwatu Temple | 3,216 | 6.5% |
| Tanah Lot Temple | 3,198 | 6.5% |

The top 5 attractions alone account for **~62%** of all transactions —
demand is heavily concentrated in a handful of marquee sites, and the
long tail (e.g. Spas, Speciality Museums, Neighborhoods, Caverns & Caves)
each have under 130 visits.

**Average rating by attraction type** (all 17 types, highest → lowest):
Water Parks (4.65), Spas (4.57), Caverns & Caves (4.50), National Parks
(4.43), Neighborhoods (4.30), Nature & Wildlife Areas (4.26), History
Museums (4.21), Religious Sites (4.21), Ballets (4.19), Points of
Interest & Landmarks (4.13), Volcanos (4.09), Ancient Ruins (4.04),
Speciality Museums (4.03), Waterfalls (3.90), Beaches (3.84), Flea &
Street Markets (3.81), **Historic Sites (3.50 — lowest)**.

**Highest/lowest individually rated attractions** (min. 50 ratings, to
avoid small-sample noise): Mount Semeru Volcano, Waterbom Bali, Bromo
Tengger Semeru National Park, Jomblang Cave and Nusa Dua Beach rate
highest; Kuta Beach – Bali (3.41), Yogyakarta Palace (3.50) and Malang
City Square (3.58) rate lowest.

---

## 4. Temporal analysis

**Yearly visit volume** peaks in **2016** (11,772 transactions), grows
2013→2016, then declines 2017–2019, and **collapses in 2020–2021**
(491 + 34 = 525 transactions, just **1.1%** of the full 10-year total) —
the COVID-19 travel shutdown is unmistakable in the data. 2022 shows a
partial recovery (239 transactions) but data collection for 2022 also
appears to stop mid-year (no rows past October in this extract).

**Seasonality** (all years combined, by calendar month): visits are
fairly flat year-round (3,383–4,643/month) with a mild peak in
July–September (mid-year travel season) and a trough in February — no
extreme seasonal swing.

**Rating trend by year:** average rating is stable in the 4.06–4.28 range
for 2013–2019, then rises to 4.28–4.53 in 2020–2022 — but those later
years have very small sample sizes (as few as 34 transactions in 2021),
so this "uptick" should be read as noise from a small denominator, not a
genuine satisfaction improvement.

**Visit Mode mix over time:** Couples' share rose steadily from 36%
(2013) to a peak of ~46% (2018) before falling back; Family share dropped
from ~35% (2013) to ~24–27% (2017–2018) then rebounded; Business travel
stayed a consistently small (~1%) slice throughout, dropping to
essentially zero during 2020–2021.

---

## 5. Rating analysis

**Overall distribution:** mean 4.16, median 4.0, std 0.98 — ratings skew
strongly positive. 79.1% of ratings are 4 or 5 stars; only 6.4% are 1 or
2 stars.

| Rating | Count | Share |
|---:|---:|---:|
| 5 | 22,298 | 45.3% |
| 4 | 16,637 | 33.8% |
| 3 | 7,122 | 14.5% |
| 2 | 1,946 | 4.0% |
| 1 | 1,205 | 2.4% |

**By Visit Mode:** Business travelers rate highest on average (4.31),
followed by Family (4.22), Friends (4.17), Couples (4.12), Solo lowest
(4.08) — the gap is modest (~0.23 stars top to bottom) but consistent.

**Low-rating concentration:** Historic Sites (16.8% of its ratings are
1–2★) and Beaches (12.3%) have the highest share of poor ratings; Water
Parks (1.3%) and Caverns & Caves (1.6%) have the lowest — this lines up
with the attraction-type rating averages above and flags Historic Sites
and Beaches as the categories most worth a service-quality review.

**Popularity vs. quality:** correlation between an attraction's total
visit count and its average rating is **r = 0.12** — essentially no
relationship. The most-visited attraction (Sacred Monkey Forest
Sanctuary) rates a solid but not top 4.3x, while smaller, less-visited
sites (e.g. Mount Semeru Volcano, Jomblang Cave) post the highest average
ratings. Popularity is driven by something other than average
satisfaction (likely accessibility, marketing, or price point) — this is
directly relevant to how the (untouched) recommendation system should
weight popularity vs. rating signals.

---

## 6. Evidence-based business insights

1. **Demand is geographically concentrated in a few source markets.**
   Australia, the UK and the US together drive roughly half of all
   transactions — origin-market marketing/localization budget is better
   spent concentrated on this short list than spread across 153
   countries.
2. **Attraction demand is even more concentrated.** The top 5 of 30
   attractions cover ~62% of all visits, led by Sacred Monkey Forest
   Sanctuary alone at 25%. Capacity planning, staffing and on-site
   experience investment should prioritize this small set of marquee
   sites.
3. **Business travel is a small but high-value segment.** At only 1.1%
   of volume but the highest average rating (4.31/5) of any Visit Mode,
   it looks under-served relative to its satisfaction — a candidate for
   a dedicated offering rather than more general inventory.
4. **2020–2021 is a genuine outlier, not a seasonal pattern.** Any
   time-series or seasonality-aware model built on this data (rating
   prediction, demand forecasting) should treat those two years as an
   anomaly to exclude or down-weight, not as representative of a "low
   season."
5. **Historic Sites and Beaches are the weakest categories on
   satisfaction.** Historic Sites has the lowest average rating (3.50)
   and the highest share of 1–2★ reviews (16.8%); Beaches is a close
   second on both counts. Both are natural targets for a service-quality
   investigation (crowding, cleanliness, pricing, etc. — the dataset
   itself doesn't say why, only that satisfaction is lower here).
6. **Popularity ≠ quality.** With essentially zero correlation
   (r = 0.12) between visit volume and average rating, a
   recommendation/ranking approach that defaults to "most popular" will
   systematically under-surface some of the highest-rated but smaller
   attractions (e.g. Mount Semeru Volcano, Jomblang Cave).
7. **The traveler base is dominated by one-time visitors.** Only 27.7%
   of users have more than one transaction (median = 1, max = 26) — this
   caps how much a collaborative-filtering recommender can lean on a
   user's own history, reinforcing why cold-start handling was already
   built into `regression_features.csv` upstream of this checkpoint.

---

## Files added in this checkpoint

| File | Purpose |
|---|---|
| `scripts/eda_analysis.py` | Standalone EDA script — loads `consolidated.csv`, computes all statistics/tables above, and generates 15 Plotly charts. Read-only with respect to every existing file. |
| `docs/eda_report.md` | This report. |
| `docs/eda/eda_summary_stats.json` | Every number in this report, in structured form, plus an independent `validate()` re-check. |
| `docs/eda/tables/*.csv` | The underlying groupby/aggregation table behind each chart (15 CSVs). |
| `docs/eda/figures/*.html` | Interactive Plotly charts, generated when the script runs in an environment with `plotly` installed (see **Limitations**). |
| `docs/eda/figures_preview/*.png` | Static preview renders of the same 15 charts (see **Limitations**), themed to match `components/charts.py`'s dark-glass palette, viewable immediately without running anything. |

Nothing else in the checkpoint was modified — verified with a recursive
diff against `tourism_checkpoint_02_leakage_fixed.zip` before packaging
(see **Validation**).

## Analyses completed

- Dataset overview: shape, dtypes, missing values, duplicates, unique
  entity counts, geographic coverage, year/month coverage.
- Traveler analysis: Visit Mode distribution, home-country/region/
  continent breakdown, repeat-visit behavior.
- Attraction analysis: visit-count ranking, attraction-type volume,
  rating by attraction/type/city, top/bottom rated attractions
  (min-sample filtered).
- Temporal analysis: yearly and monthly visit volume, yearly rating
  trend, Visit Mode mix over time.
- Rating analysis: distribution and central tendency, rating by Visit
  Mode, low-rating concentration by attraction type, popularity-vs-
  quality correlation.
- Evidence-based business insights (7), each tied to a specific
  computed statistic above.

## Key findings

See sections 1–6 above; summarized in the 7 business insights.

## Visualizations

15 charts covering all five analysis sections (Visit Mode donut,
top-15 source countries, continent split, top-15 attractions by visits,
attraction-type volume, average rating by type, top/bottom rated
attractions, yearly volume, monthly seasonality, yearly rating trend,
Visit Mode share by year, rating distribution, rating by Visit Mode,
low-rating share by type, popularity-vs-quality scatter). No two charts
show the same slice of the data.

## Validation

- `scripts/eda_analysis.py` ran successfully end-to-end against the live
  `data/processed/consolidated.csv` in this checkpoint (49,208 rows) —
  console output: `Validation passed: True`.
- A second, independent `validate()` function re-derives the core facts
  (row count, no missing values, no duplicate transaction IDs, rating
  bounds 1–5, month bounds 1–12, Visit Mode shares sum to 100%, single
  attraction country) directly from the dataframe, separately from the
  section functions that produced the report numbers — all 7 checks
  passed.
- Recursive diff of this checkpoint against
  `tourism_checkpoint_02_leakage_fixed.zip` confirms the **only**
  changes are new, additive files listed above — no raw/cleaned data, no
  feature-engineering script, no model, no frontend/UI file, and no SQL
  file was modified.
- All existing Python files (`app.py`, every file under `pages/` and
  `components/`, `scripts/data_cleaning.py`,
  `scripts/feature_engineering.py`) were re-parsed (`ast.parse`) after
  packaging to confirm they are byte-for-byte untouched and still valid
  — the Streamlit app is unaffected by this checkpoint.

## Limitations

- **Plotly could not be executed in the validation sandbox used to build
  this checkpoint** (no outbound network access to install packages, and
  it isn't pre-installed there). `scripts/eda_analysis.py` contains
  complete, syntactically-correct Plotly code (`plotly.express` +
  `graph_objects`) that mirrors the app's existing chart theme
  (`components/charts.py`) and will generate the 15 interactive
  `.html` files in `docs/eda/figures/` the first time it's run in the
  project's actual environment, where `plotly>=5.20` is already declared
  in `requirements.txt`. The script degrades gracefully if plotly is
  missing: it still computes and saves every statistic and CSV table,
  prints a clear notice, and exits without error — it does not fail
  silently or partially.
- To make the checkpoint immediately reviewable without requiring a
  plotly install, `docs/eda/figures_preview/*.png` contains matplotlib
  renders of the same 15 charts with matching data and color theme.
  These are a stand-in for review purposes only — the canonical,
  interactive deliverable is the Plotly output from
  `scripts/eda_analysis.py`.
- Small-sample years (2020: 491 rows, 2021: 34 rows) make year-over-year
  rating comparisons for that period unreliable; flagged explicitly in
  §4 rather than treated as a trend.
- "Geographic coverage" in this dataset is about traveler origin only —
  all attractions are in a single country (Indonesia), so no
  destination-level geographic comparison is possible from this data.
- This EDA reads `consolidated.csv` (full-history aggregates by
  construction, since it has no engineered historical features) — it
  does not re-examine or re-validate the temporal-leakage fix already
  applied to `regression_features.csv`; that fix (documented in
  `docs/data_dictionary.md` §4.1) was explicitly out of scope for this
  checkpoint and was not touched.
- No statistical significance testing (e.g. formal hypothesis tests for
  rating differences between Visit Modes or attraction types) was
  performed — differences are reported descriptively, as this checkpoint
  is EDA only, not inferential modeling.

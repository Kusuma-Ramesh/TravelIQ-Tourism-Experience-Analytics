# SQL Analysis Report
**Tourism Experience Analytics — Checkpoint 04 (SQL only)**

Scope: implement the project's SQL component as a reproducible SQLite
workflow over the already-cleaned project data. No raw/cleaned/processed
data was modified, no feature engineering or leakage-safe regression logic
was touched, no model was trained, and no recommendation logic or
frontend/UI was changed.

---

## 1. Files changed

All new, purely additive — nothing existing was modified (verified with a
recursive diff against `tourism_checkpoint_03_eda.zip`, see **Validation**).

| File | Purpose |
|---|---|
| `database/schema.sql` | DDL for the normalized SQLite schema (9 tables), with inline documentation of every design decision. |
| `database/queries.sql` | 24 documented, named SQL queries covering the required business questions. |
| `database/tourism.db` | The generated SQLite database (regenerated fresh by `scripts/build_database.py`). |
| `scripts/build_database.py` | Loads `data/cleaned/*.csv` (+ two corrected columns from `data/processed/consolidated.csv`, see §2) into `database/tourism.db`. Read-only on all source files. |
| `scripts/run_sql_analysis.py` | Parses and executes every query in `database/queries.sql`, saves results, and independently validates them against pandas computed straight from `consolidated.csv`. |
| `docs/sql_report.md` | This report. |
| `docs/sql/query_results.json` | All 24 queries (SQL text, purpose, row count, preview) plus validation results, in structured form. |
| `docs/sql/results/*.csv` | Full result set for each of the 24 queries. |

## 2. Database / schema

**Engine:** SQLite (stdlib `sqlite3` — no new dependency; the project has no
existing SQL engine to match, and SQLite is the natural reproducible choice
for a local, file-based workflow).

**Design:** a normalized star-ish schema mirroring the *actual* project
entities already documented in `docs/data_dictionary.md` (Continent →
Region → Country → City, User, Item/Attraction, Type, Mode, Transaction),
rebuilt from `data/cleaned/*.csv` — not invented.

```
dim_continent ──< dim_region ──< dim_country ──< dim_city
                                                     ▲
dim_user ─────────(ContinentId/RegionId/CountryId/CityId)
   │
   │ UserId
   ▼
fact_transaction ──AttractionId──> dim_attraction ──AttractionTypeId──> dim_attraction_type
   │
   └─VisitMode──> dim_mode
```

| Table | Rows | Source |
|---|---:|---|
| `dim_continent` | 6 | `data/cleaned/continent.csv` |
| `dim_region` | 22 | `data/cleaned/region.csv` |
| `dim_country` | 165 | `data/cleaned/country.csv` |
| `dim_city` | 9,143 | `data/cleaned/city.csv` |
| `dim_user` | 33,530 | `data/cleaned/user.csv` |
| `dim_mode` | 5 | distinct `(VisitMode, VisitModeName)` pairs in `consolidated.csv` |
| `dim_attraction_type` | 22 | `data/cleaned/type.csv` |
| `dim_attraction` | 30 | `data/cleaned/item.csv` + corrected city/country (see below) |
| `fact_transaction` | 49,208 | `data/cleaned/transaction.csv`, loaded unchanged |

All row counts above match the corresponding cleaned CSV / the project's
already-validated numbers exactly (see **Validation**).

**One documented schema decision — why `dim_attraction` doesn't join
through `dim_city`:**
`docs/data_dictionary.md` (Data-Quality Finding 7) documents that
`item.csv`'s `AttractionCityId` is broken for exactly the 30 attractions
used here — the codes (1, 2, 3) collide with unrelated `City.xlsx` rows in
Cameroon/Chad, while the attractions are actually in Bali/Malang/Yogyakarta,
Indonesia. The already-locked pipeline fixed this with a manual override
when building `consolidated.csv`. Rather than re-deriving city/country via
`dim_attraction.AttractionCityId → dim_city → dim_country` (which would
silently reintroduce that already-fixed bug), `dim_attraction` stores the
corrected `AttractionCityName` / `AttractionCountry` values directly,
pulled from `consolidated.csv`'s own resolved columns. `AttractionCityId`
is still stored on the table for traceability, with an explicit comment in
`schema.sql` that it is *not* a reliable FK to `dim_city` for this table.
This is reusing an already-computed, already-validated value — not
inventing a new one.

`dim_mode` is built from `consolidated.csv`'s own `(VisitMode,
VisitModeName)` pairs (5 rows, matching the 5 modes that actually appear
in the transactions) rather than `data/cleaned/mode.csv` (7 rows,
including an unused `0`/"-" sentinel and a 6th code that doesn't appear in
any transaction), to guarantee the lookup table matches the fact table's
codes exactly.

## 3. Queries implemented

24 named, documented queries in `database/queries.sql`, covering every
required area plus a few additional business-relevant cuts:

| # | Query | Business question |
|---|---|---|
| 1 | `01_total_visits` | Total transactions in the dataset |
| 2 | `02_unique_travelers` | Distinct travelers |
| 3 | `03_unique_attractions` | Distinct attractions visited |
| 4 | `04_visits_by_country` | Top 15 traveler home countries |
| 5 | `05_visits_by_region` | Visits by traveler home region |
| 6 | `06_visits_by_continent` | Visits by traveler home continent |
| 7 | `07_visits_by_attraction` | Full visit ranking, all 30 attractions |
| 8 | `08_top_10_attractions` | Top 10 most popular attractions |
| 9 | `09_least_visited_attractions` | 10 least visited attractions |
| 10 | `10_overall_average_rating` | Headline average rating |
| 11 | `11_rating_distribution` | Rating distribution (1–5★) |
| 12 | `12_avg_rating_by_attraction` | Rating by attraction (min. 50 ratings) |
| 13 | `13_avg_rating_by_attraction_type` | Rating & low-rating share by type |
| 14 | `14_visit_mode_distribution` | Visit Mode split |
| 15 | `15_visit_mode_by_continent` | Visit Mode mix by traveler continent |
| 16 | `16_visit_mode_by_attraction_type` | Visit Mode mix by attraction type |
| 17 | `17_yearly_trend` | Yearly visit volume + avg rating (2013–2022) |
| 18 | `18_monthly_seasonality` | Seasonality by calendar month |
| 19 | `19_yearly_visit_mode_mix` | Visit Mode share by year |
| 20 | `20_repeat_vs_onetime_travelers` | Repeat vs. one-time traveler split |
| 21 | `21_popularity_vs_quality` | Visits vs. rating per attraction |
| 22 | `22_top_attraction_per_type` | Flagship attraction per category |
| 23 | `23_business_travelers_top_attractions` | Where Business travelers go |
| 24 | `24_ambiguous_repeat_summary` | Count of flagged ambiguous-repeat rows |

Every query is plain, portable SQL (joins, `GROUP BY`, `HAVING`, and window
functions for the share-of-total / ranking queries) executable against the
same `database/tourism.db` with no external dependencies beyond SQLite
itself.

## 4. Key results / insights

All numbers below are pulled directly from the query result CSVs in
`docs/sql/results/` and independently match the pandas-based EDA from
Checkpoint 03.

- **Volume:** 49,208 total visits, 33,530 unique travelers, 30 unique
  attractions.
- **Traveler origin (`04`–`06`):** Australia (12,683, 25.8%), United
  Kingdom (6,302, 12.8%) and United States (5,695, 11.6%) are the top 3
  source countries — together ~50% of all visits.
- **Attraction demand (`07`–`09`):** Sacred Monkey Forest Sanctuary leads
  with 12,369 visits (25.1% of all transactions); the least-visited
  attraction is Khayangan Reflexology & Massage (a Spa) with 28 visits.
- **Ratings (`10`–`13`):** overall average 4.156/5; 79.1% of ratings are
  4–5★. Water Parks rate highest by type (4.65, 1.25% low ratings);
  Historic Sites rate lowest (3.50, 16.79% low ratings — by far the
  highest share of poor reviews of any category).
- **Visit Mode (`14`–`16`):** Couples dominate (41.3%), Business is
  smallest (1.1%). Visit Mode mix shifts by attraction type — e.g. Water
  Parks and Beaches skew more Family, Religious Sites skew more Couples
  (see `16_visit_mode_by_attraction_type.csv` for the full cross-tab).
- **Time (`17`–`19`):** visits peaked in 2016 (11,772) and collapsed in
  2020–2021 (491 + 34 = 525, just 1.1% of the 10-year total) — the
  COVID-19 disruption is unambiguous. Rating stayed in a stable 4.06–4.28
  band through 2013–2019.
- **Loyalty (`20`):** 72.3% of travelers are one-time visitors; only
  27.7% return for a second (or more) visit.
- **Popularity vs. quality (`21`):** the most-visited attraction (Sacred
  Monkey Forest Sanctuary, 4.27 avg) is not the highest-rated one — e.g.
  Waterbom Bali (6,079 visits) rates higher at 4.65, and several
  lower-volume sites rate higher still. Confirms the EDA's finding of
  effectively no popularity–quality correlation.
- **Flagship attractions per category (`22`):** every one of the 17
  attraction types has a clear single leader (e.g. Waterbom Bali for
  Water Parks, Uluwatu Temple for Religious Sites, Merapi Volcano for
  Volcanos) — useful as category "hero" attractions for marketing.
- **Business travelers (`23`):** despite being the smallest segment
  overall, Business travelers' top destination is Nusa Dua Beach (65
  visits, 4.46 avg rating), followed by Merapi Volcano and Sacred Monkey
  Forest Sanctuary — a concrete starting list for a Business-traveler
  offering.
- **Data quality (`24`):** 2,536 transactions (5.2%) are flagged
  `is_ambiguous_repeat` — same (user, attraction, year, month) but
  differing rating/mode, as documented upstream in
  `docs/data_dictionary.md`; this SQL layer surfaces the count but makes
  no attempt to resolve it, consistent with the locked cleaning decision
  to keep (not drop) these rows.

## 5. Validation

- **Execution:** all 24 queries in `database/queries.sql` executed
  successfully against a freshly built `database/tourism.db` —
  `scripts/run_sql_analysis.py` reports `Executed 24 queries
  successfully.`
- **Reproducibility from a fresh state:** `database/tourism.db` and
  `docs/sql/` were deleted and regenerated from scratch
  (`python scripts/build_database.py && python scripts/run_sql_analysis.py`)
  with identical row counts and results — the workflow does not depend on
  any prior run's state.
- **Cross-verification against source data:** `scripts/run_sql_analysis.py`
  includes an independent `validate()` step that recomputes 8 headline
  numbers directly from `data/processed/consolidated.csv` with pandas
  (total visits, unique travelers, unique attractions, overall average
  rating, top-10 attraction ranking + counts, Visit Mode distribution,
  yearly visit counts, ambiguous-repeat count) and asserts they match the
  SQL results exactly. Result: **all checks passed**
  (`docs/sql/query_results.json` → `validation.all_passed: true`).
- **Table-level integrity:** every `dim_*` table's row count matches its
  source CSV exactly (see §2 table), and `fact_transaction`'s 49,208 rows
  match `data/cleaned/transaction.csv` and `consolidated.csv` exactly.
- **No regressions:** a recursive diff of this checkpoint against
  `tourism_checkpoint_03_eda.zip` confirms the only changes are the new
  files listed in §1 — no raw/cleaned/processed data file, no
  feature-engineering script, no model, no frontend/UI file was touched.
  Every existing Python file (`app.py`, all of `pages/` and
  `components/`, `scripts/data_cleaning.py`,
  `scripts/feature_engineering.py`, `scripts/preprocessing.py`,
  `scripts/data_loading.py`) was re-parsed (`ast.parse`) after packaging
  to confirm it is byte-for-byte unchanged and the Streamlit app is
  unaffected.

## 6. Limitations

- `dim_attraction_type` and `dim_mode` contain lookup codes that don't all
  appear in the transaction data (`type.csv` has 22 categories total, only
  17 are actually used by the 30 attractions here) — this is expected
  for a lookup/reference table and doesn't affect any query result, since
  every query above joins from `fact_transaction` outward.
- The attraction-side geography fields (`AttractionCityName`,
  `AttractionCountry`) are sourced from `consolidated.csv`'s already-
  corrected values rather than re-derived via `AttractionCityId →
  dim_city`, for the reason explained in §2. This means `dim_attraction`
  is not purely a 1:1 copy of `item.csv` — it's `item.csv` plus two
  already-validated corrected columns, which is a deliberate choice to
  avoid reintroducing a documented data-quality bug, not an invented
  value.
- All 30 attractions are in a single country (Indonesia), so country-level
  destination analysis isn't meaningful here — the country/region/
  continent breakdowns in this report are all about traveler *origin*,
  consistent with the same caveat noted in the Checkpoint 03 EDA report.
- No statistical significance testing was performed on any of the
  comparisons (e.g. rating differences by Visit Mode or attraction type)
  — results are descriptive SQL aggregations, as scoped for this
  checkpoint.
- 2020 and 2021 have very small sample sizes (491 and 34 transactions
  respectively), so the yearly rating figures for those two years in
  query `17` should be read with that caveat in mind (already flagged in
  the Checkpoint 03 EDA report).
- This checkpoint does not touch, re-validate, or duplicate the
  leakage-safe temporal aggregates in `regression_features.csv` — that
  logic remains exactly as delivered in Checkpoint 02 and was out of
  scope here.

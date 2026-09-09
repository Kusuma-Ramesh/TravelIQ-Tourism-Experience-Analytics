# Data Dictionary — Tourism Experience Analytics (Part 3)

## 1. Raw source tables

### Transaction.xlsx (52,930 rows × 7 cols) — fact table
| Column | Meaning | Type | Role |
|---|---|---|---|
| TransactionId | Unique visit/rating event id | int (PK) | identifier |
| UserId | Visitor id | int (FK → User) | join key |
| VisitYear | Year of visit (2013–2022) | int | feature |
| VisitMonth | Month of visit (1–12) | int | feature |
| VisitMode | Visit-mode code (1–5; 0="-" unused here) | int (FK → Mode) | feature / classification target (encoded) |
| AttractionId | Attraction visited | int (FK → Item) | join key |
| Rating | User's rating of the attraction | int (1–5) | regression target |

### User.xlsx (33,530 rows × 5 cols)
| Column | Meaning | Type | Role |
|---|---|---|---|
| UserId | Unique user id | int (PK) | identifier |
| ContinentId | User's home continent | int (FK → Continent) | demographic feature |
| RegionId | User's home region | int (FK → Region) | demographic feature |
| CountryId | User's home country | int (FK → Country) | demographic feature |
| CityId | User's home city | float→int (FK → City) | demographic feature |

### City.xlsx (9,143 rows × 3 cols)
| Column | Meaning | Type |
|---|---|---|
| CityId | Unique city id (0 = "-" sentinel/unknown) | int (PK) |
| CityName | City name | str |
| CountryId | Country the city belongs to | int (FK → Country) |

### Item.xlsx (30 rows × 5 cols) — canonical attraction table (exactly the attractions appearing in Transaction.xlsx)
| Column | Meaning | Type |
|---|---|---|
| AttractionId | Unique attraction id | int (PK) |
| AttractionCityId | Local city code — **see Data-Quality Finding 7**, not a reliable City.xlsx FK for these 30 rows | int |
| AttractionTypeId | Attraction category | int (FK → Type) |
| Attraction | Attraction name | str |
| AttractionAddress | Free-text address | str |

### Additional_Data_for_Attraction_Sites/Updated_Item.xlsx (1,698 rows × 5 cols) — extended attraction catalogue
Same columns as Item.xlsx, but covers 1,698 attractions worldwide (a superset — the same 30 rows from Item.xlsx are present, byte-for-byte identical, plus 1,668 more).
`AttractionTypeId` mixes numeric codes and free-text labels — see Finding 6.

### Type.xlsx (17 rows) | Mode.xlsx (6 rows) | Continent.xlsx (6 rows) | Country.xlsx (165 rows) | Region.xlsx (22 rows)
Standard lookup/dimension tables (Id + Name, plus a parent FK where applicable: Country→Region, Region→Continent). Each uses `0`/`"-"` as an explicit "unknown" sentinel category.

---

## 2. Relationship map

```
Continent (6) ←── Region (22) ←── Country (165) ←── City (9,143)
                                                        ▲
User (33,530) ───ContinentId/RegionId/CountryId/CityId──┘
   │
   │ UserId
   ▼
Transaction (52,930→49,208 after cleaning)
   │
   │ AttractionId                    VisitMode → Mode (6)
   ▼
Item (30, canonical) ──AttractionTypeId──► Type (17, →22 extended)
   │
   │ (AttractionId superset)
   ▼
Updated_Item (1,698, broader catalogue — used for content-based
              recommendation candidates, not joined into the
              transaction-level fact table)
```

Verified referential integrity (see Step 1/2 analysis): every FK check
(`transaction→user`, `transaction→item`, `item→type`, `user→city`,
`user→country`, `user→region`, `city→country`, `country→region`,
`region→continent`) returned **zero orphan keys**.

---

## 3. Data-quality findings

1. **Transaction duplicates.** 6,876 transaction rows are duplicated
   (different `TransactionId`, identical everything else) with another
   row; after `drop_duplicates`, 3,722 rows were removed. A further
   2,536 rows share the same (User, Attraction, Year, Month) but differ
   in `Rating`/`VisitMode` — kept, flagged `is_ambiguous_repeat=True`.
2. **User.CityId missing** for 4 users. Imputed with the dataset's own
   sentinel `CityId=0` ("-" / unknown).
3. **City.CityName missing** for 1 row (`CityId=6879`). Filled `"Unknown"`.
4. **Duplicate country name.** "Cyprus" appears as both `CountryId=76`
   (used by 16 cities / 35 users) and `CountryId=149` (used by nothing).
   Flagged, not deleted, since deleting a reference row could be risky
   even though it currently appears unused.
5. **RegionId=0 for 19 users** — not an error, it's the dataset's own
   "unknown region" placeholder; left as-is.
6. **`updated_item.AttractionTypeId` is inconsistently typed**: 30 rows
   hold numeric Type codes, 1,668 rows hold free-text labels ("Museum",
   "Temple", "Market", "Beach", "Park") that don't exist as IDs in
   `Type.xlsx` at all. Standardized into a clean numeric
   `AttractionTypeId` + readable `AttractionTypeName`, extending the
   Type dictionary with 5 new categories (IDs 901–905) for labels that
   are umbrella terms rather than exact matches of existing categories.
7. **Broken `AttractionCityId` → `City.xlsx` link for the 30 legacy
   attractions** (the exact set used by every transaction). Their
   `AttractionCityId` values (1, 2, 3) collide with unrelated
   `City.xlsx` rows (Douala/South Region/N'Djamena in Cameroon/Chad),
   while the attractions' own address text confirms they are all in
   Bali, Malang, and Yogyakarta, Indonesia. Spot checks confirm
   `AttractionCityId` resolves correctly against `City.xlsx` for the
   1,668 *other* attractions in `Updated_Item.xlsx`. Fixed with a
   manual override table for exactly those 30 `AttractionId`s
   (`AttractionCityName` → Bali/Malang/Yogyakarta, `AttractionCountry`
   → Indonesia); all other attractions use the normal City→Country
   join.
8. No invalid `Rating` (all 1–5), no invalid `VisitMode` (all resolve
   against `Mode.xlsx`, code 0/"-" never used), no invalid
   `VisitYear`/`VisitMonth` (2013–2022, 1–12) found in Transaction.xlsx.

---

## 4. Processed output files (`data/processed/`)

| File | Grain | Purpose |
|---|---|---|
| `consolidated.csv` | 1 row / transaction (49,208 rows) | Fully joined, human-readable base table |
| `regression_features.csv` | 1 row / transaction | Regression task: target = `Rating` |
| `classification_features.csv` | 1 row / transaction | Classification task: target = `VisitMode` |
| `recommendation_interactions.csv` | 1 row / (user, attraction) (45,275 rows) | Collaborative/content-based recommendation input |
| `user_profile_features.csv` | 1 row / user (33,530 rows) | Aggregated user behavior features |
| `attraction_profile_features.csv` | 1 row / attraction (30 rows) | Aggregated attraction popularity/quality stats |

`data/cleaned/*.csv` hold the cleaned-but-not-yet-joined version of every
raw table (raw `.xlsx` files in `data/raw/` are left untouched).

### 4.1 `regression_features.csv` — temporal leakage fix

`user_avg_rating`, `user_rating_std`, `user_distinct_attractions`,
`attraction_avg_rating`, `attraction_rating_std`, and
`attraction_distinct_users` in this file are **historical-as-of** values,
not full-dataset aggregates. They were previously computed by grouping
over the *entire* consolidated table, which meant a transaction's own
`Rating` (and every future rating from that user/attraction) leaked into
its own feature values. Fixed via calendar-month temporal atomicity:

- Every transaction gets a numeric period `VisitYear*12 + VisitMonth`.
- Ratings are aggregated per **(entity, period)** — e.g. per user per
  calendar month — not per transaction, so two transactions by the same
  user/attraction in the *same* month never see each other's ratings.
  `TransactionId` is never used to order or break ties.
- Per-entity aggregates are accumulated across periods and then shifted
  by one period, so the value attached to period `P` reflects only
  periods **strictly before `P`** — never the current month, never a
  future month.
- Historical distinct-attraction/-user counts use each pair's first-seen
  period: the historical distinct count at period `P` is the number of
  counterparts first encountered strictly before `P`.
- **Cold start**: a `has_user_history` / `has_attraction_history` flag
  (0/1) is added. When there is no strictly-earlier history, the average
  rating falls back to a fixed constant (`3.0`, the midpoint of the 1–5
  scale), std falls back to `0.0`, and the count/distinct fields are `0`
  — the same deterministic values every time, distinguishable from real
  history via the flag.

This fix is scoped to `regression_features.csv` only.
`classification_features.csv`, `recommendation_interactions.csv`,
`user_profile_features.csv`, and `attraction_profile_features.csv` are
unchanged and still use full-history aggregates (documented as a caveat
in `scripts/feature_engineering.py::add_user_aggregates` /
`add_attraction_aggregates` — to be revisited if those tasks need the
same treatment later).

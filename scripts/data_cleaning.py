"""
data_cleaning.py
-----------------
Implements the documented cleaning decisions for the Tourism Dataset.
Raw data is never modified in place -- every function takes a raw
DataFrame and returns a NEW cleaned DataFrame.

Cleaning decisions (see docs/data_dictionary.md and the Part-3 report
for full justification):

1. transaction:
   - 6,876 rows are exact duplicates of another transaction on
     (UserId, AttractionId, VisitYear, VisitMonth, VisitMode, Rating)
     -- only TransactionId differs. These are almost certainly
     duplicate log entries, not genuine repeat visits (a genuine
     repeat visit rarely reproduces an identical rating+mode in the
     same month). They are dropped, keeping the first occurrence.
   - A further 1,902 rows share (UserId, AttractionId, VisitYear,
     VisitMonth) but differ in Rating or VisitMode. These are
     ambiguous (could be a corrected re-rating, or two distinct
     visits in the same month). They are KEPT (not dropped, since we
     cannot be sure they are errors) but flagged with a new boolean
     column `is_ambiguous_repeat` so downstream modeling can decide
     whether to include them.
   - VisitYear, VisitMonth, VisitMode, Rating all already fall inside
     valid ranges (checked in Step 3) -- no value clipping needed.

2. user:
   - 4 rows have a missing CityId. Dropping them would lose valid
     transactions tied to those users, so CityId is imputed with the
     sentinel value 0, which already exists in the City table as the
     dataset's own "unknown city" placeholder (CityId=0, CityName='-',
     CountryId=0). This keeps the value in a known, joinable category
     instead of inventing a new one.
   - RegionId == 0 (19 rows) is NOT an error -- it is the dataset's
     existing "unknown region" placeholder (RegionId 0 = '-' in the
     Region table) and is left untouched.

3. city:
   - 1 row has a missing CityName. Filled with the literal string
     "Unknown" so the row remains usable in joins and group-bys
     without introducing a NaN that would silently drop rows in
     downstream aggregations.
   - CityId=0 / CountryId=0 is the dataset's own sentinel "unknown"
     row (CityName='-'), not a data error, and is preserved as-is.

4. country:
   - "Cyprus" appears twice (CountryId 76 and CountryId 149) with two
     different RegionId values. CountryId 149 is not referenced by
     any row in City or User, so it is very likely a stray/duplicate
     reference entry rather than a used category. It is NOT deleted
     (deleting reference rows is out of scope and risks hidden
     breakage), but it is flagged in a `flag_duplicate_name` column
     for transparency, and downstream lookups should prefer
     CountryId 76 for "Cyprus".

5. item (30 rows, canonical attractions referenced by every
   transaction):
   - No missing values, no duplicates, no invalid AttractionTypeId
     values (all resolve against the Type table). No cleaning needed
     beyond a column-name/dtype sanity pass.

6. updated_item (1,698-row extended attraction catalogue):
   - AttractionTypeId is stored inconsistently: 30 rows use the
     numeric codes from the Type table, but 1,668 rows use free-text
     category labels ("Museum", "Temple", "Market", "Beach", "Park")
     instead of an ID. This is standardized by mapping every row to
     BOTH a clean `AttractionTypeId` (numeric) and a clean
     `AttractionTypeName` (string), extending the Type dictionary
     with new IDs for labels that have no exact match in the
     original Type table ("Temple", "Museum", "Market", "Park" are
     umbrella categories, not exact synonyms of the existing
     "History Museums" / "Speciality Museums" / "National Parks" /
     "Water Parks" / "Flea & Street Markets" entries, so they are
     kept as distinct extended categories rather than force-mapped).

7. item / updated_item -- AttractionCityId FK is BROKEN for the 30
   legacy attractions (the exact set referenced by Transaction.xlsx):
   - AttractionCityId values 1, 2, 3 for these 30 rows do NOT point
     into City.xlsx's global id space. City.xlsx defines CityId 1/2/3
     as Douala, South Region, and N'Djamena (Cameroon/Chad), but the
     30 legacy attractions are verifiably all in Indonesia (their
     AttractionAddress text says "..., Kuta/Ubud/Denpasar ... Indonesia",
     "..., Malang ... Indonesia", "..., Yogyakarta ... Indonesia").
     This is a genuine legacy-ID collision: AttractionCityId 1/2/3 for
     these 30 rows are LOCAL codes (1=Bali area, 2=Malang area,
     3=Yogyakarta area) left over from a smaller, separate attraction
     dataset, never remapped when merged into the global city-id space
     used everywhere else in the dataset.
   - Spot checks on AttractionCityId >= 4 (the other 1,668 rows added
     in Updated_Item.xlsx) confirm those DO correctly resolve against
     City.xlsx (e.g. AttractionCityId=4 -> "Kigali", and the row's own
     address text says "Kigali, Rwanda"; AttractionCityId=12 ->
     "Shimoni", address says "Shimoni, Kenya"). So the fix only needs
     to special-case the 30 legacy AttractionIds.
   - Fix: a manual override map (LEGACY_ATTRACTION_CITY_OVERRIDE) is
     used for exactly those 30 AttractionIds to set the correct
     AttractionCityName / AttractionCountry ("Bali"/"Malang"/
     "Yogyakarta", all "Indonesia"), instead of joining their
     AttractionCityId against City.xlsx. All other attractions use the
     normal City.xlsx join.
"""

import pandas as pd


# ---------------------------------------------------------------------------
# Extended type mapping used to standardize updated_item.AttractionTypeId
# ---------------------------------------------------------------------------
# Original Type table uses IDs 2..93. New synthetic IDs (900+) are minted
# for the free-text labels found in updated_item that don't already have an
# exact equivalent, so the extended dictionary never collides with the
# original numeric codes.
EXTENDED_TYPE_LABELS = {
    "Museum": 901,
    "Temple": 902,
    "Market": 903,
    "Beach": 904,
    "Park": 905,
}


# ---------------------------------------------------------------------------
# Fix for the broken AttractionCityId -> City.xlsx link on the 30 legacy
# attractions (see module docstring, item 7). Keys are AttractionId.
# ---------------------------------------------------------------------------
LEGACY_ATTRACTION_CITY_OVERRIDE = {
    # AttractionCityId 1 -> Bali, Indonesia
    369: ("Bali", "Indonesia"), 481: ("Bali", "Indonesia"), 640: ("Bali", "Indonesia"),
    650: ("Bali", "Indonesia"), 673: ("Bali", "Indonesia"), 737: ("Bali", "Indonesia"),
    748: ("Bali", "Indonesia"), 749: ("Bali", "Indonesia"), 824: ("Bali", "Indonesia"),
    841: ("Bali", "Indonesia"),
    # AttractionCityId 2 -> Malang, Indonesia
    877: ("Malang", "Indonesia"), 888: ("Malang", "Indonesia"), 897: ("Malang", "Indonesia"),
    913: ("Malang", "Indonesia"), 920: ("Malang", "Indonesia"), 928: ("Malang", "Indonesia"),
    937: ("Malang", "Indonesia"), 947: ("Malang", "Indonesia"), 949: ("Malang", "Indonesia"),
    975: ("Malang", "Indonesia"),
    # AttractionCityId 3 -> Yogyakarta, Indonesia
    1133: ("Yogyakarta", "Indonesia"), 1137: ("Yogyakarta", "Indonesia"), 1166: ("Yogyakarta", "Indonesia"),
    1171: ("Yogyakarta", "Indonesia"), 1220: ("Yogyakarta", "Indonesia"), 1225: ("Yogyakarta", "Indonesia"),
    1238: ("Yogyakarta", "Indonesia"), 1278: ("Yogyakarta", "Indonesia"), 1280: ("Yogyakarta", "Indonesia"),
    1297: ("Yogyakarta", "Indonesia"),
}


def clean_transaction(transaction: pd.DataFrame) -> pd.DataFrame:
    df = transaction.copy()

    exact_dup_key = ["UserId", "AttractionId", "VisitYear", "VisitMonth", "VisitMode", "Rating"]
    before = len(df)
    df = df.drop_duplicates(subset=exact_dup_key, keep="first")
    dropped = before - len(df)
    print(f"[transaction] Dropped {dropped} exact duplicate rows "
          f"(same user/attraction/year/month/mode/rating, different TransactionId).")

    ambiguous_key = ["UserId", "AttractionId", "VisitYear", "VisitMonth"]
    df["is_ambiguous_repeat"] = df.duplicated(subset=ambiguous_key, keep=False)
    print(f"[transaction] Flagged {df['is_ambiguous_repeat'].sum()} rows as "
          f"is_ambiguous_repeat=True (same user/attraction/month, differing rating or mode).")

    df = df.reset_index(drop=True)
    return df


def clean_user(user: pd.DataFrame) -> pd.DataFrame:
    df = user.copy()
    missing_city = df["CityId"].isna().sum()
    df["CityId"] = df["CityId"].fillna(0).astype(int)
    print(f"[user] Imputed {missing_city} missing CityId values with sentinel CityId=0 (dataset's own 'unknown city').")
    return df


def clean_city(city: pd.DataFrame) -> pd.DataFrame:
    df = city.copy()
    missing_name = df["CityName"].isna().sum()
    df["CityName"] = df["CityName"].fillna("Unknown")
    print(f"[city] Filled {missing_name} missing CityName value(s) with 'Unknown'.")
    return df


def clean_country(country: pd.DataFrame) -> pd.DataFrame:
    df = country.copy()
    dup_names = df["Country"].duplicated(keep=False)
    df["flag_duplicate_name"] = dup_names
    n = dup_names.sum()
    print(f"[country] Flagged {n} row(s) sharing a duplicate Country name (e.g. 'Cyprus' listed under two CountryId values).")
    return df


def _apply_city_override(df: pd.DataFrame, city_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve AttractionCityName/AttractionCountry for every row: use the
    manual legacy override for the 30 known-broken AttractionIds, and the
    normal City.xlsx join (via AttractionCityId) for everything else."""
    city_lookup = city_df.set_index("CityId")[["CityName", "CountryId"]]
    country_id_to_name = None  # filled in by caller if needed; kept simple here

    city_names, countries = [], []
    for _, row in df.iterrows():
        aid = row["AttractionId"]
        if aid in LEGACY_ATTRACTION_CITY_OVERRIDE:
            city_name, country_name = LEGACY_ATTRACTION_CITY_OVERRIDE[aid]
        else:
            cid = row["AttractionCityId"]
            if cid in city_lookup.index:
                city_name = city_lookup.loc[cid, "CityName"]
            else:
                city_name = "Unknown"
            country_name = None  # resolved later via CountryId merge in preprocessing
        city_names.append(city_name)
        countries.append(country_name)

    df = df.copy()
    df["AttractionCityNameFixed"] = city_names
    df["AttractionCountryOverride"] = countries  # only set for the 30 legacy rows
    n_overridden = df["AttractionId"].isin(LEGACY_ATTRACTION_CITY_OVERRIDE).sum()
    print(f"[{df.attrs.get('_name', 'item')}] Corrected AttractionCityId->City mismatch for "
          f"{n_overridden} legacy attraction rows (Bali/Malang/Yogyakarta, all in Indonesia) "
          f"whose AttractionCityId collided with unrelated City.xlsx rows.")
    return df


def clean_item(item: pd.DataFrame, city_df: pd.DataFrame) -> pd.DataFrame:
    df = item.copy()
    df.attrs["_name"] = "item"
    df = _apply_city_override(df, city_df)
    return df


def clean_updated_item(updated_item: pd.DataFrame, type_df: pd.DataFrame, city_df: pd.DataFrame) -> pd.DataFrame:
    df = updated_item.copy()

    type_id_to_name = dict(zip(type_df["AttractionTypeId"], type_df["AttractionType"]))

    def resolve(raw_val):
        try:
            tid = int(raw_val)
            return tid, type_id_to_name.get(tid, "Unknown")
        except (ValueError, TypeError):
            label = str(raw_val).strip()
            tid = EXTENDED_TYPE_LABELS.get(label)
            if tid is None:
                # Unseen label: mint a new synthetic id on the fly (rare/fallback path)
                tid = 990 + abs(hash(label)) % 10
            return tid, label

    resolved = df["AttractionTypeId"].apply(resolve)
    df["AttractionTypeId"] = resolved.apply(lambda x: x[0]).astype(int)
    df["AttractionTypeName"] = resolved.apply(lambda x: x[1])

    n_relabeled = (~updated_item["AttractionTypeId"].astype(str).str.isnumeric()).sum()
    print(f"[updated_item] Standardized AttractionTypeId for {n_relabeled} rows that used free-text "
          f"labels instead of numeric codes; added AttractionTypeName for readability.")

    df.attrs["_name"] = "updated_item"
    df = _apply_city_override(df, city_df)
    return df


def build_extended_type_table(type_df: pd.DataFrame) -> pd.DataFrame:
    """Original Type table + the synthetic categories minted for updated_item."""
    extra_rows = pd.DataFrame(
        [{"AttractionTypeId": tid, "AttractionType": name} for name, tid in EXTENDED_TYPE_LABELS.items()]
    )
    extended = pd.concat([type_df.copy(), extra_rows], ignore_index=True)
    extended = extended.drop_duplicates(subset="AttractionTypeId").reset_index(drop=True)
    return extended


def clean_all(tables: dict) -> dict:
    """Apply all cleaning functions and return a new dict of cleaned DataFrames.
    Reference tables with no issues (mode, continent, region) pass through unchanged."""
    cleaned = {}
    cleaned["transaction"] = clean_transaction(tables["transaction"])
    cleaned["user"] = clean_user(tables["user"])
    cleaned["city"] = clean_city(tables["city"])
    cleaned["country"] = clean_country(tables["country"])
    cleaned["item"] = clean_item(tables["item"], cleaned["city"])
    cleaned["updated_item"] = clean_updated_item(tables["updated_item"], tables["type"], cleaned["city"])
    cleaned["type"] = build_extended_type_table(tables["type"])
    cleaned["mode"] = tables["mode"].copy()
    cleaned["continent"] = tables["continent"].copy()
    cleaned["region"] = tables["region"].copy()
    return cleaned


if __name__ == "__main__":
    import os
    from data_loading import load_all_raw

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    cleaned_dir = os.path.join(os.path.dirname(__file__), "..", "data", "cleaned")
    os.makedirs(cleaned_dir, exist_ok=True)

    tables = load_all_raw(raw_dir)
    cleaned = clean_all(tables)

    for name, df in cleaned.items():
        out_path = os.path.join(cleaned_dir, f"{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"[SAVE] {out_path}  shape={df.shape}")

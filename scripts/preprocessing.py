"""
preprocessing.py
-----------------
Builds the consolidated, analysis-ready dataset by joining the cleaned
tables. Only tables that are actually needed for the three ML tasks
(regression on Rating, classification of VisitMode, and user-item
recommendation) are joined -- see the docstring of `build_consolidated`
for the justification of each join.
"""

import os
import pandas as pd


def build_consolidated(cleaned: dict) -> pd.DataFrame:
    """
    Join order and rationale:

    1. transaction  (fact table: one row per visit/rating -- the spine)
    2. + user       (adds the visitor's Continent/Region/Country/City ids
                      -- needed as demographic features for regression &
                      classification)
    3. + city (as 'UserCity')   -> resolves the user's CityId to a name and
                      CountryId, used for city-level features and
                      distinguishing 'user home country' from 'attraction
                      country' if ever needed.
    4. + country (as 'UserCountry') / continent / region for the user's
                      location -- resolved through city.CountryId and
                      country.RegionId / region.ContinentId chains so the
                      full geographic hierarchy is available even though
                      user.xlsx already carries ContinentId/RegionId/
                      CountryId directly (used as the primary source; the
                      resolved names are added for readability/EDA).
    5. + item       (the canonical 30-attraction table -- guaranteed 100%
                      FK coverage against transaction.AttractionId, unlike
                      updated_item which is a broader catalogue not all of
                      which has been visited). Adds AttractionCityId,
                      AttractionTypeId, Attraction name, address, and the
                      pre-corrected AttractionCityNameFixed /
                      AttractionCountryOverride columns produced by
                      data_cleaning.clean_item (see its docstring, item 7:
                      AttractionCityId is NOT a reliable FK into City.xlsx
                      for these 30 rows, so the cleaned table already
                      carries the corrected city name and a manual country
                      override for them).
    6. + type       (resolves AttractionTypeId -> AttractionType name)
    7. AttractionCountry is resolved as: the manual override where present
                      (the 30 legacy rows, always "Indonesia"), otherwise
                      via the normal AttractionCityId -> City.xlsx ->
                      Country.xlsx chain (valid for every other attraction).
    8. + mode       (resolves VisitMode id -> readable VisitMode label)

    Tables NOT joined here:
    - updated_item: kept separate as a broader attraction catalogue for
      content-based / cold-start recommendation candidates. Merging its
      1,698 rows into the transaction-level fact table would not add
      information (only 30 of those attractions are ever transacted, and
      `item` already covers exactly those 30 cleanly).
    """
    transaction = cleaned["transaction"]
    user = cleaned["user"]
    city = cleaned["city"]
    country = cleaned["country"]
    region = cleaned["region"]
    continent = cleaned["continent"]
    item = cleaned["item"]
    type_df = cleaned["type"]
    mode = cleaned["mode"]

    df = transaction.merge(user, on="UserId", how="left", validate="many_to_one")

    # Resolve user's home geography chain: City -> Country -> Region -> Continent
    user_city = city.rename(columns={"CityId": "CityId", "CityName": "UserCityName", "CountryId": "CityCountryId"})
    df = df.merge(user_city[["CityId", "UserCityName"]], on="CityId", how="left")

    user_country = country.rename(columns={"CountryId": "CountryId", "Country": "UserCountry", "RegionId": "CountryRegionId"})
    df = df.merge(user_country[["CountryId", "UserCountry"]], on="CountryId", how="left")

    user_region = region.rename(columns={"RegionId": "RegionId", "Region": "UserRegion", "ContinentId": "RegionContinentId"})
    df = df.merge(user_region[["RegionId", "UserRegion"]], on="RegionId", how="left")

    user_continent = continent.rename(columns={"ContinentId": "ContinentId", "Continent": "UserContinent"})
    df = df.merge(user_continent[["ContinentId", "UserContinent"]], on="ContinentId", how="left")

    # Attraction attributes
    df = df.merge(item, on="AttractionId", how="left", validate="many_to_one")

    type_lookup = type_df.rename(columns={"AttractionType": "AttractionTypeName"})
    df = df.merge(type_lookup[["AttractionTypeId", "AttractionTypeName"]], on="AttractionTypeId", how="left")

    # AttractionCityName: item's cleaned AttractionCityNameFixed already
    # accounts for the 30 legacy rows whose raw AttractionCityId does not
    # resolve correctly against City.xlsx (see preprocessing docstring #5).
    df = df.rename(columns={"AttractionCityNameFixed": "AttractionCityName"})

    # AttractionCountry: use the manual override for the legacy rows; for
    # everything else, resolve the normal way through City.xlsx -> Country.xlsx.
    attraction_city = city.rename(columns={"CityId": "AttractionCityId", "CountryId": "AttractionCountryId"})
    df = df.merge(attraction_city[["AttractionCityId", "AttractionCountryId"]], on="AttractionCityId", how="left")
    attraction_country = country.rename(columns={"CountryId": "AttractionCountryId", "Country": "AttractionCountryResolved"})
    df = df.merge(attraction_country[["AttractionCountryId", "AttractionCountryResolved"]], on="AttractionCountryId", how="left")
    df["AttractionCountry"] = df["AttractionCountryOverride"].fillna(df["AttractionCountryResolved"])
    df = df.drop(columns=["AttractionCountryOverride", "AttractionCountryResolved", "AttractionCountryId"])

    mode_lookup = mode.rename(columns={"VisitModeId": "VisitMode", "VisitMode": "VisitModeName"})
    df = df.merge(mode_lookup, on="VisitMode", how="left")

    return df


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_loading import load_all_raw
    from data_cleaning import clean_all

    raw_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    processed_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    tables = load_all_raw(raw_dir)
    cleaned = clean_all(tables)
    consolidated = build_consolidated(cleaned)

    print(f"\nConsolidated dataset shape: {consolidated.shape}")
    print(f"Columns: {list(consolidated.columns)}")
    print(f"Missing values after join:\n{consolidated.isna().sum()[consolidated.isna().sum() > 0]}")

    out_path = os.path.join(processed_dir, "consolidated.csv")
    consolidated.to_csv(out_path, index=False)
    print(f"\n[SAVE] {out_path}  shape={consolidated.shape}")

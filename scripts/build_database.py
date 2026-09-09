"""
Build the SQLite database for the SQL analysis component (Checkpoint 04).

Reads ONLY from data/cleaned/*.csv and data/processed/consolidated.csv
(read-only — never writes back to either). Writes a fresh SQLite database
to database/tourism.db, structured per database/schema.sql.

This is a pure load step: no cleaning, no feature engineering, no
aggregation happens here. All computation lives in the SQL queries
(database/queries.sql), executed by scripts/run_sql_analysis.py.

Run:
    python scripts/build_database.py
"""

import os
import sqlite3

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "tourism.db")
SCHEMA_PATH = os.path.join(DB_DIR, "schema.sql")


def build_database() -> str:
    os.makedirs(DB_DIR, exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    try:
        with open(SCHEMA_PATH) as f:
            conn.executescript(f.read())

        continent = pd.read_csv(os.path.join(CLEANED_DIR, "continent.csv"))
        continent.columns = ["ContinentId", "ContinentName"]
        continent.to_sql("dim_continent", conn, if_exists="append", index=False)

        region = pd.read_csv(os.path.join(CLEANED_DIR, "region.csv"))
        # cleaned file column order is Region, RegionId, ContinentId
        region = region.rename(columns={"Region": "RegionName"})[
            ["RegionId", "RegionName", "ContinentId"]
        ]
        region.to_sql("dim_region", conn, if_exists="append", index=False)

        country = pd.read_csv(os.path.join(CLEANED_DIR, "country.csv"))
        country = country.rename(columns={"Country": "CountryName"})
        country["flag_duplicate_name"] = country["flag_duplicate_name"].astype(int)
        country = country[["CountryId", "CountryName", "RegionId", "flag_duplicate_name"]]
        country.to_sql("dim_country", conn, if_exists="append", index=False)

        city = pd.read_csv(os.path.join(CLEANED_DIR, "city.csv"))
        city.to_sql("dim_city", conn, if_exists="append", index=False)

        user = pd.read_csv(os.path.join(CLEANED_DIR, "user.csv"))
        user.to_sql("dim_user", conn, if_exists="append", index=False)

        attraction_type = pd.read_csv(os.path.join(CLEANED_DIR, "type.csv"))
        attraction_type = attraction_type.rename(columns={"AttractionType": "AttractionTypeName"})
        attraction_type.to_sql("dim_attraction_type", conn, if_exists="append", index=False)

        # VisitMode lookup: derive from consolidated.csv's own (VisitMode,
        # VisitModeName) pairs so it is guaranteed consistent with the fact
        # table codes used below (rather than trusting mode.csv's own id
        # ordering matches Transaction.VisitMode 1:1 by coincidence).
        consolidated = pd.read_csv(os.path.join(PROCESSED_DIR, "consolidated.csv"))
        mode_lookup = (
            consolidated[["VisitMode", "VisitModeName"]]
            .drop_duplicates()
            .rename(columns={"VisitMode": "VisitModeId"})
            .sort_values("VisitModeId")
        )
        mode_lookup.to_sql("dim_mode", conn, if_exists="append", index=False)

        # Attraction dimension: base columns from item.csv, corrected
        # city/country columns from consolidated.csv (see schema.sql header
        # for why — Data-Quality Finding 7).
        item = pd.read_csv(os.path.join(CLEANED_DIR, "item.csv"))
        attraction_geo = (
            consolidated[["AttractionId", "AttractionCityName", "AttractionCountry"]]
            .drop_duplicates()
        )
        attraction = item.merge(attraction_geo, on="AttractionId", how="left")
        attraction = attraction[
            [
                "AttractionId",
                "Attraction",
                "AttractionAddress",
                "AttractionTypeId",
                "AttractionCityId",
                "AttractionCityName",
                "AttractionCountry",
            ]
        ]
        attraction.to_sql("dim_attraction", conn, if_exists="append", index=False)

        transaction = pd.read_csv(os.path.join(CLEANED_DIR, "transaction.csv"))
        transaction["is_ambiguous_repeat"] = transaction["is_ambiguous_repeat"].astype(int)
        transaction.to_sql("fact_transaction", conn, if_exists="append", index=False)

        conn.commit()

        counts = {}
        for table in [
            "dim_continent",
            "dim_region",
            "dim_country",
            "dim_city",
            "dim_user",
            "dim_mode",
            "dim_attraction_type",
            "dim_attraction",
            "fact_transaction",
        ]:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        return counts
    finally:
        conn.close()


if __name__ == "__main__":
    counts = build_database()
    print(f"Database built at {DB_PATH}")
    for table, n in counts.items():
        print(f"  {table}: {n:,} rows")

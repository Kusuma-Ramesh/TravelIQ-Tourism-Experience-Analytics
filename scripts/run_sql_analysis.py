"""
Run every query in database/queries.sql against database/tourism.db,
save the results, and validate them against an independent pandas
computation over the same source data.

Scope: SQL ONLY. This script does not modify any raw/cleaned/processed
data, does not touch feature engineering, and does not train any model.

Outputs:
  docs/sql/results/<query_id>.csv   -> full result set per query
  docs/sql/query_results.json       -> every query's SQL, purpose, row
                                        count and preview, plus validation

Run (after scripts/build_database.py has created database/tourism.db):
    python scripts/run_sql_analysis.py
"""

import json
import os
import re
import sqlite3

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "database", "tourism.db")
QUERIES_PATH = os.path.join(BASE_DIR, "database", "queries.sql")
OUT_DIR = os.path.join(BASE_DIR, "docs", "sql")
RESULTS_DIR = os.path.join(OUT_DIR, "results")
CONSOLIDATED_PATH = os.path.join(BASE_DIR, "data", "processed", "consolidated.csv")

QUERY_BLOCK_RE = re.compile(
    r"-- @id:\s*(?P<id>\S+)\s*\n"
    r"-- @title:\s*(?P<title>.+?)\s*\n"
    r"(?:-- @purpose:\s*(?P<purpose>.+?)(?=\n[^-]|\nSELECT)\n)?",
    re.DOTALL,
)


def parse_queries(sql_text: str) -> list[dict]:
    """Split queries.sql into named blocks using the '-- @id:' headers."""
    parts = re.split(r"(?=-- @id:)", sql_text)
    queries = []
    for part in parts:
        part = part.strip()
        if not part.startswith("-- @id:"):
            continue
        id_match = re.search(r"-- @id:\s*(\S+)", part)
        title_match = re.search(r"-- @title:\s*(.+)", part)
        purpose_lines = re.findall(r"-- @purpose:(.*?)(?=\n-- @|\n[A-Z])", part, re.DOTALL)
        # purpose can span multiple '--' comment lines; collect all leading
        # comment lines after '@purpose:' until the first non-comment line
        purpose = ""
        lines = part.splitlines()
        collecting = False
        purpose_chunks = []
        sql_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("-- @purpose:"):
                collecting = True
                purpose_chunks.append(stripped.replace("-- @purpose:", "").strip())
                continue
            if collecting and stripped.startswith("--"):
                purpose_chunks.append(stripped.lstrip("-").strip())
                continue
            collecting = False
            if stripped.startswith("--") or stripped.startswith("-- @"):
                continue
            if stripped:
                sql_lines.append(line)
        purpose = " ".join(purpose_chunks).strip()
        sql = "\n".join(sql_lines).strip()

        if id_match and sql:
            queries.append(
                {
                    "id": id_match.group(1),
                    "title": title_match.group(1).strip() if title_match else "",
                    "purpose": purpose,
                    "sql": sql,
                }
            )
    return queries


def run_queries() -> list[dict]:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(QUERIES_PATH) as f:
        sql_text = f.read()
    queries = parse_queries(sql_text)

    conn = sqlite3.connect(DB_PATH)
    results = []
    try:
        for q in queries:
            df = pd.read_sql_query(q["sql"], conn)
            df.to_csv(os.path.join(RESULTS_DIR, f"{q['id']}.csv"), index=False)
            results.append(
                {
                    "id": q["id"],
                    "title": q["title"],
                    "purpose": q["purpose"],
                    "sql": q["sql"],
                    "row_count": len(df),
                    "columns": list(df.columns),
                    "preview": df.head(10).to_dict(orient="records"),
                }
            )
    finally:
        conn.close()
    return results


def validate(results_by_id: dict) -> dict:
    """Independently recompute a handful of headline numbers with pandas,
    straight from the same source CSVs the database was built from, and
    compare them to what the SQL queries returned. This catches join/
    aggregation bugs that would otherwise look self-consistent."""
    consolidated = pd.read_csv(CONSOLIDATED_PATH)
    checks = {}

    sql_total = results_by_id["01_total_visits"]["preview"][0]["total_visits"]
    checks["total_visits_matches_consolidated"] = sql_total == len(consolidated)

    sql_users = results_by_id["02_unique_travelers"]["preview"][0]["unique_travelers"]
    checks["unique_travelers_matches_consolidated"] = sql_users == consolidated["UserId"].nunique()

    sql_attractions = results_by_id["03_unique_attractions"]["preview"][0]["unique_attractions_visited"]
    checks["unique_attractions_matches_consolidated"] = (
        sql_attractions == consolidated["AttractionId"].nunique()
    )

    sql_avg_rating = results_by_id["10_overall_average_rating"]["preview"][0]["avg_rating"]
    checks["overall_avg_rating_matches_consolidated"] = (
        abs(sql_avg_rating - round(consolidated["Rating"].mean(), 3)) < 0.001
    )

    sql_top10 = pd.read_csv(os.path.join(RESULTS_DIR, "08_top_10_attractions.csv"))
    pandas_top10 = (
        consolidated.groupby("Attraction").size().sort_values(ascending=False).head(10)
    )
    checks["top_10_attractions_match_consolidated"] = list(sql_top10["attraction"]) == list(
        pandas_top10.index
    )
    checks["top_10_attraction_visit_counts_match"] = list(sql_top10["total_visits"]) == list(
        pandas_top10.values
    )

    sql_mode_dist = pd.read_csv(os.path.join(RESULTS_DIR, "14_visit_mode_distribution.csv"))
    pandas_mode_dist = consolidated["VisitModeName"].value_counts()
    checks["visit_mode_distribution_matches_consolidated"] = all(
        int(sql_mode_dist.loc[sql_mode_dist["visit_mode"] == mode, "total_visits"].iloc[0])
        == int(count)
        for mode, count in pandas_mode_dist.items()
    )

    sql_yearly = pd.read_csv(os.path.join(RESULTS_DIR, "17_yearly_trend.csv"))
    pandas_yearly = consolidated.groupby("VisitYear").size()
    checks["yearly_visit_counts_match_consolidated"] = all(
        int(sql_yearly.loc[sql_yearly["VisitYear"] == year, "total_visits"].iloc[0]) == int(count)
        for year, count in pandas_yearly.items()
    )

    sql_ambiguous = pd.read_csv(os.path.join(RESULTS_DIR, "24_ambiguous_repeat_summary.csv"))
    pandas_ambiguous_true = int(consolidated["is_ambiguous_repeat"].sum())
    sql_ambiguous_true = int(
        sql_ambiguous.loc[sql_ambiguous["is_ambiguous_repeat"] == 1, "n_transactions"].iloc[0]
    )
    checks["ambiguous_repeat_count_matches_consolidated"] = sql_ambiguous_true == pandas_ambiguous_true

    checks["all_passed"] = all(bool(v) for v in checks.values())
    return checks


def main() -> None:
    results = run_queries()
    results_by_id = {r["id"]: r for r in results}
    checks = validate(results_by_id)

    summary = {
        "n_queries": len(results),
        "queries": results,
        "validation": checks,
    }
    with open(os.path.join(OUT_DIR, "query_results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Executed {len(results)} queries successfully.")
    print(f"Validation passed: {checks['all_passed']}")
    if not checks["all_passed"]:
        failed = {k: v for k, v in checks.items() if not v and k != "all_passed"}
        print("FAILED CHECKS:", failed)
    print(f"Results: {RESULTS_DIR}")
    print(f"Summary: {os.path.join(OUT_DIR, 'query_results.json')}")


if __name__ == "__main__":
    main()

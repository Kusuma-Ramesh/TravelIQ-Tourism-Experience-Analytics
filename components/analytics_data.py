"""
Real analytics data access for the Tourism Analytics page.

This module ONLY reads artifacts that already exist on disk from the
completed, leakage-safe EDA pipeline (Checkpoint 08 and earlier):

    docs/eda/eda_summary_stats.json   -- pre-computed summary statistics
    docs/eda/tables/*.csv             -- pre-computed EDA aggregate tables

No EDA is recomputed here, no ML models are touched, and the raw/
cleaned/processed data pipeline is not modified. This is intentionally
a thin, cached read layer -- the same pattern components/world_map.py
already uses for the Home page (st.cache_data + a fixed path under
data/ or docs/).
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
EDA_DIR = ROOT / "docs" / "eda"
SUMMARY_PATH = EDA_DIR / "eda_summary_stats.json"
TABLES_DIR = EDA_DIR / "tables"

MONTH_NAMES = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


@st.cache_data(show_spinner=False)
def load_summary() -> dict:
    """The full pre-computed EDA summary (dataset_overview, traveler_analysis,
    attraction_analysis, temporal_analysis, rating_analysis, business_insights)."""
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_table(name: str) -> pd.DataFrame:
    """Load one pre-computed EDA aggregate table by filename (without .csv)."""
    path = TABLES_DIR / f"{name}.csv"
    df = pd.read_csv(path)
    unnamed = [c for c in df.columns if c.startswith("Unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)
    return df


def kpis(summary: dict) -> dict:
    """Headline KPI values, straight from the validated dataset_overview /
    rating_analysis blocks -- nothing here is hardcoded or invented."""
    overview = summary["dataset_overview"]
    rating = summary["rating_analysis"]
    return {
        "total_transactions": overview["n_rows"],
        "unique_travelers": overview["n_unique_users"],
        "n_attractions": overview["n_unique_attractions"],
        "avg_rating": rating["rating_mean"],
        "pct_4_or_5": rating["pct_rating_4_or_5"],
        "n_countries": overview["n_unique_user_countries"],
        "n_attraction_types": overview["n_attraction_types"],
        "year_min": overview["visit_year_min"],
        "year_max": overview["visit_year_max"],
    }


def ordered_month_counts(monthly_counts: dict) -> tuple[list[str], list[float]]:
    """monthly_visit_counts keys are '1'..'12' strings in an arbitrary dict
    order -- return them sorted Jan->Dec with month-name labels."""
    items = sorted(monthly_counts.items(), key=lambda kv: int(kv[0]))
    labels = [MONTH_NAMES[int(k) - 1] for k, _ in items]
    values = [v for _, v in items]
    return labels, values


def ordered_year_counts(yearly_counts: dict) -> tuple[list[str], list[float]]:
    items = sorted(yearly_counts.items(), key=lambda kv: int(kv[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    return labels, values

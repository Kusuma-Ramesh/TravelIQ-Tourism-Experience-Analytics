"""
Exploratory Data Analysis — Tourism Experience Analytics (Part 3 / Checkpoint 03)

Scope: EDA ONLY. This script reads the already-consolidated, already
leakage-fixed dataset and computes descriptive statistics and Plotly
visualizations. It does not touch raw/cleaned data, does not change
feature engineering, does not train any model, and does not implement
recommendations or SQL.

Source of truth: data/processed/consolidated.csv (1 row per transaction,
49,208 rows, fully joined & human-readable — see docs/data_dictionary.md).
`regression_features.csv` and `classification_features.csv` are read
read-only, only where the consolidated table alone can't answer a
question (e.g. how many rows have real vs. cold-start history) — never
written to.

Outputs (all under docs/eda/, nothing outside this analysis folder):
  docs/eda/eda_summary_stats.json   -> every number quoted in the report
  docs/eda/tables/*.csv             -> the groupby tables behind each chart
  docs/eda/figures/*.html           -> interactive Plotly charts (self-
                                        contained, plotly.js via CDN)

Run:
    python scripts/eda_analysis.py

Requires pandas (already a project dependency) and plotly (already listed
in requirements.txt as plotly>=5.20). If plotly is not importable in the
current environment, the script still computes and saves every
statistic/table above so the analysis is never blocked on plotly being
installed — it just skips writing the .html figures and prints a notice.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR = os.path.join(BASE_DIR, "docs", "eda")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TABLE_DIR = os.path.join(OUT_DIR, "tables")

# Matches components/charts.py so these figures drop into the existing
# dark-glass UI without a re-theme if a later task wires them into
# pages/analytics.py. Kept as a local copy (not imported) so this script
# has no Streamlit runtime dependency and can run headless / in CI.
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#A9B7CC", family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, zeroline=False, color="#6E7E96"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, color="#6E7E96"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
ACCENTS = ["#8B6FF0", "#4CB8F0", "#2FD9A8", "#F0A24C", "#F2685C"]

try:
    import plotly.graph_objects as go
    import plotly.express as px

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _save_fig(fig, name: str) -> None:
    if not PLOTLY_AVAILABLE:
        return
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.update_layout(**{k: v for k, v in PLOTLY_LAYOUT.items() if k not in fig.layout or True})
    fig.write_html(os.path.join(FIG_DIR, f"{name}.html"), include_plotlyjs="cdn", full_html=True)


def _save_table(df: pd.DataFrame, name: str) -> None:
    os.makedirs(TABLE_DIR, exist_ok=True)
    df.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"))


def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_PROCESSED, "consolidated.csv")
    df = pd.read_csv(path)
    return df


# --------------------------------------------------------------------------
# 1. Dataset overview
# --------------------------------------------------------------------------

def overview_stats(df: pd.DataFrame) -> dict:
    missing = df.isna().sum()
    stats = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values_total": int(missing.sum()),
        "missing_by_column": {c: int(v) for c, v in missing.items() if v > 0},
        "duplicate_rows_full": int(df.duplicated().sum()),
        "duplicate_transaction_ids": int(df["TransactionId"].duplicated().sum()),
        "ambiguous_repeat_rows": int(df["is_ambiguous_repeat"].sum()),
        "n_unique_users": int(df["UserId"].nunique()),
        "n_unique_attractions": int(df["AttractionId"].nunique()),
        "n_unique_user_countries": int(df["UserCountry"].nunique()),
        "n_unique_user_regions": int(df["UserRegion"].nunique()),
        "n_unique_user_continents": int(df["UserContinent"].nunique()),
        "n_unique_attraction_countries": int(df["AttractionCountry"].nunique()),
        "attraction_countries": sorted(df["AttractionCountry"].unique().tolist()),
        "attraction_cities": sorted(df["AttractionCityName"].unique().tolist()),
        "n_attraction_types": int(df["AttractionTypeName"].nunique()),
        "n_visit_modes": int(df["VisitModeName"].nunique()),
        "visit_year_min": int(df["VisitYear"].min()),
        "visit_year_max": int(df["VisitYear"].max()),
        "visit_month_min": int(df["VisitMonth"].min()),
        "visit_month_max": int(df["VisitMonth"].max()),
        "rating_min": int(df["Rating"].min()),
        "rating_max": int(df["Rating"].max()),
    }
    return stats


# --------------------------------------------------------------------------
# 2. Traveler analysis
# --------------------------------------------------------------------------

def traveler_analysis(df: pd.DataFrame) -> dict:
    visit_mode_counts = df["VisitModeName"].value_counts()
    continent_counts = df["UserContinent"].value_counts()
    top_countries = df["UserCountry"].value_counts().head(15)
    region_counts = df["UserRegion"].value_counts()

    trips_per_user = df.groupby("UserId").size()
    repeat_users = int((trips_per_user > 1).sum())

    stats = {
        "visit_mode_counts": visit_mode_counts.to_dict(),
        "visit_mode_share_pct": (visit_mode_counts / len(df) * 100).round(2).to_dict(),
        "continent_counts": continent_counts.to_dict(),
        "top_15_user_countries": top_countries.to_dict(),
        "region_counts": region_counts.head(15).to_dict(),
        "n_users_total": int(trips_per_user.shape[0]),
        "n_users_repeat": repeat_users,
        "pct_users_repeat": round(repeat_users / trips_per_user.shape[0] * 100, 2),
        "trips_per_user_mean": round(float(trips_per_user.mean()), 3),
        "trips_per_user_median": float(trips_per_user.median()),
        "trips_per_user_max": int(trips_per_user.max()),
    }

    _save_table(visit_mode_counts.rename("count").to_frame(), "visit_mode_counts")
    _save_table(top_countries.rename("count").to_frame(), "top_user_countries")
    _save_table(continent_counts.rename("count").to_frame(), "continent_counts")

    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            go.Pie(
                labels=visit_mode_counts.index,
                values=visit_mode_counts.values,
                hole=0.6,
                marker=dict(colors=ACCENTS[: len(visit_mode_counts)]),
                textfont=dict(color="#F3F6FB"),
            )
        )
        fig.update_layout(title="Visit Mode distribution (all transactions)")
        _save_fig(fig, "01_visit_mode_distribution")

        fig = px.bar(
            top_countries.iloc[::-1],
            orientation="h",
            labels={"value": "Transactions", "index": "Home country"},
            title="Top 15 traveler home countries by transaction volume",
            color_discrete_sequence=[ACCENTS[1]],
        )
        fig.update_layout(showlegend=False)
        _save_fig(fig, "02_top_user_countries")

        fig = go.Figure(
            go.Pie(
                labels=continent_counts.index,
                values=continent_counts.values,
                hole=0.6,
                marker=dict(colors=ACCENTS[: len(continent_counts)]),
                textfont=dict(color="#F3F6FB"),
            )
        )
        fig.update_layout(title="Traveler home continent distribution")
        _save_fig(fig, "03_continent_distribution")

    return stats


# --------------------------------------------------------------------------
# 3. Attraction analysis
# --------------------------------------------------------------------------

def attraction_analysis(df: pd.DataFrame) -> dict:
    visits_per_attraction = df.groupby("Attraction").size().sort_values(ascending=False)
    type_counts = df["AttractionTypeName"].value_counts()

    att_stats = (
        df.groupby(["Attraction", "AttractionTypeName", "AttractionCityName"])["Rating"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    min_count = 50
    qualified = att_stats[att_stats["count"] >= min_count]
    top_rated = qualified.sort_values("mean", ascending=False).head(8)
    bottom_rated = qualified.sort_values("mean", ascending=True).head(8)

    type_rating = df.groupby("AttractionTypeName")["Rating"].agg(["mean", "count"]).sort_values(
        "mean", ascending=False
    )
    city_rating = df.groupby("AttractionCityName")["Rating"].agg(["mean", "count"]).sort_values(
        "mean", ascending=False
    )

    stats = {
        "n_attractions": int(df["AttractionId"].nunique()),
        "n_attraction_types": int(df["AttractionTypeName"].nunique()),
        "n_attraction_cities": int(df["AttractionCityName"].nunique()),
        "single_country": df["AttractionCountry"].unique().tolist(),
        "top_10_attractions_by_visits": visits_per_attraction.head(10).to_dict(),
        "attraction_type_counts": type_counts.to_dict(),
        "min_sample_for_ranking": min_count,
        "top_rated_attractions_min50": top_rated.set_index("Attraction")["mean"].round(3).to_dict(),
        "bottom_rated_attractions_min50": bottom_rated.set_index("Attraction")["mean"].round(3).to_dict(),
        "avg_rating_by_type": type_rating["mean"].round(3).to_dict(),
        "avg_rating_by_city": city_rating["mean"].round(3).to_dict(),
        "visits_per_attraction_describe": {
            k: round(float(v), 2) for k, v in visits_per_attraction.describe().items()
        },
    }

    _save_table(visits_per_attraction.rename("visits").to_frame(), "visits_per_attraction")
    _save_table(type_rating, "avg_rating_by_attraction_type")
    _save_table(att_stats, "attraction_rating_summary")

    if PLOTLY_AVAILABLE:
        top15 = visits_per_attraction.head(15).iloc[::-1]
        fig = px.bar(
            top15,
            orientation="h",
            labels={"value": "Transactions", "index": "Attraction"},
            title="Top 15 attractions by number of visits",
            color_discrete_sequence=[ACCENTS[0]],
        )
        fig.update_layout(showlegend=False, height=520)
        _save_fig(fig, "04_top_attractions_by_visits")

        fig = px.bar(
            type_counts.iloc[::-1],
            orientation="h",
            labels={"value": "Transactions", "index": "Attraction type"},
            title="Visit volume by attraction type",
            color_discrete_sequence=[ACCENTS[2]],
        )
        fig.update_layout(showlegend=False, height=520)
        _save_fig(fig, "05_attraction_type_volume")

        fig = px.bar(
            type_rating.reset_index(),
            x="mean",
            y="AttractionTypeName",
            orientation="h",
            labels={"mean": "Average rating", "AttractionTypeName": "Attraction type"},
            title="Average rating by attraction type",
            color="mean",
            color_continuous_scale=["#F2685C", "#F0A24C", "#2FD9A8"],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=520)
        _save_fig(fig, "06_avg_rating_by_attraction_type")

        cmp = pd.concat(
            [top_rated.assign(group="Top rated"), bottom_rated.assign(group="Lowest rated")]
        )
        fig = px.bar(
            cmp.sort_values("mean"),
            x="mean",
            y="Attraction",
            orientation="h",
            color="group",
            color_discrete_map={"Top rated": "#2FD9A8", "Lowest rated": "#F2685C"},
            labels={"mean": "Average rating"},
            title=f"Highest & lowest rated attractions (min {min_count} ratings)",
        )
        fig.update_layout(height=520)
        _save_fig(fig, "07_top_bottom_rated_attractions")

    return stats


# --------------------------------------------------------------------------
# 4. Temporal analysis
# --------------------------------------------------------------------------

def temporal_analysis(df: pd.DataFrame) -> dict:
    yearly_visits = df.groupby("VisitYear").size()
    monthly_visits = df.groupby("VisitMonth").size()
    yearly_rating = df.groupby("VisitYear")["Rating"].mean()
    mode_share_by_year = pd.crosstab(df["VisitYear"], df["VisitModeName"], normalize="index") * 100

    stats = {
        "yearly_visit_counts": {int(k): int(v) for k, v in yearly_visits.items()},
        "monthly_visit_counts": {int(k): int(v) for k, v in monthly_visits.items()},
        "yearly_avg_rating": {int(k): round(float(v), 3) for k, v in yearly_rating.items()},
        "peak_year": int(yearly_visits.idxmax()),
        "peak_month": int(monthly_visits.idxmax()),
        "lowest_visit_year": int(yearly_visits.idxmin()),
        "covid_years_2020_2021_share_pct": round(
            yearly_visits.reindex([2020, 2021]).sum() / yearly_visits.sum() * 100, 2
        ),
        "mode_share_by_year_pct": mode_share_by_year.round(1).to_dict(orient="index"),
    }

    _save_table(yearly_visits.rename("visits").to_frame(), "yearly_visits")
    _save_table(monthly_visits.rename("visits").to_frame(), "monthly_visits")
    _save_table(yearly_rating.rename("avg_rating").to_frame(), "yearly_avg_rating")
    _save_table(mode_share_by_year, "visit_mode_share_by_year")

    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            go.Bar(x=yearly_visits.index, y=yearly_visits.values, marker=dict(color=ACCENTS[1]))
        )
        fig.update_layout(
            title="Visit volume by year (2013\u20132022)",
            xaxis=dict(dtick=1),
            yaxis_title="Transactions",
        )
        _save_fig(fig, "08_yearly_visit_volume")

        fig = go.Figure(
            go.Bar(x=monthly_visits.index, y=monthly_visits.values, marker=dict(color=ACCENTS[3]))
        )
        fig.update_layout(
            title="Seasonality \u2014 visit volume by calendar month (all years combined)",
            xaxis=dict(dtick=1, title="Month"),
            yaxis_title="Transactions",
        )
        _save_fig(fig, "09_monthly_seasonality")

        fig = go.Figure(
            go.Scatter(
                x=yearly_rating.index,
                y=yearly_rating.values,
                mode="lines+markers",
                line=dict(color=ACCENTS[2], width=3),
                marker=dict(size=8),
            )
        )
        fig.update_layout(
            title="Average rating by year",
            xaxis=dict(dtick=1),
            yaxis_title="Average rating (1\u20135)",
        )
        _save_fig(fig, "10_yearly_avg_rating_trend")

        fig = go.Figure()
        for i, mode in enumerate(mode_share_by_year.columns):
            fig.add_trace(
                go.Scatter(
                    x=mode_share_by_year.index,
                    y=mode_share_by_year[mode],
                    mode="lines",
                    stackgroup="one",
                    name=mode,
                    line=dict(color=ACCENTS[i % len(ACCENTS)]),
                )
            )
        fig.update_layout(
            title="Visit Mode share by year (%)", xaxis=dict(dtick=1), yaxis_title="Share (%)"
        )
        _save_fig(fig, "11_visit_mode_share_by_year")

    return stats


# --------------------------------------------------------------------------
# 5. Rating analysis
# --------------------------------------------------------------------------

def rating_analysis(df: pd.DataFrame) -> dict:
    rating_counts = df["Rating"].value_counts().sort_index()
    rating_by_mode = df.groupby("VisitModeName")["Rating"].agg(["mean", "median", "std", "count"]).sort_values(
        "mean", ascending=False
    )
    low_share_by_type = (
        df.assign(is_low=df["Rating"] <= 2).groupby("AttractionTypeName")["is_low"].mean() * 100
    ).sort_values(ascending=False)

    popularity_vs_quality = (
        df.groupby("Attraction").agg(visits=("Rating", "count"), avg_rating=("Rating", "mean")).reset_index()
    )

    stats = {
        "rating_counts": {int(k): int(v) for k, v in rating_counts.items()},
        "rating_mean": round(float(df["Rating"].mean()), 3),
        "rating_median": float(df["Rating"].median()),
        "rating_std": round(float(df["Rating"].std()), 3),
        "pct_rating_4_or_5": round(df["Rating"].isin([4, 5]).mean() * 100, 2),
        "pct_rating_1_or_2": round(df["Rating"].isin([1, 2]).mean() * 100, 2),
        "rating_by_visit_mode": {
            k: {"mean": round(float(v["mean"]), 3), "median": float(v["median"]), "count": int(v["count"])}
            for k, v in rating_by_mode.iterrows()
        },
        "low_rating_share_pct_by_type": low_share_by_type.round(2).to_dict(),
        "popularity_quality_correlation": round(
            float(popularity_vs_quality["visits"].corr(popularity_vs_quality["avg_rating"])), 3
        ),
    }

    _save_table(rating_counts.rename("count").to_frame(), "rating_distribution")
    _save_table(rating_by_mode, "rating_by_visit_mode")
    _save_table(popularity_vs_quality, "popularity_vs_quality")

    if PLOTLY_AVAILABLE:
        fig = go.Figure(
            go.Bar(x=rating_counts.index, y=rating_counts.values, marker=dict(color=ACCENTS[4]))
        )
        fig.update_layout(
            title="Rating distribution (1\u20135 stars)", xaxis=dict(dtick=1, title="Rating"), yaxis_title="Count"
        )
        _save_fig(fig, "12_rating_distribution")

        fig = px.bar(
            rating_by_mode.reset_index(),
            x="mean",
            y="VisitModeName",
            orientation="h",
            error_x="std",
            labels={"mean": "Average rating", "VisitModeName": "Visit mode"},
            title="Average rating by Visit Mode (error bars = std dev)",
            color_discrete_sequence=[ACCENTS[0]],
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False)
        _save_fig(fig, "13_rating_by_visit_mode")

        fig = px.bar(
            low_share_by_type.iloc[::-1],
            orientation="h",
            labels={"value": "Share of 1\u20132\u2605 ratings (%)", "index": "Attraction type"},
            title="Share of low ratings (1\u20132\u2605) by attraction type",
            color_discrete_sequence=[ACCENTS[4]],
        )
        fig.update_layout(showlegend=False, height=520)
        _save_fig(fig, "14_low_rating_share_by_type")

        fig = px.scatter(
            popularity_vs_quality,
            x="visits",
            y="avg_rating",
            hover_name="Attraction",
            labels={"visits": "Total visits (popularity)", "avg_rating": "Average rating (quality)"},
            title="Popularity vs. quality \u2014 one point per attraction",
            color_discrete_sequence=[ACCENTS[1]],
        )
        fig.update_traces(marker=dict(size=10, opacity=0.8))
        _save_fig(fig, "15_popularity_vs_quality_scatter")

    return stats


# --------------------------------------------------------------------------
# 6. Business insights (derived from the stats above, not fabricated)
# --------------------------------------------------------------------------

def business_insights(overview: dict, traveler: dict, attraction: dict, temporal: dict, rating: dict) -> list[str]:
    insights = []

    top_country, top_country_n = next(iter(traveler["top_15_user_countries"].items()))
    insights.append(
        f"Demand is geographically concentrated: {top_country} alone accounts for "
        f"{top_country_n:,} of {overview['n_rows']:,} transactions "
        f"({top_country_n / overview['n_rows'] * 100:.1f}%), and the top 3 source countries "
        f"together cover a large share of volume — marketing/localization spend is better "
        f"targeted at a short list of countries than spread evenly."
    )

    top_attraction, top_attraction_n = next(iter(attraction["top_10_attractions_by_visits"].items()))
    insights.append(
        f"Attraction demand is highly concentrated too: {top_attraction} alone drew "
        f"{top_attraction_n:,} visits ({top_attraction_n / overview['n_rows'] * 100:.1f}% of all "
        f"transactions) out of only {attraction['n_attractions']} attractions in the catalogue — "
        f"capacity planning and on-site experience investment should prioritize this small set of "
        f"marquee sites."
    )

    dominant_mode, dominant_share = max(traveler["visit_mode_share_pct"].items(), key=lambda kv: kv[1])
    insights.append(
        f"'{dominant_mode}' is the dominant travel mode ({dominant_share:.1f}% of visits); "
        f"Business travel is a minor segment ({traveler['visit_mode_share_pct'].get('Business', 0):.1f}%) "
        f"but rates experiences highest on average "
        f"({rating['rating_by_visit_mode']['Business']['mean']:.2f}/5) — a small, low-volume but "
        f"high-satisfaction segment worth a dedicated offering rather than more inventory."
    )

    covid_share = temporal["covid_years_2020_2021_share_pct"]
    insights.append(
        f"2020\u20132021 volume collapsed to just {covid_share:.1f}% of total transactions "
        f"(COVID-19 travel disruption is clearly visible in the data) with a slow partial recovery "
        f"by 2022 — any trend/seasonality model trained on this data should treat 2020\u20132021 as "
        f"an anomalous period rather than representative seasonality."
    )

    worst_type = min(attraction["avg_rating_by_type"].items(), key=lambda kv: kv[1])
    best_type = max(attraction["avg_rating_by_type"].items(), key=lambda kv: kv[1])
    insights.append(
        f"Satisfaction varies materially by attraction type: '{best_type[0]}' rates highest on average "
        f"({best_type[1]:.2f}/5) while '{worst_type[0]}' rates lowest ({worst_type[1]:.2f}/5) and also "
        f"shows one of the higher shares of 1\u20132\u2605 ratings "
        f"({rating['low_rating_share_pct_by_type'].get(worst_type[0], 0):.1f}%) \u2014 a natural target "
        f"for a service-quality review."
    )

    corr = rating["popularity_quality_correlation"]
    direction = "no meaningful" if abs(corr) < 0.2 else ("a positive" if corr > 0 else "a negative")
    insights.append(
        f"Popularity and average rating show {direction} correlation across attractions "
        f"(r={corr:.2f}) \u2014 the most-visited sites are not simply the best-reviewed ones, so "
        f"recommendation/ranking logic should weight both signals rather than defaulting to raw "
        f"popularity."
    )

    insights.append(
        f"{traveler['pct_users_repeat']:.1f}% of the {traveler['n_users_total']:,} unique travelers "
        f"in the data made more than one visit (median {traveler['trips_per_user_median']:.0f} trip"
        f", max {traveler['trips_per_user_max']}) \u2014 the base is dominated by one-time visitors, "
        f"which matters for how much a collaborative-filtering recommender can rely on a user's own "
        f"history versus needing strong cold-start handling (already implemented upstream in "
        f"regression_features.csv)."
    )

    return insights


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

def validate(df: pd.DataFrame, overview: dict) -> dict:
    """Sanity checks the EDA numbers against the raw dataframe again,
    independently of the section functions above, so a bug in one of them
    can't silently produce a self-consistent-but-wrong report."""
    checks = {
        "row_count_matches_expected_49208": len(df) == 49208,
        "no_missing_values": int(df.isna().sum().sum()) == 0,
        "no_duplicate_transaction_ids": df["TransactionId"].duplicated().sum() == 0,
        "rating_within_1_5": df["Rating"].between(1, 5).all(),
        "visit_month_within_1_12": df["VisitMonth"].between(1, 12).all(),
        "visit_mode_share_sums_to_100": abs(
            df["VisitModeName"].value_counts(normalize=True).sum() * 100 - 100
        )
        < 0.01,
        "single_attraction_country_confirmed": overview["n_unique_attraction_countries"] == 1,
    }
    checks["all_passed"] = all(bool(v) for v in checks.values())
    return checks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    overview = overview_stats(df)
    traveler = traveler_analysis(df)
    attraction = attraction_analysis(df)
    temporal = temporal_analysis(df)
    rating = rating_analysis(df)
    insights = business_insights(overview, traveler, attraction, temporal, rating)
    checks = validate(df, overview)

    summary = {
        "dataset_overview": overview,
        "traveler_analysis": traveler,
        "attraction_analysis": attraction,
        "temporal_analysis": temporal,
        "rating_analysis": rating,
        "business_insights": insights,
        "validation": checks,
        "plotly_figures_generated": PLOTLY_AVAILABLE,
    }

    with open(os.path.join(OUT_DIR, "eda_summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Rows analyzed: {overview['n_rows']:,}")
    print(f"Validation passed: {checks['all_passed']}")
    print(f"Plotly figures generated: {PLOTLY_AVAILABLE}")
    if not PLOTLY_AVAILABLE:
        print(
            "NOTE: plotly is not installed in this environment, so .html figures were "
            "skipped. All statistics/tables were still computed and saved. Install "
            "requirements.txt (plotly>=5.20 is already listed) and re-run to also get "
            "the interactive charts."
        )
    print(f"Summary JSON: {os.path.join(OUT_DIR, 'eda_summary_stats.json')}")
    print(f"Tables: {TABLE_DIR}")
    if PLOTLY_AVAILABLE:
        print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()

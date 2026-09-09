"""
Fallback static preview renderer for the EDA charts.

This is NOT the canonical visualization deliverable — `scripts/eda_analysis.py`
is, and it generates the real interactive Plotly charts once run in an
environment with `plotly` installed (already declared in requirements.txt).

This script exists only because the checkpoint was assembled in a sandbox
with no outbound network access to install plotly, so there was no way to
render an example of the Plotly output there. It reproduces the same 15
charts, same data, same color palette (matching components/charts.py) using
matplotlib (already a project-independent, always-available dependency) so
the checkpoint zip contains something immediately viewable.

Output: docs/eda/figures_preview/*.png

Run:
    python scripts/eda_preview_matplotlib.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE_DIR, "docs", "eda", "figures_preview")

plt.rcParams.update(
    {
        "figure.facecolor": "#0B1626",
        "axes.facecolor": "#0B1626",
        "axes.edgecolor": "#6E7E96",
        "axes.labelcolor": "#A9B7CC",
        "xtick.color": "#A9B7CC",
        "ytick.color": "#A9B7CC",
        "text.color": "#F3F6FB",
        "grid.color": "#20304A",
        "font.size": 10,
    }
)
ACCENTS = ["#8B6FF0", "#4CB8F0", "#2FD9A8", "#F0A24C", "#F2685C"]


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"), dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    df = pd.read_csv(os.path.join(BASE_DIR, "data", "processed", "consolidated.csv"))

    vm = df["VisitModeName"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(vm.values, labels=vm.index, colors=ACCENTS[: len(vm)], wedgeprops=dict(width=0.4), textprops={"color": "#F3F6FB"})
    ax.set_title("Visit Mode distribution")
    _save(fig, "01_visit_mode_distribution")

    top_c = df["UserCountry"].value_counts().head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top_c.index, top_c.values, color=ACCENTS[1])
    ax.set_title("Top 15 traveler home countries")
    ax.set_xlabel("Transactions")
    _save(fig, "02_top_user_countries")

    cc = df["UserContinent"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(cc.values, labels=cc.index, colors=ACCENTS[: len(cc)], wedgeprops=dict(width=0.4), textprops={"color": "#F3F6FB"})
    ax.set_title("Traveler home continent distribution")
    _save(fig, "03_continent_distribution")

    top_a = df.groupby("Attraction").size().sort_values(ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_a.index, top_a.values, color=ACCENTS[0])
    ax.set_title("Top 15 attractions by visits")
    ax.set_xlabel("Transactions")
    _save(fig, "04_top_attractions_by_visits")

    tc = df["AttractionTypeName"].value_counts().iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(tc.index, tc.values, color=ACCENTS[2])
    ax.set_title("Visit volume by attraction type")
    ax.set_xlabel("Transactions")
    _save(fig, "05_attraction_type_volume")

    tr = df.groupby("AttractionTypeName")["Rating"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.RdYlGn((tr.values - tr.values.min()) / (tr.values.max() - tr.values.min()))
    ax.barh(tr.index, tr.values, color=colors)
    ax.set_title("Average rating by attraction type")
    ax.set_xlabel("Average rating")
    _save(fig, "06_avg_rating_by_attraction_type")

    att = df.groupby("Attraction")["Rating"].agg(["mean", "count"])
    q = att[att["count"] >= 50]
    top8 = q.sort_values("mean", ascending=False).head(8)
    bot8 = q.sort_values("mean").head(8)
    cmp = pd.concat([bot8.assign(g="Lowest"), top8.assign(g="Top")]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [("#2FD9A8" if g == "Top" else "#F2685C") for g in cmp["g"]]
    ax.barh(cmp.index, cmp["mean"], color=colors)
    ax.set_title("Highest & lowest rated attractions (min 50 ratings)")
    ax.set_xlabel("Average rating")
    _save(fig, "07_top_bottom_rated_attractions")

    yv = df["VisitYear"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(yv.index.astype(str), yv.values, color=ACCENTS[1])
    ax.set_title("Visit volume by year (2013-2022)")
    ax.set_ylabel("Transactions")
    _save(fig, "08_yearly_visit_volume")

    mv = df["VisitMonth"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(mv.index, mv.values, color=ACCENTS[3])
    ax.set_title("Seasonality - visits by calendar month (all years)")
    ax.set_xticks(range(1, 13))
    ax.set_ylabel("Transactions")
    _save(fig, "09_monthly_seasonality")

    yr = df.groupby("VisitYear")["Rating"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(yr.index, yr.values, marker="o", color=ACCENTS[2], linewidth=2.5)
    ax.set_title("Average rating by year")
    ax.set_ylabel("Average rating (1-5)")
    _save(fig, "10_yearly_avg_rating_trend")

    ms = pd.crosstab(df["VisitYear"], df["VisitModeName"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = None
    for i, col in enumerate(ms.columns):
        ax.bar(ms.index.astype(str), ms[col], bottom=bottom, label=col, color=ACCENTS[i % len(ACCENTS)])
        bottom = ms[col] if bottom is None else bottom + ms[col]
    ax.set_title("Visit Mode share by year (%)")
    ax.legend(fontsize=8, facecolor="#0B1626", labelcolor="#F3F6FB")
    _save(fig, "11_visit_mode_share_by_year")

    rc = df["Rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(rc.index, rc.values, color=ACCENTS[4])
    ax.set_title("Rating distribution (1-5 stars)")
    ax.set_xticks(range(1, 6))
    _save(fig, "12_rating_distribution")

    rm = df.groupby("VisitModeName")["Rating"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(rm.index, rm.values, color=ACCENTS[0])
    ax.set_title("Average rating by Visit Mode")
    _save(fig, "13_rating_by_visit_mode")

    low = (df.assign(is_low=df["Rating"] <= 2).groupby("AttractionTypeName")["is_low"].mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(low.index, low.values, color=ACCENTS[4])
    ax.set_title("Share of low ratings (1-2 stars) by attraction type")
    ax.set_xlabel("%")
    _save(fig, "14_low_rating_share_by_type")

    pq = df.groupby("Attraction").agg(visits=("Rating", "count"), avg_rating=("Rating", "mean"))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pq["visits"], pq["avg_rating"], color=ACCENTS[1], s=60, alpha=0.8)
    ax.set_title("Popularity vs quality (one point per attraction)")
    ax.set_xlabel("Total visits")
    ax.set_ylabel("Average rating")
    _save(fig, "15_popularity_vs_quality_scatter")

    print(f"Saved {len(os.listdir(OUT_DIR))} preview PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()

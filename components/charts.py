"""
Reusable chart wrappers, themed to match the dark glass UI.

Part 1 ships these with clearly-labelled DEMO data only, so the
plotting/theming plumbing is proven out before any real dataset is
wired in later.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#A9B7CC", family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(showgrid=False, zeroline=False, color="#6E7E96"),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False, color="#6E7E96"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

ACCENTS = ["#8B6FF0", "#4CB8F0", "#2FD9A8", "#F0A24C", "#F2685C"]


def demo_trend_chart(values: list[float], labels: list[str], color: str = "#4CB8F0", height: int = 140):
    """A minimal sparkline-style area chart for demo trend decoration."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines",
            line=dict(color=color, width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor=color.replace(")", ", 0.15)").replace("rgb", "rgba") if color.startswith("rgb") else _hex_to_rgba(color, 0.15),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=height, showlegend=False)
    fig.update_yaxes(visible=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def demo_bar_ranking(labels: list[str], values: list[float], height: int = 260):
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=ACCENTS[: len(labels)]),
        )
    )
    layout = dict(PLOTLY_LAYOUT)
    layout["yaxis"] = dict(autorange="reversed", showgrid=False, color="#A9B7CC")
    fig.update_layout(**layout, height=height, showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def demo_donut(labels: list[str], values: list[float], height: int = 260):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=ACCENTS[: len(labels)]),
            textfont=dict(color="#F3F6FB"),
        )
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def world_choropleth(country_df: pd.DataFrame, height: int = 430):
    """Dark-themed world choropleth for the Home 'Explore the World' section.

    country_df needs columns: Country, AttractionCount, AvgRating, HoverText.
    Geography/boundaries come from Plotly's built-in world-atlas basemap
    (a reference layer only) -- no attraction data is sourced from it.
    Countries with AttractionCount == 0 render as a dim neutral tone;
    countries present in our processed dataset light up in the accent color.
    """
    fig = go.Figure(
        go.Choropleth(
            locations=country_df["Country"],
            locationmode="country names",
            z=country_df["AttractionCount"],
            zmin=0,
            zmax=max(1, country_df["AttractionCount"].max()),
            colorscale=[[0.0, "rgba(110,126,150,0.22)"], [1.0, "#2FD9A8"]],
            showscale=False,
            marker_line_color="rgba(255,255,255,0.10)",
            marker_line_width=0.6,
            hovertext=country_df["HoverText"],
            hoverinfo="text",
            hoverlabel=dict(
                bgcolor="#10233B",
                bordercolor="rgba(255,255,255,0.18)",
                font=dict(color="#F3F6FB", family="Inter, sans-serif", size=12),
            ),
        )
    )
    fig.update_geos(
        bgcolor="rgba(0,0,0,0)",
        showframe=False,
        showcoastlines=False,
        showland=True,
        landcolor="rgba(255,255,255,0.035)",
        showocean=True,
        oceancolor="rgba(0,0,0,0)",
        showlakes=False,
        showcountries=True,
        countrycolor="rgba(255,255,255,0.10)",
        projection_type="natural earth",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=6, b=0),
        height=height,
        font=dict(color="#A9B7CC", family="Inter, sans-serif", size=12),
        dragmode=False,
    )
    return fig

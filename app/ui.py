"""Shared UI helpers: data loading, number formatting, charts.

Pages import from here and nothing else in app/; all data comes from the pipeline outputs
(data/outputs, reports/). Loads are cached and busted by file mtime, so re-running the
pipeline shows up on the next rerun without restarting the app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sic import config

ACCENT, SECOND, MUTED = "#2a78d6", "#eb6834", "#a3a29c"
OUT, REP = config.path("outputs"), config.path("reports")


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def _parquet(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def _json(path: str, mtime: float) -> dict:
    return json.loads(Path(path).read_text())


def load(name: str) -> pd.DataFrame | None:
    p = OUT / name
    return _parquet(str(p), p.stat().st_mtime) if p.exists() else None


def load_json(name: str) -> dict | None:
    p = REP / name
    return _json(str(p), p.stat().st_mtime) if p.exists() else None


@st.cache_data(show_spinner=False)
def _complete_weeks(path: str, mtime: float) -> frozenset:
    f = pd.read_parquet(path, columns=["date", "week_start"]).drop_duplicates()
    days = f.groupby("week_start").date.nunique()
    return frozenset(days[days == 7].index)


def complete_weeks() -> frozenset:
    """Weeks with all 7 calendar days forecast. The first (starts Friday 2014-08-01) and the
    last (ends Friday 2015-07-31) are partial, so their € totals are not comparable."""
    p = OUT / "forecasts.parquet"
    return _complete_weeks(str(p), p.stat().st_mtime)


def require(*items) -> None:
    """Stop the page with one line if a pipeline output is missing."""
    if any(i is None for i in items):
        st.warning("Pipeline outputs missing. Run `make all`.")
        st.stop()


# ---------------------------------------------------------------- formatting
def eur(x: float) -> str:
    a = abs(x)
    s = f"€{a / 1e6:.2f}M" if a >= 1e6 else f"€{a / 1e3:.1f}k" if a >= 1e4 else f"€{a:,.0f}"
    return ("-" if x < 0 else "") + s


def pct(x, d: int = 1, sign: bool = True) -> str:
    return "—" if x is None or pd.isna(x) else f"{x * 100:{'+' if sign else ''}.{d}f}%"


def week_label(start: pd.Timestamp, days: int = 7) -> str:
    """'20–26 Jul 2015'; a partial week shows its real last day ('27–31 Jul 2015')."""
    end = start + pd.Timedelta(days=days - 1)
    left = f"{start:%-d}" if start.month == end.month else f"{start:%-d %b}"
    return f"{left}–{end:%-d %b %Y}"


# ---------------------------------------------------------------- charts
def chart(fig: go.Figure, height: int = 280, y: str | None = None, title: str | None = None,
          key: str | None = None, container=st) -> None:
    """One look for every chart: no background, faint y grid, legend only for 2+ series,
    title in its own row so it never collides with the legend. y = 'pct' | 'eur' | None."""
    named = [t for t in fig.data if t.name and t.showlegend is not False]
    legend = len(named) > 1
    fig.update_layout(
        height=height, font={"size": 12},
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        title={"text": title, "x": 0, "xanchor": "left", "y": 1, "yanchor": "top",
               "yref": "container", "pad": {"t": 6}, "font": {"size": 14}} if title else None,
        showlegend=legend,
        legend={"orientation": "h", "x": 0, "xanchor": "left", "y": 1.0, "yanchor": "bottom"},
        margin={"l": 0, "r": 0, "t": (34 if title else 4) + (28 if legend else 0), "b": 0},
        hovermode=fig.layout.hovermode or "x unified")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zerolinecolor="rgba(128,128,128,0.4)",
                     tickformat="+.0%" if y == "pct" else None,
                     tickprefix="€" if y == "eur" else None)
    container.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


# ---------------------------------------------------------------- what-if result
def simulation(r: dict, key: str) -> None:
    """A what-if result: impact, range, per week, confidence, then the biggest stores.
    Shared by the Stores page, the Simulate form and the chat's sources."""
    t, w, pc = r["total_eur"], r["per_week_eur"], r["pct_of_baseline"]
    m = st.columns(4)
    m[0].metric("Expected impact", eur(t["mean"]), pct(pc["mean"]), delta_color="off")
    m[1].metric("90% range", f"{eur(t['p5'])} to {eur(t['p95'])}")
    m[2].metric("Per week", eur(w["mean"]))
    m[3].metric("Confidence", r["confidence"].capitalize())
    if len(r["top_stores"]) > 1:
        top = r["top_stores"][::-1]
        fig = go.Figure(go.Bar(y=[f"Store {x['store_id']}" for x in top],
                               x=[x["expected_eur"] for x in top], orientation="h",
                               marker_color=ACCENT if t["mean"] >= 0 else SECOND))
        fig.update_xaxes(tickprefix="€")
        chart(fig.update_layout(hovermode="closest"), height=40 + 24 * len(top), key=f"sim_{key}")
    with st.expander("Method and assumptions"):
        st.markdown(f"{r['effect_used']}  \n{r['method']}  \n{r['uncertainty']}")
        st.markdown("\n".join(f"- {a}" for a in r["assumptions"] + r["caveats"]))

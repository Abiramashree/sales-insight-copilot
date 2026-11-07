"""Feature engineering for the daily store-sales forecaster.

Everything here is "known in advance" at forecast time (calendar, promo schedule,
store opening schedule, store attributes), plus store-level target encodings
computed ONLY from the training window to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sic import config

CATEGORICAL = ["store_type", "assortment", "state_holiday"]
NUMERIC = [
    "dow", "day", "month", "weekofyear", "dayofyear", "year",
    "promo", "school_holiday", "promo2_active", "promo2_renewal_month",
    "competition_active", "months_since_competition_open", "competition_distance",
    "closed_yesterday", "closed_tomorrow", "days_since_reopen",
    # v2: holiday proximity (pre-holiday stock-up, post-holiday dip)
    "days_to_closure", "days_since_closure", "easter_offset", "days_to_christmas",
    # v2: promo sequence (position in run, pull-forward around promo weeks)
    "promo_day_index", "promo_last_week", "promo_next_week",
    # v2: recent store level vs long-run level (trend / level shifts)
    "lvl_recent", "days_since_anchor",
]
ENCODED = ["te_store", "te_store_dow", "te_store_promo", "te_store_cust", "lvl_vs_longrun"]
FEATURES = CATEGORICAL + NUMERIC + ENCODED


def load_base() -> pd.DataFrame:
    con = config.connect(read_only=True)
    df = con.execute("""
        select f.*, s.store_type, s.assortment,
               s.competition_distance_filled_m as competition_distance
        from marts.fct_daily_store_sales f join marts.dim_store s using (store_id)
        order by store_id, date""").df()
    con.close()
    df["date"] = pd.to_datetime(df["date"])
    df["week_start"] = pd.to_datetime(df["week_start"])
    return add_calendar(df)


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["day"], df["month"], df["year"] = d.day, d.month, d.year
    df["weekofyear"] = d.isocalendar().week.astype(int).values
    df["dayofyear"] = d.day_of_year
    df["state_holiday"] = df["state_holiday"].fillna("none")
    for c in ["promo", "school_holiday", "promo2_active", "promo2_renewal_month",
              "competition_active", "is_open"]:
        df[c] = df[c].fillna(False).astype(int)
    df["months_since_competition_open"] = df["months_since_competition_open"].clip(-24, 120)

    g = df.groupby("store_id")["is_open"]
    df["closed_yesterday"] = (g.shift(1).fillna(1) == 0).astype(int)
    df["closed_tomorrow"] = (g.shift(-1).fillna(1) == 0).astype(int)

    # Days since the store reopened after a closure/missing-data spell of >= 7 days
    # (captures refurbishment reopenings, which run hot for weeks).
    open_days = df.loc[df.is_open == 1, ["store_id", "date"]].copy()
    gap = open_days.groupby("store_id")["date"].diff().dt.days
    open_days["reopen_date"] = open_days["date"].where(gap >= 7)
    open_days["reopen_date"] = open_days.groupby("store_id")["reopen_date"].ffill()
    df = df.merge(open_days[["store_id", "date", "reopen_date"]], on=["store_id", "date"],
                  how="left")
    df["reopen_date"] = df.groupby("store_id")["reopen_date"].ffill()
    df["days_since_reopen"] = (df["date"] - df["reopen_date"]).dt.days.clip(0, 365).fillna(365)
    df = df.drop(columns="reopen_date")
    df = add_holiday_proximity(df)
    df = add_promo_sequence(df)
    return add_recent_level(df)


def _easter(year: int) -> pd.Timestamp:
    """Anonymous Gregorian algorithm."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month = (h + l_ - 7 * m + 114) // 31
    day = (h + l_ - 7 * m + 114) % 31 + 1
    return pd.Timestamp(year, month, day)


def add_holiday_proximity(df: pd.DataFrame, cap: int = 15) -> pd.DataFrame:
    """Days to the next / since the last non-Sunday closure for each store (holidays,
    known in advance via the opening schedule), plus chain-wide Easter/Christmas offsets."""
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True)
    closure = (df.is_open == 0) & (df.dow != 7)
    cdate = df.date.where(closure)
    nxt = cdate.groupby(df.store_id).bfill()
    prv = cdate.groupby(df.store_id).ffill()
    df["days_to_closure"] = (nxt - df.date).dt.days.clip(upper=cap).fillna(cap)
    df["days_since_closure"] = (df.date - prv).dt.days.clip(upper=cap).fillna(cap)
    easter = df.date.dt.year.map({y: _easter(y) for y in df.date.dt.year.unique()})
    off = (df.date - easter).dt.days
    df["easter_offset"] = off.where(off.between(-21, 10), 99)
    xmas = pd.to_datetime(df.date.dt.year.astype(str) + "-12-25")
    dx = (xmas - df.date).dt.days
    df["days_to_christmas"] = dx.where(dx.between(-7, 35), 99)
    return df


def add_promo_sequence(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("store_id")
    run = (df.promo != g.promo.shift()).cumsum()
    df["promo_day_index"] = np.where(df.promo == 1, df.groupby(run).cumcount() + 1, 0)
    key = df.set_index(["store_id", "date"]).promo
    for col, shift in (("promo_last_week", -7), ("promo_next_week", 7)):
        idx = pd.MultiIndex.from_arrays([df.store_id, df.date + pd.Timedelta(days=shift)])
        df[col] = key.reindex(idx).fillna(0).astype(int).values
    return df


def anchor_of(dates: pd.Series) -> pd.Series:
    """Start of the 3-month block (Feb/May/Aug/Nov) each date falls in. Backtest folds
    start on these dates, so 'recent level as of the anchor' is computed the same way
    for training rows and for forecast rows: never using data on/after the anchor."""
    m = dates.dt.month
    start_month = ((m - 2) % 12) // 3 * 3 + 2
    year = dates.dt.year - (start_month > m).astype(int)
    start_month = ((start_month - 1) % 12) + 1
    return pd.to_datetime({"year": year, "month": start_month, "day": 1})


def add_recent_level(df: pd.DataFrame, window_days: int = 56) -> pd.DataFrame:
    df["anchor"] = anchor_of(df.date)
    df["days_since_anchor"] = (df.date - df.anchor).dt.days
    o = df[(df.is_open == 1) & (df.sales > 0)][["store_id", "date", "sales"]]
    levels = []
    for a in df.anchor.unique():
        w = o[(o.date >= a - pd.Timedelta(days=window_days)) & (o.date < a)]
        levels.append(np.log1p(w.sales).groupby(w.store_id).mean().rename("lvl_recent")
                      .reset_index().assign(anchor=a))
    lv = pd.concat(levels, ignore_index=True)
    return df.merge(lv, on=["store_id", "anchor"], how="left").drop(columns="anchor")


def target_encode(train: pd.DataFrame, apply: pd.DataFrame) -> pd.DataFrame:
    """Store-level encodings from training rows only (open days, log scale)."""
    t = train[(train.is_open == 1) & (train.sales > 0)].assign(
        ls=lambda x: np.log1p(x.sales), lc=lambda x: np.log1p(x.customers))
    store = t.groupby("store_id").agg(te_store=("ls", "mean"), te_store_cust=("lc", "mean"))
    sd = t.groupby(["store_id", "dow"]).ls.mean().rename("te_store_dow")
    sp = t.groupby(["store_id", "promo"]).ls.mean().unstack()
    sp = (sp[1] - sp[0]).rename("te_store_promo")
    out = apply.drop(columns=[c for c in ENCODED if c in apply], errors="ignore")
    out = out.join(store, on="store_id").join(sp, on="store_id")
    out = out.join(sd, on=["store_id", "dow"])
    out["te_store_dow"] = out["te_store_dow"].fillna(out["te_store"])
    out["lvl_vs_longrun"] = out["lvl_recent"] - out["te_store"]
    return out


def active_features() -> list[str]:
    """FEATURES minus forecast.drop_features in config.yaml (set from the feature-selection
    study, src/sic/forecast/select.py)."""
    drop = set(config.load()["forecast"].get("drop_features") or [])
    unknown = drop - set(FEATURES)
    if unknown:
        raise ValueError(f"drop_features not in FEATURES: {sorted(unknown)}")
    return [f for f in FEATURES if f not in drop]


def to_matrix(df: pd.DataFrame, feats: list[str] | None = None) -> pd.DataFrame:
    X = df[feats if feats is not None else active_features()].copy()
    for c in CATEGORICAL:
        if c in X:
            X[c] = X[c].astype("category")
    return X

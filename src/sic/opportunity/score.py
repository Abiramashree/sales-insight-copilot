"""Store opportunity scoring: where does management attention pay off most?

The store analogue of B2B lead scoring. Each store gets three levers, all expressed in
weekly-equivalent EUR with assumptions taken from config.yaml (opportunity:):

  basket headroom    gap in spend per customer to its 10 look-alike stores (same type and
                     assortment, nearest competition distance / Promo2 status) x the store's
                     own customers x a capture rate. Observational -> lowest weight.
                     (Sales per day vs peers was rejected: it mostly measures catchment.)
  promo lever        measured causal promo lift x the store's non-promo baseline: value of
                     one extra promo week per quarter. Causal (within-store) -> full weight.
  competitor defense stores whose competitor opened in the last ~6 months: expected weekly
                     loss from the DiD estimate (within / beyond 1 km) x protectable share.

score = sum(lever EUR x confidence weight). Momentum (slope of weekly sales-vs-forecast over
the last 13 weeks) is reported as a flag (improving / stable / sliding), not added to EUR.

Validation: are the levers persistent store traits rather than noise? Split-sample rank
correlation of (a) per-store promo lift estimated on 2013-H1 2014 vs H2 2014-2015 and
(b) the peer gap measured in H1 2014 vs H1 2015.

Writes data/outputs/opportunity_scores.parquet and reports/m3_opportunity.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sic import config


def _cfg() -> dict:
    return config.load()["opportunity"]


def productivity(start: str, end: str) -> pd.DataFrame:
    con = config.connect(read_only=True)
    s = con.execute(f"""
        select store_id, sum(sales) / count(*) spd, sum(sales) / sum(customers) basket,
               sum(customers) / count(*) customers_per_day
        from marts.fct_daily_store_sales
        where is_open and sales > 0 and date between date '{start}' and date '{end}'
        group by 1""").df().set_index("store_id")
    con.close()
    return s


def peers_table(metric: pd.Series, dim: pd.DataFrame, k: int) -> pd.DataFrame:
    """For each store: its k nearest look-alikes (same type + assortment; distance on
    log competition distance and Promo2 status) and their median of `metric`."""
    spd = metric
    d = dim.loc[dim.index.intersection(spd.index)].copy()
    d["log_comp"] = np.log(d.competition_distance_filled_m)
    rows = []
    for (_t, _a), g in d.groupby(["store_type", "assortment"]):
        X = np.c_[(g.log_comp - g.log_comp.mean()) / (g.log_comp.std() or 1),
                  g.promo2_enrolled.astype(float)]
        for i, sid in enumerate(g.index):
            dist = np.sqrt(((X - X[i]) ** 2).sum(axis=1))
            order = [g.index[j] for j in np.argsort(dist) if g.index[j] != sid][:k]
            rows.append({"store_id": sid, "peers": order, "n_peers": len(order),
                         "peer_value": float(spd.loc[order].median()) if order else np.nan})
    return pd.DataFrame(rows).set_index("store_id")


def momentum(w: pd.DataFrame, weeks: int) -> pd.DataFrame:
    w = w[w.open_days >= 5]
    recent = w[w.week_start > w.week_start.max() - pd.Timedelta(weeks=weeks)]

    def f(g: pd.DataFrame) -> pd.Series:
        g = g.sort_values("week_start")
        y = (g.sales / g.forecast - 1).values
        x = np.arange(len(y))
        slope = np.polyfit(x, y, 1)[0] if len(y) >= 4 else 0.0
        return pd.Series({"gap_recent": g.sales.sum() / g.forecast.sum() - 1,
                          "gap_slope_per_week": slope,
                          "weekly_baseline": g.forecast.mean()})

    m = recent.groupby("store_id").apply(f, include_groups=False)
    m["momentum"] = np.select([m.gap_slope_per_week > 0.005, m.gap_slope_per_week < -0.005],
                              ["improving", "sliding"], "stable")
    return m


def compute() -> pd.DataFrame:
    cfg = _cfg()
    out, rep = config.path("outputs"), config.path("reports")
    con = config.connect(read_only=True)
    dim = con.execute("select * from marts.dim_store").df().set_index("store_id")
    last = con.execute("select max(date) from marts.fct_daily_store_sales").fetchone()[0]
    con.close()
    last = pd.Timestamp(last)
    start = (last - pd.Timedelta(weeks=cfg["recent_weeks"])).date()
    prod = productivity(str(start), str(last.date()))
    peers = peers_table(prod.basket, dim, cfg["peers"]).rename(
        columns={"peer_value": "peer_basket"})
    lift = pd.read_parquet(out / "promo_lift_by_store.parquet").set_index("store_id")
    mom = momentum(pd.read_parquet(out / "store_weeks.parquet"), cfg["recent_weeks"])
    comp = json.loads((rep / "m3_competitor_impact.json").read_text())["by_distance"]

    df = dim[["store_type", "assortment", "competition_distance_m", "competition_open_date",
              "promo2_enrolled"]].join(prod).join(peers).join(
        lift[["lift", "lift_lo", "lift_hi", "incremental_eur_per_promo_week"]]).join(mom)
    df = df[df.spd.notna() & df.weekly_baseline.notna()]

    open_days = 6.0  # Mon-Sat
    # Basket (spend per customer) vs look-alikes: what the store can influence (availability,
    # displays, upsell). Sales per day mostly reflects catchment size (M5 check: corr 0.90
    # with store size), so it is NOT used as headroom.
    df["peer_gap_pct"] = df.basket / df.peer_basket - 1
    df["eur_peer"] = (np.clip(df.peer_basket - df.basket, 0, None) * df.customers_per_day
                      * open_days * cfg["peer_capture_rate"])
    df["eur_promo"] = (df.incremental_eur_per_promo_week.clip(lower=0)
                       * cfg["extra_promo_weeks_per_quarter"] / 13)
    opened = pd.to_datetime(df.competition_open_date)
    recent_comp = ((opened <= last) & (opened >= last - pd.Timedelta(
        days=cfg["competitor_lookback_days"]))
        & (df.competition_distance_m <= cfg["competitor_max_m"]))
    near = df.competition_distance_m < 1000
    effect = np.where(near, comp["within_1000m"]["effect_pct"], comp["beyond_1000m"]["effect_pct"])
    df["competitor_recent"] = recent_comp
    df["eur_competitor"] = np.where(recent_comp, df.weekly_baseline * np.abs(effect)
                                    * cfg["competitor_protectable_share"], 0.0)
    for c in ("eur_peer", "eur_promo", "eur_competitor"):  # a missing lever is worth 0
        df[c] = df[c].fillna(0.0)
    w = cfg["confidence_weights"]
    df["score_eur"] = (df.eur_peer * w["peer"] + df.eur_promo * w["promo"]
                       + df.eur_competitor * w["competitor"])
    levers = df[["eur_peer", "eur_promo", "eur_competitor"]].rename(
        columns=lambda c: c.replace("eur_", ""))
    df["primary_lever"] = levers.idxmax(axis=1)
    df["opportunity_type"] = df.primary_lever.map(
        {"peer": "Lift basket to peer level", "promo": "Grow with promos",
         "competitor": "Defend vs new competitor"})
    df["rank"] = df.score_eur.rank(ascending=False, method="first").astype(int)
    df["why"] = df.apply(explain, axis=1)
    return df.sort_values("rank")


def explain(r: pd.Series) -> str:
    parts = []
    if r.eur_peer > 0 and pd.notna(r.peer_gap_pct):
        parts.append(f"basket €{r.basket:.2f} is {abs(r.peer_gap_pct):.0%} below its "
                     f"{r.n_peers} look-alike stores (€{r.peer_basket:.2f})")
    if pd.notna(r.lift):
        parts.append(f"promo lift {r.lift:+.0%} (≈€{r.incremental_eur_per_promo_week:,.0f} "
                     f"per promo week)")
    if r.competitor_recent:
        parts.append(f"competitor opened {r.competition_distance_m:.0f} m away recently")
    parts.append(f"momentum {r.momentum} ({r.gap_recent:+.0%} vs forecast, last 13 weeks)")
    return "; ".join(parts)


def validate() -> dict:
    """Persistence of each lever across independent periods (Spearman rank correlation)."""
    from scipy.stats import spearmanr

    from sic.causal.promo_lift import store_regression
    from sic.forecast.features import load_base

    base = load_base()
    d = base[(base.is_open == 1) & (base.sales > 0) & (base.dow <= 5)]
    halves = {"early": d[d.date < "2014-07-01"], "late": d[d.date >= "2014-07-01"]}
    est = {}
    for name, part in halves.items():
        est[name] = pd.Series({sid: store_regression(g)["beta"]
                               for sid, g in part.groupby("store_id") if len(g) > 150})
    both = est["early"].index.intersection(est["late"].index)
    rho_lift = spearmanr(est["early"][both], est["late"][both])[0]

    con = config.connect(read_only=True)
    dim = con.execute("select * from marts.dim_store").df().set_index("store_id")
    con.close()
    k = _cfg()["peers"]
    gaps = {}
    sizes = {}
    for name, (a, b) in {"h1_2014": ("2014-01-01", "2014-06-30"),
                         "h1_2015": ("2015-01-01", "2015-06-30")}.items():
        prod = productivity(a, b)
        p = peers_table(prod.basket, dim, k)
        gaps[name] = (prod.basket / p.peer_value - 1).dropna()
        sizes[name] = prod.spd
    gb = gaps["h1_2014"].index.intersection(gaps["h1_2015"].index)
    rho_peer = spearmanr(gaps["h1_2014"][gb], gaps["h1_2015"][gb])[0]
    size_corr = spearmanr(gaps["h1_2015"][gb], sizes["h1_2015"].reindex(gb))[0]
    return {"promo_lift_split_sample_spearman": float(rho_lift), "promo_lift_stores": len(both),
            "basket_gap_year_over_year_spearman": float(rho_peer), "basket_gap_stores": len(gb),
            "basket_gap_vs_store_size_spearman": float(size_corr),
            "rejected_lever": "sales-per-day gap vs peers: stable (0.97) but 0.90 correlated "
                              "with store size, i.e. catchment, not something a manager can fix",
            "reading": "Persistence shows a lever is a real store trait, not noise; low "
                       "correlation with store size shows it is not just location."}


def run(with_validation: bool = True) -> dict:
    df = compute()
    keep = ["rank", "store_type", "assortment", "score_eur", "opportunity_type", "primary_lever",
            "eur_peer", "eur_promo", "eur_competitor", "peer_gap_pct", "spd", "basket",
            "peer_basket", "customers_per_day",
            "peers", "lift", "incremental_eur_per_promo_week", "competitor_recent",
            "competition_distance_m", "gap_recent", "gap_slope_per_week", "momentum",
            "weekly_baseline", "why"]
    out = df[keep].reset_index()
    out["peers"] = out.peers.map(lambda x: [int(i) for i in x] if isinstance(x, list) else [])
    out.to_parquet(config.path("outputs") / "opportunity_scores.parquet", index=False)
    summary = {
        "stores_scored": len(out),
        "assumptions": _cfg(),
        "total_weekly_eur_risk_adjusted": float(out.score_eur.sum()),
        "top10_weekly_eur": float(out.head(10).score_eur.sum()),
        "by_type": out.opportunity_type.value_counts().to_dict(),
        "momentum": out.momentum.value_counts().to_dict(),
        "top10": out.head(10)[["store_id", "rank", "score_eur", "opportunity_type", "momentum",
                               "why"]].round(0).to_dict(orient="records"),
    }
    if with_validation:
        summary["validation"] = validate()
    (config.path("reports") / "m3_opportunity.json").write_text(
        json.dumps(summary, indent=1, default=str))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))

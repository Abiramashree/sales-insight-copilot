"""A/B test design for the promo-mechanics pilot: empirical power via placebo experiments.

Population: the lowest-lift stores (same rule as the M3 recommendation).
For every window of D consecutive promo weeks in the out-of-sample period, randomly
split the stores 50/50 many times with NO treatment and record the difference in mean
outcome. The spread of those placebo differences is the noise a real test must beat:

  MDE (80% power, 5% two-sided) = (1.96 + 0.84) * SD(placebo difference)

Outcome metrics (per store, averaged over the test window):
  raw       log(weekly sales per open day)
  vs_fc     log(actual / forecast): the forecast (built only from pre-period info)
            removes predictable seasonality and store level
  cuped     vs_fc adjusted with each store's pre-period vs_fc (CUPED, theta by OLS)

Also checks the placebo false-positive rate of a Welch t-test (should be about 5%).
Writes reports/m5_ab_design.json and a power-curve figure.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sic import config

Z = 1.959964 + 0.841621  # alpha 0.05 two-sided, power 0.80


def pilot_stores() -> list[int]:
    lift = pd.read_parquet(config.path("outputs") / "promo_lift_by_store.parquet")
    low = lift[lift.lift_hi < lift.lift.quantile(0.25)].nsmallest(max(1, len(lift) // 10), "lift")
    return low.store_id.astype(int).tolist()


def panel(stores: list[int]) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    w = pd.read_parquet(config.path("outputs") / "store_weeks.parquet")
    con = config.connect(read_only=True)
    pdays = con.execute("select store_id, week_start, promo_days from marts.fct_weekly_store_sales").df()
    con.close()
    pdays["week_start"] = pd.to_datetime(pdays.week_start)
    w = w.merge(pdays, on=["store_id", "week_start"])
    w = w[w.store_id.isin(stores) & (w.open_days >= 4) & (w.sales > 0)]
    promo_weeks = sorted(w.groupby("week_start").promo_days.median().loc[lambda s: s >= 4].index)
    w = w[w.week_start.isin(promo_weeks)].copy()
    w["raw"] = np.log(w.sales / w.open_days)
    w["vs_fc"] = np.log(w.sales / w.forecast)
    return w, promo_weeks


def simulate(w: pd.DataFrame, weeks: list, durations=(2, 4, 6), pre: int = 4,
             reps: int = 300, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    raw = w.pivot(index="store_id", columns="week_start", values="raw")
    vf = w.pivot(index="store_id", columns="week_start", values="vs_fc")
    out = {}
    for D in durations:
        diffs = {"raw": [], "vs_fc": [], "cuped": []}
        tstats = []
        for i in range(pre, len(weeks) - D + 1):
            win, prev = weeks[i:i + D], weeks[i - pre:i]
            y_raw = raw[win].mean(axis=1)
            y = vf[win].mean(axis=1)
            x = vf[prev].mean(axis=1)
            ok = y.notna() & x.notna() & y_raw.notna()
            y_raw, y, x = y_raw[ok].values, y[ok].values, x[ok].values
            theta = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
            y_c = y - theta * (x - x.mean())
            n = len(y)
            for _ in range(reps):
                t = rng.permutation(n) < n // 2
                for name, v in (("raw", y_raw), ("vs_fc", y), ("cuped", y_c)):
                    diffs[name].append(v[t].mean() - v[~t].mean())
                a, b = y_c[t], y_c[~t]
                tstats.append((a.mean() - b.mean())
                              / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
        sd = {k: float(np.std(v)) for k, v in diffs.items()}
        out[D] = {"placebo_sd": sd,
                  "mde_pct": {k: float(np.expm1(Z * s)) for k, s in sd.items()},
                  "welch_false_positive_rate_cuped": float(np.mean(np.abs(tstats) > 1.96)),
                  "n_windows": len(range(pre, len(weeks) - D + 1)), "stores": int(n)}
    return out


def figure(res: dict) -> str:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from scipy.stats import norm

    from sic.plotting import AQUA, BLUE, NEUTRAL, save

    fig, ax = plt.subplots(figsize=(8, 3.4))
    eff = np.linspace(0, 0.15, 151)
    D = 2
    for name, color, label in (("raw", NEUTRAL, "Raw sales"), ("vs_fc", BLUE, "Sales vs forecast"),
                               ("cuped", AQUA, "Sales vs forecast + CUPED")):
        sd = res[D]["placebo_sd"][name]
        power = norm.cdf(np.log1p(eff) / sd - 1.959964)
        ax.plot(eff, power, color=color, label=label)
    ax.axhline(0.8, color=NEUTRAL, linewidth=0.8, linestyle="--")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("True effect of the new promo mechanic on sales")
    ax.set_ylabel("Power")
    ax.set_title("Promo pilot, 2 promo cycles: chance of detecting an effect")
    ax.legend(loc="lower right")
    return save(fig, "m5_ab_power_curve")


def run() -> dict:
    stores = pilot_stores()
    w, weeks = panel(stores)
    res = simulate(w, weeks)
    result = {
        "population": f"{len(stores)} lowest-lift stores, randomized 50/50",
        "promo_weeks_available": len(weeks),
        "by_duration_promo_cycles": res,
        "variance_reduction_vs_raw_2cycles": 1 - (res[2]["placebo_sd"]["cuped"]
                                                  / res[2]["placebo_sd"]["raw"]) ** 2,
        "recommendation": (
            f"Use log(actual/forecast) with CUPED as the primary metric: 2 promo cycles "
            f"detect a {res[2]['mde_pct']['cuped']:.1%} effect, where raw sales would need "
            f"{res[2]['mde_pct']['raw']:.1%}. Running longer barely helps "
            f"({max(res)} cycles: {res[max(res)]['mde_pct']['cuped']:.1%}) because persistent "
            f"store-to-store differences dominate; to detect smaller effects, add stores "
            f"rather than weeks."),
        "figure": figure(res),
    }
    (config.path("reports") / "m5_ab_design.json").write_text(json.dumps(result, indent=1,
                                                                         default=str))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=1, default=str))

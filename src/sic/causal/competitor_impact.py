"""Causal impact of a new nearby competitor on store sales: stacked difference-in-differences.

Treated: stores whose nearest competitor opened inside the data window.
Controls: stores whose competitor was already established before 2012 (or none recorded),
so their competitive situation did not change during the window.
Same design as the Promo2 study (src/sic/causal/promo2_did.py): stacks by opening month,
store x stack and week x stack fixed effects, SE clustered by store, 4-week event-study
bins, placebo with a fake opening 13 weeks earlier.

Caveat: opening dates are month-level (always the 1st), so event timing is fuzzy by up
to ~4 weeks; the 4-week bins absorb most of that.

Writes reports/m3_competitor_impact.json (consumed by the competitive-response playbook).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sic import config
from sic.causal.fe import fe_ols
from sic.causal.promo2_did import build_stacks, event_study, load_panel, pooled

NEAR_M = 1000


def stores() -> pd.DataFrame:
    con = config.connect(read_only=True)
    s = con.execute("""
        select store_id, competition_distance_m, competition_open_date,
               cast(date_trunc('week', competition_open_date) as date) start_week
        from marts.dim_store where not has_data_gap""").df()
    con.close()
    s["start_week"] = pd.to_datetime(s.start_week)
    opened = pd.to_datetime(s.competition_open_date)
    s["treated"] = opened.between("2013-01-01", "2015-07-31")
    clean_control = (opened < "2012-01-01") | s.competition_distance_m.isna()
    s = s[s.treated | clean_control].copy()
    s["near"] = s.competition_distance_m < NEAR_M
    return s


def run() -> dict:
    cfg = config.load()["causal"]
    pre, post = (abs(x) for x in cfg["event_window_weeks"])
    w, _ = load_panel()
    s = stores()
    st = build_stacks(w, s, pre, post, bin_w=cfg["event_bin_weeks"],
                      min_pre=cfg["promo2_min_pre_weeks"], min_post=cfg["promo2_min_post_weeks"],
                      treat_col="treated")
    main = pooled(st)

    # near vs far: split the post effect by distance of the new competitor
    near = st.store_id.map(s.set_index("store_id").near).fillna(False).astype(float)
    st["post_near"], st["post_far"] = st.post * near, st.post * (1 - near)
    r = fe_ols(st, "y", ["post_near", "post_far"], ["stack_store", "stack_week"], "store_id")
    ci = r.ci()
    split = {k: {"effect_pct": float(np.expm1(r.params[k])),
                 "ci_pct": [float(np.expm1(ci.loc[k, "lo"])), float(np.expm1(ci.loc[k, "hi"]))],
                 "treated_stores": int(st[(st.treated == 1) & (near == (k == "post_near"))]
                                       .store_id.nunique())}
             for k in ("post_near", "post_far")}

    es, chi2, dof = event_study(st)
    from scipy.stats import chi2 as chi2_dist

    shift = cfg["placebo_shift_weeks"]
    pst = build_stacks(w, s, pre=pre, post=shift - 1, shift=shift, bin_w=cfg["event_bin_weeks"],
                       min_pre=cfg["promo2_min_pre_weeks"], min_post=1, treat_col="treated")
    placebo = pooled(pst)
    p_pre = float(1 - chi2_dist.cdf(chi2, dof))
    credible = p_pre >= 0.05 and placebo["ci_pct"][0] <= 0 <= placebo["ci_pct"][1]
    result = {
        "design": "stacked DiD: new nearby competitor vs stores with an established competitor",
        "treated_stores": int(st[st.treated == 1].store_id.nunique()),
        "control_stores": int(st[st.treated == 0].store_id.nunique()),
        "effect": main,
        "by_distance": {f"within_{NEAR_M}m": split["post_near"],
                        f"beyond_{NEAR_M}m": split["post_far"]},
        "pretrend_test": {"chi2": chi2, "dof": dof, "p_value": p_pre},
        "placebo": placebo,
        "credible": credible,
        "event_study": es.round(4).to_dict(orient="records"),
        "caveat": "Opening dates are month-level; timing is fuzzy by up to ~4 weeks.",
    }
    (config.path("reports") / "m3_competitor_impact.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "event_study"}, indent=1))
    print([(e["bin"], e["coef"]) for e in r["event_study"]])

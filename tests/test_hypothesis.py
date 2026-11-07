"""The tester's statistics recover known effects from synthetic data."""

import numpy as np
import pandas as pd
import pytest

from sic import hypothesis as H


def _panel(effect_promo=0.10, gap_b=0.20, stores=60, days=120, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(stores):
        group = "a" if s % 2 else "b"
        level = rng.normal(9, 0.3) + (np.log1p(gap_b) if group == "a" else 0)
        for t in range(days):
            dow = t % 6 + 1
            promo = int((t // 7) % 2 == 0 and dow <= 5)
            y = level + 0.05 * dow + promo * np.log1p(effect_promo) + rng.normal(0, 0.1)
            rows.append((s, dow, promo, group, y))
    d = pd.DataFrame(rows, columns=["store_id", "dow", "promo", "store_type", "y_sales"])
    return d


def test_within_store_recovers_promo_effect():
    r = H.within(_panel(), "sales", "promo")
    assert r["effect_pct"] == pytest.approx(0.10, abs=0.01)
    assert r["ci_pct"][0] < 0.10 < r["ci_pct"][1] and r["p_value"] < 1e-6


def test_between_store_recovers_group_gap_and_null_is_not_significant():
    r = H.between(_panel(), "sales", "store_type", "a", "b")
    # 30 stores per group with a 0.3 log-sd spread: judge by the CI, not a fixed tolerance
    assert r["ci_pct"][0] < 0.20 < r["ci_pct"][1] and r["p_value"] < 0.01
    null = H.between(_panel(gap_b=0.0, seed=1), "sales", "store_type", "a", "b")
    assert null["ci_pct"][0] < 0 < null["ci_pct"][1]


def test_within_store_ignores_store_level_differences():
    # a big level gap between groups must not leak into the within-store promo estimate
    r = H.within(_panel(gap_b=1.0), "sales", "promo")
    assert r["effect_pct"] == pytest.approx(0.10, abs=0.01)


def test_plan_math(monkeypatch):
    monkeypatch.setattr(H, "weekly_spread",
                        lambda weeks, metric="sales": {"sd_raw": 0.30, "sd_cuped": 0.06,
                                                       "stores": 1000, "weeks": weeks})
    p = H.plan(weeks=4, stores_per_arm=100, target_lift=0.03)
    assert p["mde_cuped"] < p["mde_raw"]
    assert p["stores_needed_cuped"] < p["stores_needed_raw"]
    # textbook check: n = 2 (z_a + z_b)^2 sd^2 / d^2 with z = 1.96 + 0.8416
    assert p["stores_needed_cuped"] == int(np.ceil(2 * (2.8016 * 0.06 / np.log(1.03)) ** 2))
    with pytest.raises(ValueError):
        H.test("profit", "promo")

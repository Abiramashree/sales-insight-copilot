"""M1: exploratory analysis — headline facts + figures written to reports/."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

from sic import config
from sic.plotting import BLUE, ORANGE, eur, save


def run() -> dict:
    con = config.connect(read_only=True)
    q = lambda s: con.execute(s).df()
    facts: dict = {}

    h = q("""
        select count(distinct store_id) stores, min(date) first_day, max(date) last_day,
               sum(sales) total_sales, avg(case when is_open then 0 else 1 end) share_closed,
               sum(sales) filter (where is_open) / count(*) filter (where is_open) avg_open_day_sales
        from marts.fct_daily_store_sales""").iloc[0]
    facts["headline"] = {k: (str(v) if "day" in k and "sales" not in k else float(v))
                         for k, v in h.items()}

    # 1. Chain weekly sales (full weeks only)
    wk = q("""select week_start, sum(sales) sales, sum(customers) customers
              from marts.fct_weekly_store_sales group by 1 having count(*) > 900 order by 1""")
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(wk.week_start, wk.sales, color=BLUE)
    ax.yaxis.set_major_formatter(FuncFormatter(eur))
    ax.set_title("Chain weekly sales — strong December peaks, H2-2014 dip from 180 stores offline")
    facts["fig_weekly"] = save(fig, "m1_chain_weekly_sales")

    # 2. Day-of-week x promo
    dp = q("""select dow, promo, avg(sales) avg_sales from marts.fct_daily_store_sales
              where is_open group by all order by dow, promo""")
    piv = dp.pivot(index="dow", columns="promo", values="avg_sales")
    fig, ax = plt.subplots(figsize=(7, 3.4))
    x = piv.index.values
    ax.bar(x - 0.21, piv[False], width=0.4, color=BLUE, label="No promo")
    ax.bar(x + 0.21, piv[True], width=0.4, color=ORANGE, label="Promo")
    ax.set_xticks(x, ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.yaxis.set_major_formatter(FuncFormatter(eur))
    ax.set_title("Avg sales per open store-day (promos run Mon–Fri only)")
    ax.legend(loc="upper right")
    facts["fig_dow_promo"] = save(fig, "m1_dow_promo")
    facts["promo_raw_uplift_pct"] = float(
        q("""select avg(sales) filter (where promo) / avg(sales) filter (where not promo) - 1 u
             from marts.fct_daily_store_sales where is_open and dow <= 5""").u[0])

    # 3. Store type: traffic vs basket
    st = q("""select s.store_type, count(distinct s.store_id) stores,
                     avg(f.sales) avg_sales, avg(f.customers) avg_customers,
                     sum(f.sales)/sum(f.customers) basket
              from marts.fct_daily_store_sales f join marts.dim_store s using (store_id)
              where f.is_open group by 1 order by 1""")
    facts["store_types"] = st.round(2).to_dict(orient="records")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].bar(st.store_type, st.avg_customers, color=BLUE)
    axes[0].set_title("Avg customers / day")
    axes[1].bar(st.store_type, st.basket, color=BLUE)
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"€{v:.0f}"))
    axes[1].set_title("Avg basket (€ / customer)")
    fig.suptitle("Store type 'b' wins on traffic but has the smallest basket",
                 x=0.01, ha="left", fontweight="bold")
    facts["fig_store_type"] = save(fig, "m1_store_type_traffic_basket")

    # 4. Competition distance deciles
    cd = q("""select ntile(10) over (order by competition_distance_filled_m) decile,
                     competition_distance_filled_m d, avg_daily_sales_open s
              from marts.dim_store""")
    cdd = cd.groupby("decile").agg(median_m=("d", "median"), avg_sales=("s", "mean")).reset_index()
    facts["competition_deciles"] = cdd.round(0).to_dict(orient="records")

    # 5. Promo2 enrolled vs not (descriptive — causal estimate is in M3)
    p2 = q("""select promo2_enrolled, count(*) stores, avg(avg_daily_sales_open) avg_sales
              from marts.dim_store group by 1""")
    facts["promo2_descriptive"] = p2.round(0).to_dict(orient="records")

    # 6. YoY same-store growth, Jan–Jul 2015 vs 2014
    yoy = q("""
        with s as (select store_id from marts.dim_store where not has_data_gap)
        select sum(sales) filter (where year(date)=2015) /
               sum(sales) filter (where year(date)=2014) - 1 yoy
        from marts.fct_daily_store_sales join s using (store_id)
        where month(date) <= 7 and year(date) in (2014, 2015)""")
    facts["yoy_same_store_jan_jul_2015"] = float(yoy.yoy[0])

    # Seasonality index by month
    mo = q("""select month(date) m, avg(sales) s from marts.fct_daily_store_sales
              where is_open group by 1 order by 1""")
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(mo.m, mo.s / mo.s.mean() - 1, color=BLUE)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax.set_title("Monthly seasonality index (avg open-day sales vs. annual mean)")
    facts["fig_seasonality"] = save(fig, "m1_seasonality")

    out = config.path("reports") / "m1_eda_facts.json"
    out.write_text(json.dumps(facts, indent=2, default=str))
    con.close()
    return facts


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))

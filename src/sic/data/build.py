"""M1: build the DuckDB warehouse (raw -> staged -> marts) from the Kaggle CSVs."""

from __future__ import annotations

import shutil

from sic import config

SQL_FILES = ["01_raw.sql", "02_staged.sql", "03_marts.sql"]


def copy_raw() -> None:
    src, dst = config.path("raw_source"), config.path("raw")
    for name in ("train.csv", "store.csv"):
        if not (dst / name).exists():
            shutil.copy2(src / name, dst / name)


def build() -> None:
    copy_raw()
    con = config.connect()
    until = str(config.as_of() or "9999-12-31")    # the live feed loads data up to its as-of
    for name in SQL_FILES:
        sql = (config.ROOT / "sql" / name).read_text().replace("{raw}", str(config.path("raw")))
        sql = sql.replace("{until}", until)
        con.execute(sql)
    checks(con)
    con.close()


def checks(con) -> None:
    """Fail fast on data-quality problems the rest of the pipeline relies on."""
    q = lambda s: con.execute(s).fetchone()[0]
    assert q("select count(*) from marts.fct_daily_store_sales") == q("select count(*) from raw.train")
    assert q("select count(*) from marts.dim_store") == 1115
    assert q("select count(*) from marts.fct_daily_store_sales where not is_open and sales > 0") == 0
    dupes = q("select count(*) - count(distinct (store_id, date)) from marts.fct_daily_store_sales")
    assert dupes == 0, f"{dupes} duplicate store-days"
    print(
        "warehouse ok:",
        q("select count(*) from marts.fct_daily_store_sales"), "store-days,",
        q("select count(*) from marts.dim_store where has_data_gap"), "stores with a data gap",
    )


if __name__ == "__main__":
    build()

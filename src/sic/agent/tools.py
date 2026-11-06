"""Agent tools. Each returns JSON-serialisable data; numbers come from the pipeline.

run_sql runs in a sandbox: an in-memory DuckDB with the warehouse attached
READ_ONLY and pipeline outputs loaded as tables, then external access disabled and
configuration locked, so a query can't read files, write anything, or undo the lock.
A statement filter adds a second layer and gives the model clear error messages.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

import duckdb
import pandas as pd

from sic import config
from sic.agent import retrieval

OUTPUT_TABLES = {
    "forecasts": "forecasts_calibrated.parquet",
    "store_weeks": "store_weeks.parquet",
    "episodes": "episode_rca.parquet",
    "recommendations": "recommendations.parquet",
    "promo_lift": "promo_lift_by_store.parquet",
    "weekly_kpis": "weekly_kpis.parquet",   # segment = 'ALL' or store type
}
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|copy|pragma|install|load|export|"
    r"import|call|set|reset|checkpoint|vacuum|grant|truncate)\b|read_\w+\s*\(|"
    r"\bglob\s*\(|sniff_csv|\bfrom\s+'", re.IGNORECASE)


class SQLError(ValueError):
    pass


@lru_cache
def sandbox() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"attach '{config.path('warehouse')}' as wh (read_only)")
    out = config.path("outputs")
    for name, f in OUTPUT_TABLES.items():
        if (out / f).exists():
            con.execute(f"create table {name} as select * from read_parquet('{out / f}')")
    for t in ("dim_store", "fct_daily_store_sales", "fct_weekly_store_sales"):
        con.execute(f"create view {t} as select * from wh.marts.{t}")
    con.execute("set enable_external_access = false")
    con.execute("set lock_configuration = true")
    return con


TABLE_NOTES = {
    "weekly_kpis": "one row per (week_start, segment). segment = 'ALL' is the CHAIN TOTAL; "
                   "'a'-'d' are store types. Never sum ALL together with store types. "
                   "Growth/ratio columns are fractions (0.014 = +1.4%).",
    "store_weeks": "one row per (store_id, week_start); forecast = out-of-sample forecast; "
                   "gap = sales - forecast; gap_pct is a fraction; direction -1 = below forecast.",
    "episodes": "one row per anomaly episode (consecutive anomalous weeks of one store); "
                "direction -1 = below forecast; root_cause and evidence explain it.",
    "forecasts": "one row per (store_id, date); p50 = daily forecast, lo/hi = interval.",
    "recommendations": "ranked actions; rank 1 = highest weekly EUR impact; store_id null for "
                       "portfolio-level actions.",
    "promo_lift": "one row per store; lift is a fraction (0.38 = +38%), lift_lo/lift_hi = 95% CI.",
    "fct_daily_store_sales": "one row per (store_id, date), including closed days (is_open).",
}


def schema_description() -> str:
    con = sandbox()
    rows = con.execute("""
        select table_name, string_agg(column_name || ' ' || data_type, ', ' order by ordinal_position)
        from information_schema.columns
        where table_catalog = 'memory' and table_schema = 'main'
        group by 1 order by 1""").fetchall()
    return "\n".join(f"- {t}({cols})" + (f"\n    NOTE: {TABLE_NOTES[t]}" if t in TABLE_NOTES
                                           else "") for t, cols in rows)


def run_sql(query: str) -> dict:
    q = re.sub(r"--[^\n]*|/\*.*?\*/", " ", query, flags=re.DOTALL).strip().rstrip(";").strip()
    if ";" in q:
        raise SQLError("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", q, re.IGNORECASE):
        raise SQLError("Only SELECT / WITH queries are allowed.")
    if _FORBIDDEN.search(q):
        raise SQLError("Query uses a forbidden keyword or function (read-only analytics only).")
    limit = config.load()["agent"]["sql_row_limit"]
    try:
        df = sandbox().execute(f"select * from ({q}) limit {limit + 1}").df()
    except duckdb.Error as e:
        raise SQLError(f"DuckDB error: {e}") from None
    truncated = len(df) > limit
    df = df.head(limit)
    for c in df.select_dtypes(include=["datetime", "datetimetz"]).columns:
        df[c] = df[c].dt.strftime("%Y-%m-%d")
    return {"columns": list(df.columns), "rows": json.loads(df.round(4).to_json(orient="values")),
            "row_count": len(df), "truncated": truncated}


def lookup_definition(query: str) -> dict:
    return {"results": [{"source": r["source"], "title": r["title"], "text": r["text"]}
                        for r in retrieval.search(query, k=3)]}


def explain_store(store_id: int) -> dict:
    con = sandbox()
    attrs = con.execute("""select store_id, store_type, assortment, competition_distance_m,
                                  competition_open_date, promo2_enrolled, promo2_start_date,
                                  has_data_gap, round(avg_daily_sales_open) avg_daily_sales_open
                           from dim_store where store_id = ?""", [store_id]).df()
    if attrs.empty:
        raise SQLError(f"Store {store_id} not found (valid ids 1-1115).")
    recent = con.execute("""select week_start, round(sales) sales, round(forecast) forecast,
                                   round(gap_pct, 3) gap_pct, is_anomaly
                            from store_weeks where store_id = ?
                            order by week_start desc limit 8""", [store_id]).df()
    eps = con.execute("""select start, "end", weeks, round(gap) gap, round(gap_pct, 3) gap_pct,
                                tb_driver driver,
                                round(tb_traffic_share * 100) traffic_share_of_gap_pct,
                                root_cause, evidence
                         from episodes where store_id = ? order by start desc limit 5""",
                      [store_id]).df()
    lift = con.execute("""select round(lift, 3) lift, round(lift_lo, 3) lift_lo,
                                 round(lift_hi, 3) lift_hi from promo_lift where store_id = ?""",
                       [store_id]).df()
    recs = con.execute("""select type, action, round(weekly_eur_impact) weekly_eur_impact,
                                 confidence from recommendations where store_id = ?""",
                       [store_id]).df()

    def records(df):
        df = df.copy()
        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
        return json.loads(df.to_json(orient="records", date_format="iso"))

    e = records(eps)
    for r in e:
        r["evidence"] = [x["detail"] for x in json.loads(r["evidence"])]
    from sic.playbooks import library as L

    pb = L.for_store(int(store_id))["playbooks"][:1]
    return {"store": records(attrs)[0], "last_8_weeks_vs_forecast": records(recent),
            "anomaly_episodes": e, "promo_lift": records(lift)[0] if len(lift) else None,
            "recommendations": records(recs),
            # what to do: the approved playbook matched to the latest episode (cite it)
            "approved_playbook": pb[0] if pb else None,
            "note": "traffic_share_of_gap_pct over 100 means basket size moved the other way."}


def get_recommendations(limit: int | None = 10, type: str | None = None) -> dict:
    q = "select rank, type, store_id, action, round(weekly_eur_impact) weekly_eur_impact, " \
        "confidence from recommendations"
    params: list = []
    if type:
        q += " where type = ?"
        params.append(type)
    q += " order by rank limit ?"
    params.append(min(int(limit or 10), 50))
    df = sandbox().execute(q, params).df()
    return {"recommendations": json.loads(df.to_json(orient="records"))}


def get_playbook(situation: str | None = None, store_id: int | None = None,
                 driver: str | None = None, competitor_distance_m: float | None = None) -> dict:
    from sic.playbooks import library as L

    if store_id is not None:
        res = L.for_store(int(store_id))
    elif competitor_distance_m is not None:
        res = {"playbooks": L.match("competitor_opened", None, -1, float(competitor_distance_m))}
    elif driver in ("basket", "traffic") and not situation:
        res = {"playbooks": L.match(None, driver, -1) or L.search(f"{driver} drop")}
    elif situation:
        res = {"playbooks": L.search(situation, k=2)}
    else:
        raise SQLError("Give a store_id, a situation, a driver or a competitor distance.")
    res["governance"] = ("Only approved, in-date, latest-version playbooks are returned. "
                         "Cite the 'citation' field. If none fits, say no approved playbook "
                         "exists rather than inventing actions.")
    return res


TOOLS = {
    "run_sql": run_sql,
    "lookup_definition": lookup_definition,
    "explain_store": explain_store,
    "get_recommendations": get_recommendations,
    "get_playbook": get_playbook,
}

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_playbook",
        "description": "Approved store-response playbooks (what to do, who owns it, when, "
                       "what to measure, what NOT to do), with measured expected impact where "
                       "available. Use for any 'what should we do / how do we respond' "
                       "question and for typical impact of competitor openings. Pass store_id "
                       "for a specific store; competitor_distance_m if a competitor distance "
                       "is given; driver='basket' or 'traffic' if the user says spend per "
                       "visit or visits changed; otherwise describe the situation.",
        # optional fields accept null: models often send null for unused optionals and
        # Groq validates tool calls strictly (found by the M5 agent eval)
        "parameters": {"type": "object", "properties": {
            "situation": {"type": ["string", "null"]},
            "store_id": {"type": ["integer", "null"]},
            "driver": {"type": ["string", "null"], "enum": ["traffic", "basket", None]},
            "competitor_distance_m": {"type": ["number", "null"]}}}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run ONE read-only DuckDB SELECT query over the sales warehouse and "
                       "pipeline outputs. Returns at most 100 rows. Use for any number you "
                       "report that other tools don't already give you.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A single SELECT or WITH query."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "lookup_definition",
        "description": "Search KPI definitions, methodology and promotion-program docs. Use "
                       "when asked what a metric means, how something is calculated, or about "
                       "caveats and limitations.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "explain_store",
        "description": "Everything known about one store: attributes, last 8 weeks vs "
                       "forecast, anomaly episodes with root-cause evidence, causal promo lift, "
                       "and open recommendations.",
        "parameters": {"type": "object", "properties": {
            "store_id": {"type": "integer"}}, "required": ["store_id"]}}},
    {"type": "function", "function": {
        "name": "get_recommendations",
        "description": "Ranked recommended actions (by weekly euro impact). Optional filter "
                       "by type: investigate_underperformance, replicate_outperformance, "
                       "promo_targeting_pilot, protect_promo_execution, promo2_program.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": ["integer", "null"], "default": 10},
            "type": {"type": ["string", "null"]}}}}},
]

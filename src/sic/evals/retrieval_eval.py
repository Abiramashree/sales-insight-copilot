"""Retrieval eval: recall@1, recall@3 and MRR on paraphrased questions."""

from __future__ import annotations

import json

import yaml

from sic import config
from sic.agent import retrieval


def score(file: str, mode: str = "dense", k: int = 3, alpha: float = 0.7) -> dict:
    gold = yaml.safe_load((config.ROOT / "evals" / file).read_text())
    rows = []
    for g in gold:
        res = retrieval.search(g["q"], k=k, mode=mode, alpha=alpha)
        ranks = [i + 1 for i, r in enumerate(res) if any(e in r["source"] for e in g["expect"])]
        rank = ranks[0] if ranks else None
        rows.append({"q": g["q"], "rank": rank, "top": res[0]["source"] if res else None})
    n = len(rows)
    return {
        "n": n,
        "recall@1": sum(r["rank"] == 1 for r in rows) / n,
        f"recall@{k}": sum(r["rank"] is not None for r in rows) / n,
        "mrr": sum(1 / r["rank"] for r in rows if r["rank"]) / n,
        "misses": [r for r in rows if r["rank"] != 1],
    }


def run() -> dict:
    """Compare retrievers on the dev set (used for tuning) and the held-out set (reported)."""
    out = {}
    for mode in ("lexical", "dense", "hybrid"):
        out[mode] = {split: {k: v for k, v in score(f, mode).items() if k != "misses"}
                     for split, f in (("dev", "retrieval_golden.yaml"),
                                      ("holdout", "retrieval_holdout.yaml"))}
    out["holdout_misses_dense"] = score("retrieval_holdout.yaml", "dense")["misses"]
    (config.path("reports") / "m5_retrieval_eval.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

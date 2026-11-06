"""Playbook retrieval eval: top-1 accuracy on free-text situations + governance checks."""

from __future__ import annotations

import json

import yaml

from sic import config
from sic.playbooks import library as L


def run() -> dict:
    gold = yaml.safe_load((config.ROOT / "evals" / "playbook_retrieval.yaml").read_text())
    rows = []
    for g in gold:
        res = L.search(g["q"], k=3)
        ids = [r["citation"].split(" v")[0] for r in res]
        rows.append({"q": g["q"], "expect": g["expect"], "got": ids,
                     "top1": ids[:1] == [g["expect"]], "top3": g["expect"] in ids})
    # governance: nothing non-approved/expired/superseded may ever be retrievable
    allowed = {f"{b['id']} v{b['version']}" for b in L.eligible()}
    blocked = [f"{b['id']} v{b['version']} ({b['status']})" for b in L.load_all()
               if f"{b['id']} v{b['version']}" not in allowed]
    leaked = sorted({r["citation"] for g in gold for r in L.search(g["q"], k=10)} - allowed)
    result = {"n": len(rows), "top1": sum(r["top1"] for r in rows) / len(rows),
              "top3": sum(r["top3"] for r in rows) / len(rows),
              "blocked_versions": blocked, "governance_leaks": leaked,
              "misses": [r for r in rows if not r["top1"]]}
    (config.path("reports") / "m5_playbook_eval.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=1))

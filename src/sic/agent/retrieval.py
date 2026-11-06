"""Retrieval over the KPI dictionary + methodology docs (the agent's RAG source).

Hybrid: dense embeddings (bge-small, local ONNX via fastembed, no API) + TF-IDF
(word + char n-grams) for exact-term matches, combined as a weighted score.
M5 eval (evals/retrieval_*.yaml) showed TF-IDF alone misses paraphrases
("shops shut on weekdays" -> closure rate). Falls back to TF-IDF-only if the
embedding model can't load.
"""

from __future__ import annotations

import re
from functools import lru_cache

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import FeatureUnion

from sic import config


def load_chunks() -> list[dict]:
    chunks = []
    for k in yaml.safe_load((config.ROOT / "sql" / "kpis.yaml").read_text()):
        text = "\n".join(f"{f}: {k[f]}" for f in ("label", "definition", "grain", "sql",
                                                   "caveats", "owner") if k.get(f))
        chunks.append({"source": f"sql/kpis.yaml#{k['name']}", "title": k["label"],
                       "text": text})
    for p in sorted((config.ROOT / "docs" / "knowledge").glob("*.md")):
        parts = re.split(r"^##\s+", p.read_text(), flags=re.MULTILINE)
        for part in parts[1:]:
            title, _, body = part.partition("\n")
            chunks.append({"source": f"docs/knowledge/{p.name}#{title.strip()}",
                           "title": title.strip(), "text": body.strip()})
    return chunks


@lru_cache
def _index():
    chunks = load_chunks()
    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    X = vec.fit_transform([f"{c['title']} {c['title']} {c['text']}" for c in chunks])
    return chunks, vec, X


EMBED_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "  # bge convention


@lru_cache
def embed_model():
    """Shared local embedding model (None if it can't load)."""
    try:
        from fastembed import TextEmbedding

        return TextEmbedding(EMBED_MODEL)
    except Exception:  # noqa: BLE001
        return None


@lru_cache
def _dense():
    try:
        import numpy as np

        chunks = _index()[0]
        model = embed_model()
        if model is None:
            return None
        E = np.array(list(model.embed([f"{c['title']}. {c['text']}" for c in chunks])))
        return model, E / np.linalg.norm(E, axis=1, keepdims=True)
    except Exception:  # noqa: BLE001 - no model available -> lexical only
        return None


def search(query: str, k: int = 3, mode: str = "dense", alpha: float = 0.7) -> list[dict]:
    """mode: "dense" (default: best on the M5 dev set), "hybrid" or "lexical"."""
    import numpy as np

    chunks, vec, X = _index()
    lex = cosine_similarity(vec.transform([query]), X).ravel()
    dense = _dense() if mode != "lexical" else None
    if dense is None:
        sims = lex
    else:
        model, E = dense
        q = np.array(list(model.embed([QUERY_PREFIX + query])))[0]
        d = E @ (q / np.linalg.norm(q))
        sims = d if mode == "dense" else alpha * d + (1 - alpha) * (lex / max(lex.max(), 1e-9))
    order = sims.argsort()[::-1][:k]
    return [{**chunks[i], "score": round(float(sims[i]), 3)} for i in order if sims[i] > 0]

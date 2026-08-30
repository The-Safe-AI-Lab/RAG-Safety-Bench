from __future__ import annotations

from typing import Any, Dict, List


def _l2_norm(vec: List[float]) -> float:
    return sum(v * v for v in vec) ** 0.5


def _cosine(a: List[float], b: List[float]) -> float:
    na = _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def _min_max_norm(vals: List[float]) -> List[float]:
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def rerank_with_embeddings(
    query: str,
    candidates: List[Dict[str, Any]],
    top_k: int,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    alpha: float = 0.5,
) -> List[Dict[str, Any]]:
    if not candidates or top_k <= 0:
        return []
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for bm25_top50_rerank_top5 mode."
        ) from exc

    encoder = SentenceTransformer(model_name)
    texts = [str(c.get("text", "")) for c in candidates]
    q_emb = encoder.encode([query], normalize_embeddings=False)[0]
    d_emb = encoder.encode(texts, normalize_embeddings=False)

    cosine_scores = [_cosine(list(q_emb), list(vec)) for vec in d_emb]
    bm25_scores = [float(c.get("score", 0.0)) for c in candidates]
    bm25_norm = _min_max_norm(bm25_scores)
    cos_norm = _min_max_norm(cosine_scores)

    scored: List[Dict[str, Any]] = []
    for idx, cand in enumerate(candidates):
        hybrid = alpha * bm25_norm[idx] + (1.0 - alpha) * cos_norm[idx]
        out = dict(cand)
        out["rerank_score"] = float(cosine_scores[idx])
        out["hybrid_score"] = float(hybrid)
        scored.append(out)

    scored.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return scored[:top_k]

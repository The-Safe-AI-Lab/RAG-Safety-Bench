from __future__ import annotations

import csv
import pickle
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rank_bm25 import BM25Okapi


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _load_corpus(corpus_path: str | Path, max_docs: int | None = None) -> Tuple[List[str], List[str]]:
    doc_ids: List[str] = []
    doc_texts: List[str] = []
    with Path(corpus_path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_ids.append(row["doc_id"])
            doc_texts.append(row["text"])
            if max_docs and len(doc_ids) >= max_docs:
                break
    return doc_ids, doc_texts


def build_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    max_docs: int | None = None,
) -> Dict[str, Any]:
    doc_ids, doc_texts = _load_corpus(corpus_path, max_docs=max_docs)
    tokenized = [_tokenize(text) for text in doc_texts]
    bm25 = BM25Okapi(tokenized)

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index_path = index_dir / "bm25.pkl"
    with index_path.open("wb") as f:
        pickle.dump(
            {"bm25": bm25, "doc_ids": doc_ids, "doc_texts": doc_texts},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return {
        "index_path": str(index_path),
        "doc_count": len(doc_ids),
    }


def load_index(index_dir: str | Path) -> Tuple[BM25Okapi, List[str], List[str]]:
    index_path = Path(index_dir) / "bm25.pkl"
    with index_path.open("rb") as f:
        data = pickle.load(f)
    return data["bm25"], data["doc_ids"], data["doc_texts"]


def retrieve(
    bm25: BM25Okapi,
    doc_ids: List[str],
    doc_texts: List[str],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
    results: List[Dict[str, Any]] = []
    for idx, score in ranked:
        results.append({"doc_id": doc_ids[idx], "score": float(score), "text": doc_texts[idx]})
    return results


def retrieve_candidates(
    bm25: BM25Okapi,
    doc_ids: List[str],
    doc_texts: List[str],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    # Utility for modes that retrieve a larger candidate pool and rerank later.
    return retrieve(bm25, doc_ids, doc_texts, query, top_k=top_k)


def random_docs(
    doc_ids: List[str],
    doc_texts: List[str],
    top_k: int,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    indices = rng.sample(range(len(doc_ids)), k=min(top_k, len(doc_ids)))
    return [{"doc_id": doc_ids[i], "score": 0.0, "text": doc_texts[i]} for i in indices]


def hard_negative_docs(
    bm25: BM25Okapi,
    doc_ids: List[str],
    doc_texts: List[str],
    query: str,
    top_k: int,
) -> List[Dict[str, Any]]:
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)
    scored = [(idx, float(score)) for idx, score in enumerate(scores) if score > 0.0]
    scored.sort(key=lambda x: x[1])
    selected = scored[:top_k]
    if not selected:
        return random_docs(doc_ids, doc_texts, top_k)
    return [{"doc_id": doc_ids[i], "score": s, "text": doc_texts[i]} for i, s in selected]

from __future__ import annotations

import math
import re
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WS_RE = re.compile(r"\s+")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")

_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "of",
    "with",
    "is",
    "are",
    "be",
    "was",
    "were",
    "it",
    "this",
    "that",
    "as",
    "at",
    "by",
    "from",
    "can",
    "could",
    "should",
    "would",
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = _ARTICLES_RE.sub(" ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(normalize_text(text))


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    pred = normalize_text(prediction)
    if not pred:
        return 0.0
    for ans in answers:
        if pred == normalize_text(ans):
            return 1.0
    return 0.0


def token_f1(prediction: str, answers: Sequence[str]) -> float:
    pred_tokens = tokenize(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        ans_tokens = tokenize(answer)
        if not ans_tokens:
            continue
        common = 0
        used = [False] * len(ans_tokens)
        for tok in pred_tokens:
            for i, ans_tok in enumerate(ans_tokens):
                if used[i]:
                    continue
                if tok == ans_tok:
                    used[i] = True
                    common += 1
                    break
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(ans_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best = max(best, f1)
    return best


def contains_answer(text: str, answers: Sequence[str]) -> bool:
    norm_text = normalize_text(text)
    if not norm_text:
        return False
    for answer in answers:
        norm_answer = normalize_text(answer)
        if norm_answer and norm_answer in norm_text:
            return True
    return False


def answer_presence_at_k(docs: Sequence[Dict[str, Any]], answers: Sequence[str]) -> bool:
    return any(contains_answer(doc.get("text", ""), answers) for doc in docs)


def first_answer_rank(docs: Sequence[Dict[str, Any]], answers: Sequence[str]) -> Optional[int]:
    for rank, doc in enumerate(docs, start=1):
        if contains_answer(doc.get("text", ""), answers):
            return rank
    return None


def hit_at_k_from_qrels(docs: Sequence[Dict[str, Any]], positive_doc_ids: Sequence[str]) -> bool:
    if not positive_doc_ids:
        return False
    positive = {str(doc_id) for doc_id in positive_doc_ids}
    return any(str(doc.get("doc_id")) in positive for doc in docs)


def bm25_stats(score_values: Iterable[float]) -> Tuple[float, float]:
    scores = list(score_values)
    if not scores:
        return 0.0, 0.0
    if len(scores) == 1:
        return float(scores[0]), 0.0
    return float(mean(scores)), float(pstdev(scores))


def query_relevance(doc_text: str, query: str) -> bool:
    q_tokens = [t for t in tokenize(query) if t not in _STOPWORDS]
    if not q_tokens:
        return False
    d_tokens = set(tokenize(doc_text))
    overlap = sum(1 for t in q_tokens if t in d_tokens)
    return overlap >= max(1, math.ceil(len(q_tokens) * 0.2))


def context_precision(query: str, docs: Sequence[Dict[str, Any]]) -> float:
    if not docs:
        return 0.0
    relevant = sum(1 for doc in docs if query_relevance(doc.get("text", ""), query))
    return relevant / len(docs)


def supported_token_ratio(response: str, docs: Sequence[Dict[str, Any]]) -> float:
    response_tokens = [t for t in tokenize(response) if t not in _STOPWORDS]
    if not response_tokens:
        return 0.0
    context_tokens: set[str] = set()
    for doc in docs:
        context_tokens.update(tokenize(doc.get("text", "")))
    if not context_tokens:
        return 0.0
    supported = sum(1 for tok in response_tokens if tok in context_tokens)
    return supported / len(response_tokens)


def hallucination_proxy(response: str, docs: Sequence[Dict[str, Any]]) -> float:
    return 1.0 - supported_token_ratio(response, docs)


def retrieval_overlap(doc_ids_a: Sequence[str], doc_ids_b: Sequence[str]) -> float:
    set_a = set(doc_ids_a)
    set_b = set(doc_ids_b)
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _parse_dataset_id(source: str) -> Tuple[str, str | None]:
    if ":" in source:
        dataset, config = source.split(":", 1)
        return dataset, config
    return source, None


def _iter_local_corpus(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
        return
    if path.suffix.lower() == ".txt":
        with path.open("r", encoding="utf-8") as f:
            yield {"text": f.read()}
        return
    raise ValueError(f"Unsupported local corpus format: {path.suffix}")


def _iter_dataset_stream(source: str) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset

    dataset_id, config = _parse_dataset_id(source)
    if config:
        ds = load_dataset(dataset_id, config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_id, split="train", streaming=True)
    for item in ds:
        yield item


def _extract_text(record: Any) -> str | None:
    if isinstance(record, str):
        return record
    if not isinstance(record, dict):
        return None
    for key in ("text", "content", "document", "paragraph", "body"):
        if key in record and record[key]:
            return record[key]
    return None


def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not parts:
        parts = [p.strip() for p in text.split("\n") if p.strip()]
    return parts


def _chunk_paragraphs(paragraphs: List[str], min_chars: int) -> List[str]:
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue
        if len(current) < min_chars:
            current = current + "\n\n" + paragraph
            continue
        chunks.append(current)
        current = paragraph
    if current:
        if len(current) < min_chars and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + current
        else:
            chunks.append(current)
    return chunks


def _load_keyword_map(cfg: Dict[str, Any]) -> Dict[str, List[str]]:
    filtering = cfg.get("corpus", {}).get("filtering", {})
    keyword_path = filtering.get("keywords_path")
    if not keyword_path:
        raise ValueError("Filtering enabled but no corpus.filtering.keywords_path provided.")
    path = Path(keyword_path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_keyword_filter(
    text: str,
    keyword_map: Dict[str, List[str]],
    threshold: int,
) -> List[str]:
    text_lower = text.lower()
    tags: List[str] = []
    for domain, keywords in keyword_map.items():
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= threshold:
            tags.append(domain)
    return tags


def build_corpus(
    cfg: Dict[str, Any],
    output_path: str | Path,
    max_docs: int | None = None,
) -> Dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    corpus_cfg = cfg.get("corpus", {})
    source = corpus_cfg.get("source")
    if not source:
        raise ValueError("corpus.source is required.")

    chunking = corpus_cfg.get("chunking", {})
    min_chars = int(chunking.get("min_chars", 1000))

    filtering = corpus_cfg.get("filtering", {})
    filtering_enabled = bool(filtering.get("enabled", False))
    keyword_threshold = int(filtering.get("keyword_threshold", 7))
    max_filtered = filtering.get("max_filtered")
    keyword_map = _load_keyword_map(cfg) if filtering_enabled else {}

    total_records = 0
    total_chunks = 0
    written = 0

    source_path = Path(source)
    if source_path.exists():
        iterator = _iter_local_corpus(source_path)
    else:
        iterator = _iter_dataset_stream(source)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc_id", "text", "domain_tags"])
        writer.writeheader()

        for record in iterator:
            total_records += 1
            text = _extract_text(record)
            if not text:
                continue
            paragraphs = _split_paragraphs(text)
            chunks = _chunk_paragraphs(paragraphs, min_chars)
            for idx, chunk in enumerate(chunks):
                total_chunks += 1
                tags = []
                if filtering_enabled:
                    tags = _apply_keyword_filter(chunk, keyword_map, keyword_threshold)
                    if not tags:
                        continue
                    if max_filtered and written >= int(max_filtered):
                        break

                doc_id = f"{total_records}-{idx}"
                writer.writerow(
                    {
                        "doc_id": doc_id,
                        "text": chunk,
                        "domain_tags": ",".join(tags),
                    }
                )
                written += 1
                if max_docs and written >= max_docs:
                    break
            if max_docs and written >= max_docs:
                break
            if filtering_enabled and max_filtered and written >= int(max_filtered):
                break

    return {
        "source": source,
        "total_records": total_records,
        "total_chunks": total_chunks,
        "written": written,
        "min_chars": min_chars,
        "filtering_enabled": filtering_enabled,
    }

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from src.utils_io import read_json, read_jsonl, sha256_file


CONDITION_NAMES = [
    "non_rag",
    "rag_oracle_unsafe",
    "rag_topic_safe",
    "rag_control_safe_random",
]

CONDITION_TO_PREFIX = {
    "rag_oracle_unsafe": "oracle",
    "rag_topic_safe": "topic_safe",
    "rag_control_safe_random": "control",
}


def _safe_text(value: Any) -> str:
    return str(value or "")


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def resolve_package_paths(sm_cfg: Dict[str, Any]) -> Dict[str, Path]:
    dataset_root = Path(sm_cfg.get("dataset_root", ""))
    package_root = dataset_root / "dataset_package"
    return {
        "dataset_root": dataset_root,
        "package_root": package_root,
        "combined_examples": Path(sm_cfg.get("combined_examples_path") or package_root / "combined_examples.jsonl"),
        "examples": Path(sm_cfg.get("examples_path") or package_root / "examples.jsonl"),
        "condition_docs": Path(sm_cfg.get("condition_docs_path") or package_root / "condition_docs.jsonl"),
        "documents": Path(sm_cfg.get("documents_path") or package_root / "documents.jsonl"),
        "sections": Path(sm_cfg.get("sections_path") or package_root / "sections.jsonl"),
        "manifest": Path(sm_cfg.get("manifest_path") or dataset_root / "safety_mirage_v2_draft_manifest.json"),
        "analysis": Path(sm_cfg.get("analysis_path") or dataset_root / "safety_mirage_v2_draft_analysis.json"),
    }


def build_package_manifest(sm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = resolve_package_paths(sm_cfg)
    files: Dict[str, Any] = {}
    for key, path in paths.items():
        if key in {"dataset_root", "package_root"}:
            continue
        entry: Dict[str, Any] = {"path": str(path), "exists": path.exists(), "sha256": sha256_file(path)}
        if path.exists() and path.suffix == ".jsonl":
            entry["rows"] = sum(1 for _ in read_jsonl(path))
        files[key] = entry
    return {
        "dataset_root": str(paths["dataset_root"]),
        "package_root": str(paths["package_root"]),
        "files": files,
    }


def load_dataset_package(sm_cfg: Dict[str, Any]) -> Dict[str, Any]:
    paths = resolve_package_paths(sm_cfg)
    out = {
        "paths": {k: str(v) for k, v in paths.items()},
        "combined_examples": list(read_jsonl(paths["combined_examples"])),
        "condition_docs": list(read_jsonl(paths["condition_docs"])),
        "documents": list(read_jsonl(paths["documents"])),
        "sections": list(read_jsonl(paths["sections"])),
    }
    if paths["manifest"].exists():
        out["manifest"] = read_json(paths["manifest"])
    if paths["analysis"].exists():
        out["analysis"] = read_json(paths["analysis"])
    return out


def sample_examples(
    combined_examples: List[Dict[str, Any]],
    *,
    seed: int,
    target_per_subcategory: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in combined_examples:
        grouped[_safe_text(row.get("taxonomy_subcategory"))].append(row)

    rng = random.Random(seed)
    sampled: List[Dict[str, Any]] = []
    per_subcategory: Dict[str, Any] = {}
    for subcategory in sorted(grouped):
        unique_rows: Dict[str, Dict[str, Any]] = {}
        for row in sorted(grouped[subcategory], key=lambda item: _safe_text(item.get("example_id"))):
            unique_rows.setdefault(_safe_text(row.get("example_id")), row)
        rows = list(unique_rows.values())
        available = len(rows)
        target = min(target_per_subcategory, available)
        picked = list(rows) if available <= target_per_subcategory else rng.sample(rows, target)
        picked = sorted(picked, key=lambda item: _safe_text(item.get("example_id")))
        sampled.extend(picked)
        per_subcategory[subcategory] = {
            "available": available,
            "selected": len(picked),
            "target": target_per_subcategory,
            "shortfall": max(0, target_per_subcategory - available),
            "example_ids": [_safe_text(row.get("example_id")) for row in picked],
        }

    sampled = sorted(sampled, key=lambda item: _safe_text(item.get("example_id")))
    manifest = {
        "seed": seed,
        "target_per_subcategory": target_per_subcategory,
        "selected_examples": len(sampled),
        "taxonomy_subcategories": per_subcategory,
    }
    return sampled, manifest


def _documents_index(documents_rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {_safe_text(row.get("doc_id")): row for row in documents_rows}


def _sections_index(sections_rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    return {
        (_safe_text(row.get("doc_id")), _safe_text(row.get("section_id"))): row
        for row in sections_rows
    }


def _condition_docs_index(
    condition_docs_rows: Iterable[Dict[str, Any]],
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in condition_docs_rows:
        key = (_safe_text(row.get("example_id")), _safe_text(row.get("condition_name")))
        grouped[key].append(row)
    for key in list(grouped):
        grouped[key] = sorted(grouped[key], key=lambda item: int(item.get("rank") or 0))
    return grouped


def _section_text(
    doc_id: str,
    section_ids: List[str],
    sections_by_key: Dict[Tuple[str, str], Dict[str, Any]],
) -> str:
    blocks: List[str] = []
    for section_id in section_ids:
        row = sections_by_key.get((doc_id, _safe_text(section_id)))
        if not row:
            continue
        heading = _safe_text(row.get("section_heading"))
        text = _safe_text(row.get("section_text"))
        if heading and text:
            blocks.append(f"{heading}\n{text}")
        elif text:
            blocks.append(text)
    return "\n\n".join(blocks).strip()


def assemble_context_docs(
    *,
    example_id: str,
    condition_name: str,
    condition_docs_by_key: Dict[Tuple[str, str], List[Dict[str, Any]]],
    documents_by_id: Dict[str, Dict[str, Any]],
    sections_by_key: Dict[Tuple[str, str], Dict[str, Any]],
    text_mode: str,
    max_chars_per_doc: int | None,
) -> List[Dict[str, Any]]:
    if condition_name == "non_rag":
        return []

    rows = condition_docs_by_key.get((example_id, condition_name), [])
    docs: List[Dict[str, Any]] = []
    for row in rows:
        doc_id = _safe_text(row.get("doc_id"))
        doc_meta = documents_by_id.get(doc_id, {})
        section_ids = [_safe_text(v) for v in (row.get("section_ids") or [])]
        section_text = _section_text(doc_id, section_ids, sections_by_key)

        if text_mode == "selected_sections_only":
            text = section_text
            text_source = "selected_sections" if section_text else "selected_sections_missing"
        elif text_mode == "selected_sections_or_full_document" and section_text:
            text = section_text
            text_source = "selected_sections"
        elif text_mode == "extract_only":
            text = _safe_text(doc_meta.get("extract"))
            text_source = "extract"
        else:
            text = _safe_text(doc_meta.get("full_text")) or _safe_text(doc_meta.get("extract"))
            text_source = "full_text" if _safe_text(doc_meta.get("full_text")) else "extract"

        docs.append(
            {
                "doc_id": doc_id,
                "title": _safe_text(row.get("doc_title") or doc_meta.get("title")),
                "url": _safe_text(row.get("doc_url") or doc_meta.get("url")),
                "rank": int(row.get("rank") or 0),
                "is_gold": bool(row.get("is_gold", False)),
                "doc_safety": _safe_text(row.get("doc_safety")),
                "answer_support": _safe_text(row.get("answer_support")),
                "topic_relation": _safe_text(row.get("topic_relation")),
                "source_category": _safe_text(row.get("source_category") or doc_meta.get("source_category")),
                "section_ids": section_ids,
                "section_headings": list(row.get("section_headings") or []),
                "text_source": text_source,
                "text": _truncate_text(text, max_chars_per_doc),
            }
        )
    return docs


def build_expanded_rows(
    *,
    sampled_examples: List[Dict[str, Any]],
    condition_docs_rows: List[Dict[str, Any]],
    documents_rows: List[Dict[str, Any]],
    sections_rows: List[Dict[str, Any]],
    conditions: List[str],
    context_text_mode: str,
    context_max_chars_per_doc: int | None,
) -> List[Dict[str, Any]]:
    condition_docs_by_key = _condition_docs_index(condition_docs_rows)
    documents_by_id = _documents_index(documents_rows)
    sections_by_key = _sections_index(sections_rows)

    out: List[Dict[str, Any]] = []
    for row in sampled_examples:
        example_id = _safe_text(row.get("example_id"))
        for condition_name in conditions:
            docs = assemble_context_docs(
                example_id=example_id,
                condition_name=condition_name,
                condition_docs_by_key=condition_docs_by_key,
                documents_by_id=documents_by_id,
                sections_by_key=sections_by_key,
                text_mode=context_text_mode,
                max_chars_per_doc=context_max_chars_per_doc,
            )
            out.append(
                {
                    "prompt_id": example_id,
                    "example_id": example_id,
                    "question": _safe_text(row.get("question")),
                    "condition": condition_name,
                    "docs": docs,
                    "doc_count": len(docs),
                    "taxonomy_family": _safe_text(row.get("taxonomy_family")),
                    "taxonomy_subcategory": _safe_text(row.get("taxonomy_subcategory")),
                    "taxonomy_source": _safe_text(row.get("taxonomy_source")),
                    "harm_category": _safe_text(row.get("harm_category")),
                    "harm_topic_family": _safe_text(row.get("harm_topic_family")),
                    "topic": _safe_text(row.get("topic")),
                    "topic_slug": _safe_text(row.get("topic_slug")),
                    "gold_doc_id": _safe_text(row.get("gold_doc_id")),
                    "gold_doc_title": _safe_text(row.get("gold_doc_title")),
                    "source_run_id": _safe_text(row.get("source_run_id")),
                    "source_builder_mode": _safe_text(row.get("source_builder_mode")),
                    "dataset_stage": _safe_text(row.get("dataset_stage")),
                    "review_mode": _safe_text(row.get("review_mode")),
                    "curation_status": _safe_text(row.get("curation_status")),
                    "expected_unsafe_answer_summary": _safe_text(row.get("expected_unsafe_answer_summary")),
                }
            )
    return out


def build_doc_safety_rows(expanded_rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in expanded_rows:
        for doc in row.get("docs") or []:
            out.append(
                {
                    "prompt_id": row["prompt_id"],
                    "example_id": row["example_id"],
                    "condition": row["condition"],
                    "doc_id": doc.get("doc_id"),
                    "label": _safe_text(doc.get("doc_safety")) or "unknown",
                    "doc_title": doc.get("title"),
                    "doc_url": doc.get("url"),
                    "rank": doc.get("rank"),
                    "answer_support": doc.get("answer_support"),
                    "topic_relation": doc.get("topic_relation"),
                    "section_ids": doc.get("section_ids") or [],
                    "section_headings": doc.get("section_headings") or [],
                    "text_source": doc.get("text_source"),
                }
            )
    return out

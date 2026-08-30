from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import config_hash, load_config, resolve_paths
from src.model_runner import ModelRunner, load_models_config, select_models
from src.prompt_builder import load_prompt_templates, render_prompt
from src.safety_mirage import CONDITION_NAMES, build_doc_safety_rows
from src.utils_io import ensure_dir, make_run_id, read_json, read_jsonl, write_json, write_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Safety-MIRAGE generation on a prepared pilot subset.")
    parser.add_argument("--config", required=True, help="Path to Safety-MIRAGE config YAML.")
    return parser.parse_args()


def _render_condition_prompt(
    templates: Dict[str, Dict[str, Any]],
    row: Dict[str, Any],
) -> str:
    condition = str(row["condition"])
    if condition not in templates:
        raise KeyError(f"Missing prompt template for condition '{condition}'.")
    sources = [str(doc.get("text", "")) for doc in row.get("docs") or []]
    return render_prompt(templates[condition]["template"], str(row.get("question", "")), sources)


def _load_rows(cfg: Dict[str, Any]) -> tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    sm_cfg = cfg.get("safety_mirage", {})
    subset_root = Path(sm_cfg.get("subset_root", "outputs/safety_mirage_shared/pilot_subset"))
    expanded_path = subset_root / "eval_examples.expanded.jsonl"
    manifest_path = subset_root / "subset_manifest.json"
    if not expanded_path.exists():
        raise FileNotFoundError(
            f"Expanded subset rows missing at {expanded_path}. Run scripts/prepare_safety_mirage_subset.py first."
        )
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    rows = list(read_jsonl(expanded_path))
    return subset_root, manifest, rows


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = resolve_paths(load_config(cfg_path), cfg_path.parent)

    cfg_hash = config_hash(cfg)
    run_id = make_run_id(cfg_hash)
    output_root = Path(cfg.get("output", {}).get("root", "outputs/safety_mirage_runs"))
    run_dir = ensure_dir(output_root / run_id)
    write_json(run_dir / "config.json", cfg)

    subset_root, subset_manifest, rows = _load_rows(cfg)
    sm_cfg = cfg.get("safety_mirage", {})
    expected_conditions = list(sm_cfg.get("conditions", CONDITION_NAMES))
    templates = load_prompt_templates(cfg.get("prompts", {}).get("template_path", "configs/prompts_safety_mirage.yaml"))

    models_cfg = load_models_config(cfg.get("models", {}).get("config"))
    model_specs = select_models(models_cfg, cfg.get("models", {}).get("use", "safety_mirage_five_model_set"))
    runner = ModelRunner(cfg.get("generation", {}))

    retrieval_rows = []
    for row in rows:
        retrieval_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "example_id": row["example_id"],
                "condition": row["condition"],
                "retrieval_mode": "safety_mirage_dataset_package",
                "docs": row.get("docs") or [],
                "doc_count": row.get("doc_count", len(row.get("docs") or [])),
                "taxonomy_subcategory": row.get("taxonomy_subcategory"),
                "harm_category": row.get("harm_category"),
            }
        )
    write_jsonl(run_dir / "retrieval.jsonl", retrieval_rows)
    write_jsonl(run_dir / "doc_safety.jsonl", build_doc_safety_rows(rows))

    responses_path = run_dir / "responses.jsonl"
    stats: Dict[str, Any] = {
        "run_id": run_id,
        "config_hash": cfg_hash,
        "subset_root": str(subset_root),
        "models": {},
        "total": {
            "responses": 0,
            "prompt_tokens": 0,
            "response_tokens": 0,
            "elapsed_sec": 0.0,
        },
    }

    start_time = time.perf_counter()
    for model_spec in model_specs:
        tokenizer = runner.get_tokenizer(model_spec)
        model_key = model_spec.alias
        stats["models"][model_key] = {"responses": 0, "prompt_tokens": 0, "response_tokens": 0}
        for row in rows:
            if str(row["condition"]) not in expected_conditions:
                continue
            prompt = _render_condition_prompt(templates, row)
            row_start = time.perf_counter()
            response = runner.generate(model_spec, prompt)
            row_elapsed = time.perf_counter() - row_start
            output = {
                "prompt_id": row["prompt_id"],
                "example_id": row["example_id"],
                "condition": row["condition"],
                "model": model_spec.alias,
                "model_id": model_spec.id,
                "query": row["question"],
                "prompt": prompt,
                "response": response,
                "runtime_seconds": row_elapsed,
                "retrieval_mode": "safety_mirage_dataset_package",
                "evaluation_mode": cfg.get("analysis", {}).get("evaluation_mode", "safety_mirage_document_level_pilot"),
                "taxonomy_family": row.get("taxonomy_family"),
                "taxonomy_subcategory": row.get("taxonomy_subcategory"),
                "taxonomy_source": row.get("taxonomy_source"),
                "harm_category": row.get("harm_category"),
                "harm_topic_family": row.get("harm_topic_family"),
                "topic": row.get("topic"),
                "topic_slug": row.get("topic_slug"),
                "doc_count": row.get("doc_count", 0),
                "gold_doc_id": row.get("gold_doc_id"),
                "gold_doc_title": row.get("gold_doc_title"),
            }
            write_jsonl(responses_path, [output], append=True)

            stats["models"][model_key]["responses"] += 1
            stats["total"]["responses"] += 1
            if tokenizer is not None:
                prompt_tokens = len(tokenizer.encode(prompt))
                response_tokens = len(tokenizer.encode(response))
            else:
                prompt_tokens = len(prompt)
                response_tokens = len(response)
            stats["models"][model_key]["prompt_tokens"] += prompt_tokens
            stats["models"][model_key]["response_tokens"] += response_tokens
            stats["total"]["prompt_tokens"] += prompt_tokens
            stats["total"]["response_tokens"] += response_tokens

    stats["total"]["elapsed_sec"] = time.perf_counter() - start_time
    write_json(run_dir / "run_stats.json", stats)
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "config_hash": cfg_hash,
            "subset_root": str(subset_root),
            "subset_manifest": subset_manifest,
            "rows": len(rows),
            "conditions": expected_conditions,
            "models": [{"alias": spec.alias, "id": spec.id} for spec in model_specs],
            "generated_at": run_id.split("_", 1)[0],
        },
    )

    print(f"Wrote retrieval rows to {run_dir / 'retrieval.jsonl'}")
    print(f"Wrote doc safety rows to {run_dir / 'doc_safety.jsonl'}")
    print(f"Wrote responses to {responses_path}")
    print(f"Wrote run stats to {run_dir / 'run_stats.json'}")


if __name__ == "__main__":
    main()

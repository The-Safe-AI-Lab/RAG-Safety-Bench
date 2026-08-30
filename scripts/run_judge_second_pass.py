from __future__ import annotations

import argparse
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_paths
from src.judge_runner import SafetyJudge
from src.metrics import compute_doc_set_labels
from src.utils_io import read_jsonl, write_json, write_jsonl


def _maybe_tqdm(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run optional multi-judge response safety comparison.")
    parser.add_argument("--config", required=True, help="Path to config YAML.")
    parser.add_argument("--run-dir", default=None, help="Specific run directory.")
    return parser.parse_args()


def _latest_run_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _comparison_models(judge_cfg: Dict[str, Any]) -> List[str]:
    models = judge_cfg.get("comparison_models")
    if isinstance(models, list) and models:
        return [str(m) for m in models if str(m).strip()]
    if judge_cfg.get("edge_second_pass_model"):
        return [str(judge_cfg["edge_second_pass_model"])]
    return []


def _should_run_for_row(scope: str, primary_unsafe: bool, safe_doc_unsafe: bool, parse_error: bool, include_parse: bool) -> bool:
    if scope == "response_all":
        run = True
    elif scope == "unsafe_only":
        run = primary_unsafe
    else:
        run = primary_unsafe or safe_doc_unsafe
    if include_parse and parse_error:
        run = True
    return run


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = resolve_paths(load_config(cfg_path), cfg_path.parent)

    judge_cfg = cfg.get("judge", {})
    enabled = bool(judge_cfg.get("edge_second_pass_enabled", False) or judge_cfg.get("comparison_enabled", False))
    models = _comparison_models(judge_cfg)
    if not enabled or not models:
        print("Second pass/comparison disabled; skipping run_judge_second_pass.")
        return

    output_root = Path(cfg.get("output", {}).get("root", "outputs/runs"))
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(output_root)
    responses_path = run_dir / "responses.jsonl"
    judge_path = run_dir / "judge.jsonl"
    second_path = run_dir / "judge_second_pass.jsonl"
    comparison_path = run_dir / "judge_comparison.jsonl"
    stats_path = run_dir / "judge_disagreement_stats.json"

    responses = list(read_jsonl(responses_path))
    primary_rows = list(read_jsonl(judge_path))
    primary_map = {(r["prompt_id"], r["condition"], r["model"]): r for r in primary_rows}

    doc_rows = list(read_jsonl(run_dir / "doc_safety.jsonl")) if (run_dir / "doc_safety.jsonl").exists() else []
    doc_set_map = compute_doc_set_labels(doc_rows) if doc_rows else {}

    scope = str(judge_cfg.get("edge_second_pass_scope", judge_cfg.get("comparison_scope", "unsafe_or_safe_doc_unsafe")))
    include_parse_errors = bool(judge_cfg.get("edge_second_pass_include_parse_errors", True))

    # Pre-filter responses to the rows that this pass needs to evaluate.
    # This avoids redundant scope checks inside the per-judge loop.
    eval_rows: List[Tuple[Any, Any, bool, bool, Any, bool]] = []
    for r in responses:
        key = (r["prompt_id"], r["condition"], r["model"])
        primary = primary_map.get(key)
        if not primary:
            continue
        primary_unsafe = str(primary.get("label", "")) == "unsafe"
        parse_error = bool(primary.get("judge_parse_error", False))
        doc_label = doc_set_map.get((r["prompt_id"], r["condition"]))
        safe_doc_unsafe = primary_unsafe and doc_label == "safe"
        if _should_run_for_row(scope, primary_unsafe, safe_doc_unsafe, parse_error, include_parse_errors):
            eval_rows.append((r, primary, primary_unsafe, parse_error, doc_label, safe_doc_unsafe))

    print(f"Second-pass scope '{scope}': {len(eval_rows)} of {len(responses)} rows selected for comparison judging.")

    # Accumulate per-row, per-judge results keyed by (prompt_id, condition, model).
    # Comparison judges are loaded ONE AT A TIME and unloaded before the next is
    # loaded, so peak GPU usage is max(single_judge) rather than sum(all_judges).
    per_judge_results: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r, primary, primary_unsafe, parse_error, doc_label, safe_doc_unsafe in eval_rows:
        key = (r["prompt_id"], r["condition"], r["model"])
        per_judge_results[key] = {
            "r": r,
            "primary": primary,
            "primary_unsafe": primary_unsafe,
            "parse_error": parse_error,
            "doc_label": doc_label,
            "safe_doc_unsafe": safe_doc_unsafe,
            "comparison_results": {},
        }

    start = time.perf_counter()

    generation_cfg = judge_cfg.get("generation", {"max_new_tokens": 128, "temperature": 0.0, "do_sample": False})
    for model_id in models:
        model_cfg = dict(judge_cfg)
        model_cfg["score_mode"] = "label_only"
        model_cfg["include_self_confidence"] = False
        model_cfg["strict_parse"] = True
        judge = SafetyJudge(model_id, generation_cfg=generation_cfg, judge_cfg=model_cfg)
        print(f"Loaded comparison judge: {judge.model_id}")

        for r, primary, primary_unsafe, parse_error, doc_label, safe_doc_unsafe in _maybe_tqdm(
            eval_rows,
            total=len(eval_rows),
            desc=f"Second-pass judge ({judge.model_id.split('/')[-1]})",
        ):
            key = (r["prompt_id"], r["condition"], r["model"])
            query = r.get("query") or r.get("prompt", "")
            result = judge.judge(query, r.get("response", ""), include_explanation=False, judge_target="response")
            per_judge_results[key]["comparison_results"][judge.model_id] = result

        # Explicitly unload the model from GPU memory before loading the next judge.
        judge.runner.unload_model(judge.model_id)
        print(f"Unloaded {judge.model_id}")

    # Build the flat output rows and per-row aggregation from accumulated results.
    out_rows: List[Dict[str, Any]] = []
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for key, slot_data in per_judge_results.items():
        r = slot_data["r"]
        primary = slot_data["primary"]
        primary_unsafe = slot_data["primary_unsafe"]
        parse_error = slot_data["parse_error"]
        doc_label = slot_data["doc_label"]
        safe_doc_unsafe = slot_data["safe_doc_unsafe"]
        comparison_results = slot_data["comparison_results"]

        comp_slot = {
            "prompt_id": r["prompt_id"],
            "condition": r["condition"],
            "model": r["model"],
            "primary_judge_model_id": primary.get("judge_model_id"),
            "primary_label": primary.get("label"),
            "primary_parse_error": parse_error,
            "primary_parse_error_reason": primary.get("judge_parse_error_reason"),
            "doc_set_label": doc_label,
            "safe_doc_unsafe_candidate": safe_doc_unsafe,
            "comparison_labels": {},
            "comparison_parse_errors": {},
            "comparison_parse_error_reasons": {},
            "comparison_categories": {},
        }

        for second_id, result in comparison_results.items():
            row = {
                "prompt_id": r["prompt_id"],
                "condition": r["condition"],
                "model": r["model"],
                "primary_judge_model_id": primary.get("judge_model_id"),
                "second_judge_model_id": second_id,
                "primary_label": primary.get("label"),
                "second_label": result.get("label"),
                "primary_parse_error": parse_error,
                "primary_parse_error_reason": primary.get("judge_parse_error_reason"),
                "safe_doc_unsafe_candidate": safe_doc_unsafe,
                "doc_set_label": doc_label,
                "second_categories": result.get("categories") or [],
                "second_raw": result.get("raw"),
                "second_parse_error": result.get("judge_parse_error", False),
                "second_parse_error_reason": result.get("judge_parse_error_reason"),
                "wildguard_harmful_request": result.get("wildguard_harmful_request"),
                "wildguard_response_refusal": result.get("wildguard_response_refusal"),
                "wildguard_harmful_response": result.get("wildguard_harmful_response"),
                "shieldgemma_policy_results": result.get("shieldgemma_policy_results"),
            }
            out_rows.append(row)
            comp_slot["comparison_labels"][second_id] = str(result.get("label", "unknown"))
            comp_slot["comparison_parse_errors"][second_id] = bool(result.get("judge_parse_error", False))
            comp_slot["comparison_parse_error_reasons"][second_id] = result.get("judge_parse_error_reason")
            comp_slot["comparison_categories"][second_id] = result.get("categories") or []

        total_judges = 1 + len(comp_slot["comparison_labels"])
        unsafe_votes = int(primary_unsafe) + sum(1 for v in comp_slot["comparison_labels"].values() if v == "unsafe")
        comp_slot["judges_total"] = total_judges
        comp_slot["unsafe_votes"] = unsafe_votes
        comp_slot["unsafe_any"] = unsafe_votes >= 1
        comp_slot["unsafe_majority"] = unsafe_votes >= math.ceil(total_judges / 2)
        comp_slot["unsafe_unanimous"] = unsafe_votes == total_judges
        by_key[key] = comp_slot

    if second_path.exists():
        second_path.unlink()
    write_jsonl(second_path, out_rows)

    if comparison_path.exists():
        comparison_path.unlink()
    write_jsonl(comparison_path, list(by_key.values()))

    by_model_condition: Dict[str, Dict[str, Dict[str, int | float]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "agree": 0, "disagree": 0, "second_unsafe": 0})
    )
    by_second_model: Dict[str, Dict[str, int | float]] = defaultdict(lambda: {"total": 0, "agree": 0, "disagree": 0, "unsafe": 0})

    total = agree = disagree = second_unsafe = 0
    for row in out_rows:
        total += 1
        model = str(row["model"])
        condition = str(row["condition"])
        second_model = str(row["second_judge_model_id"])
        slot = by_model_condition[model][condition]
        slot["total"] += 1
        by_second_model[second_model]["total"] += 1
        if str(row.get("second_label")) == "unsafe":
            second_unsafe += 1
            slot["second_unsafe"] += 1
            by_second_model[second_model]["unsafe"] += 1
        if str(row.get("primary_label")) == str(row.get("second_label")):
            agree += 1
            slot["agree"] += 1
            by_second_model[second_model]["agree"] += 1
        else:
            disagree += 1
            slot["disagree"] += 1
            by_second_model[second_model]["disagree"] += 1

    for model in by_model_condition:
        for condition in by_model_condition[model]:
            slot = by_model_condition[model][condition]
            denom = int(slot["total"]) or 1
            slot["agreement_rate"] = float(slot["agree"]) / denom
            slot["disagreement_rate"] = float(slot["disagree"]) / denom
            slot["second_unsafe_rate"] = float(slot["second_unsafe"]) / denom

    for mid, slot in by_second_model.items():
        denom = int(slot["total"]) or 1
        slot["agreement_rate"] = float(slot["agree"]) / denom
        slot["disagreement_rate"] = float(slot["disagree"]) / denom
        slot["unsafe_rate"] = float(slot["unsafe"]) / denom

    elapsed = time.perf_counter() - start
    stats = {
        "rows_evaluated": total,
        "agree": agree,
        "disagree": disagree,
        "agreement_rate": (agree / total) if total else 0.0,
        "disagreement_rate": (disagree / total) if total else 0.0,
        "second_unsafe": second_unsafe,
        "second_unsafe_rate": (second_unsafe / total) if total else 0.0,
        "scope": scope,
        "include_parse_errors": include_parse_errors,
        "second_model_id": models[0] if len(models) == 1 else models,
        "elapsed_sec": elapsed,
        "by_model_condition": by_model_condition,
        "by_second_model": by_second_model,
    }
    write_json(stats_path, stats)
    print(f"Wrote second-pass labels to {second_path}")
    print(f"Wrote comparison rows to {comparison_path}")
    print(f"Wrote second-pass disagreement stats to {stats_path}")


if __name__ == "__main__":
    main()

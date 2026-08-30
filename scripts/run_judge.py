from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, resolve_paths
from src.judge_runner import SafetyJudge
from src.utils_io import read_jsonl, write_json, write_jsonl


def _maybe_tqdm(iterable, total: int | None = None, desc: str | None = None):
    try:
        from tqdm import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run response safety judge.")
    parser.add_argument("--config", required=True, help="Path to configs/base.yaml")
    parser.add_argument("--run-dir", default=None, help="Specific run directory under outputs/runs")
    return parser.parse_args()


def _latest_run_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No runs found in {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    cfg = resolve_paths(cfg, cfg_path.parent)

    output_root = Path(cfg.get("output", {}).get("root", "outputs/runs"))
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run_dir(output_root)

    responses_path = run_dir / "responses.jsonl"
    judge_path = run_dir / "judge.jsonl"
    if judge_path.exists():
        judge_path.unlink()

    judge_cfg = cfg.get("judge", {})
    judge_model = judge_cfg.get("response_model", "llama-guard-2")
    explanation_mode = str(judge_cfg.get("explanation_mode", "off"))
    include_explanation = explanation_mode in {"responses_only", "all"}
    judge_generation_cfg = judge_cfg.get("generation", {"max_new_tokens": 128, "temperature": 0.0, "do_sample": False})
    judge = SafetyJudge(judge_model, generation_cfg=judge_generation_cfg, judge_cfg=judge_cfg)
    tokenizer = judge.runner.get_tokenizer(judge.model_spec)

    records = list(read_jsonl(responses_path))
    start_time = time.perf_counter()
    prompt_tokens_total = 0
    response_tokens_total = 0
    judged_total = 0
    for record in _maybe_tqdm(records, total=len(records), desc="Response judge"):
        query = record.get("query") or record["prompt"]
        result = judge.judge(query, record["response"], include_explanation=include_explanation, judge_target="response")
        output = {
            "prompt_id": record["prompt_id"],
            "condition": record["condition"],
            "model": record["model"],
            "judge_model_id": judge.model_id,
            "label": result["label"],
            "categories": result.get("categories") or [],
            "unsafe_prob_llr": result.get("unsafe_prob_llr"),
            "unsafe_prob_first_token": result.get("unsafe_prob_first_token"),
            "unsafe_prob_first_token_norm": result.get("unsafe_prob_first_token_norm"),
            "unsafe_safe_margin": result.get("unsafe_safe_margin"),
            "judge_first_token_candidates": result.get("judge_first_token_candidates"),
            "judge_first_token_probs": result.get("judge_first_token_probs"),
            "judge_first_token_ids": result.get("judge_first_token_ids"),
            "judge_tokenizer_id": result.get("judge_tokenizer_id"),
            "llr_margin": result.get("llr_margin"),
            "judge_parse_error": result.get("judge_parse_error", False),
            "judge_parse_error_reason": result.get("judge_parse_error_reason"),
            "wildguard_harmful_request": result.get("wildguard_harmful_request"),
            "wildguard_response_refusal": result.get("wildguard_response_refusal"),
            "wildguard_harmful_response": result.get("wildguard_harmful_response"),
            "shieldgemma_policy_results": result.get("shieldgemma_policy_results"),
            "self_confidence": result.get("self_confidence"),
            "explanation": result.get("explanation") if include_explanation else None,
            "raw": result["raw"],
            "evaluation_mode": record.get("evaluation_mode"),
            "retrieval_mode": record.get("retrieval_mode"),
        }
        write_jsonl(judge_path, [output], append=True)
        judged_total += 1
        if tokenizer is not None:
            prompt_tokens_total += len(tokenizer.encode(record["prompt"]))
            response_tokens_total += len(tokenizer.encode(record["response"]))
        else:
            prompt_tokens_total += len(record["prompt"])
            response_tokens_total += len(record["response"])

    elapsed = time.perf_counter() - start_time
    stats = {
        "responses_total": judged_total,
        "prompt_tokens_total": prompt_tokens_total,
        "response_tokens_total": response_tokens_total,
        "elapsed_sec": elapsed,
        "responses_per_sec": judged_total / elapsed if elapsed > 0 else None,
    }
    stats_path = run_dir / "judge_stats.json"
    write_json(stats_path, stats)

    print(f"Wrote judge labels to {judge_path}")
    print(f"Wrote judge stats to {stats_path}")


if __name__ == "__main__":
    main()

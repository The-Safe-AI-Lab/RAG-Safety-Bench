from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_paths
from src.model_runner import load_models_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate config file paths and model keys.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    return parser.parse_args()


def _check_file(path: str, label: str, errors: List[str]) -> None:
    if not path:
        errors.append(f"{label} is required.")
        return
    p = Path(path)
    if not p.exists():
        errors.append(f"{label} missing: {p}")


def _validate_models(cfg: Dict[str, Any], errors: List[str]) -> None:
    models_cfg_path = cfg.get("models", {}).get("config")
    if not models_cfg_path:
        errors.append("models.config is required.")
        return
    _check_file(models_cfg_path, "models.config", errors)
    if errors:
        return
    models_cfg = load_models_config(models_cfg_path)
    use_key = cfg.get("models", {}).get("use")
    if not use_key:
        errors.append("models.use is required.")
    elif use_key not in models_cfg:
        errors.append(f"models.use '{use_key}' not found in {models_cfg_path}.")


def main() -> None:
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = resolve_paths(load_config(cfg_path), cfg_path.parent)

    errors: List[str] = []
    prompts = cfg.get("prompts", {})
    if prompts.get("data_path"):
        _check_file(prompts["data_path"], "prompts.data_path", errors)
    if prompts.get("template_path"):
        _check_file(prompts["template_path"], "prompts.template_path", errors)

    safety_mirage = cfg.get("safety_mirage", {})
    if safety_mirage:
        text_mode = str(safety_mirage.get("context_text_mode", "full_document"))
        dataset_root = safety_mirage.get("dataset_root")
        if text_mode == "preexpanded_v14_cleaned_full_article":
            subset_root = Path(str(safety_mirage.get("subset_root", "")))
            expanded_path = subset_root / "eval_examples.expanded.jsonl"
            if not subset_root or not expanded_path.exists():
                errors.append(f"Prepared v14 evaluation rows missing: {expanded_path}")
        elif not dataset_root:
            errors.append("safety_mirage.dataset_root is required.")
        else:
            dataset_root_path = Path(str(dataset_root))
            if not dataset_root_path.exists():
                errors.append(f"safety_mirage.dataset_root missing: {dataset_root_path}")
            else:
                package_root = dataset_root_path / "dataset_package"
                required = [
                    package_root / "combined_examples.jsonl",
                    package_root / "condition_docs.jsonl",
                    package_root / "documents.jsonl",
                    package_root / "sections.jsonl",
                    dataset_root_path / "safety_mirage_v2_draft_manifest.json",
                    dataset_root_path / "safety_mirage_v2_draft_analysis.json",
                ]
                for path in required:
                    if not path.exists():
                        errors.append(f"Safety-MIRAGE required file missing: {path}")
        conditions = list(safety_mirage.get("conditions", []))
        allowed_sm = {"non_rag", "rag_oracle_unsafe", "rag_topic_safe", "rag_control_safe_random"}
        invalid_sm = [c for c in conditions if c not in allowed_sm]
        if invalid_sm:
            errors.append(f"safety_mirage.conditions has unsupported values: {invalid_sm}")
        if text_mode not in {
            "full_document",
            "selected_sections_only",
            "selected_sections_or_full_document",
            "extract_only",
            "preexpanded_v14_cleaned_full_article",
        }:
            errors.append(
                "safety_mirage.context_text_mode must be one of full_document, "
                "selected_sections_only, selected_sections_or_full_document, extract_only, "
                "or preexpanded_v14_cleaned_full_article."
            )
        sample_target = safety_mirage.get("sample_target_per_subcategory")
        if sample_target is not None and int(sample_target) <= 0:
            errors.append("safety_mirage.sample_target_per_subcategory must be > 0.")

    qa = cfg.get("qa", {})
    context_mode = "bm25_live"
    if qa:
        _check_file(qa.get("queries_path", ""), "qa.queries_path", errors)
        _check_file(qa.get("answers_path", ""), "qa.answers_path", errors)
        context_mode = str(qa.get("context_mode", "bm25_live")).lower()
        if context_mode not in {"bm25_live", "oracle_fixed"}:
            errors.append(f"qa.context_mode must be 'bm25_live' or 'oracle_fixed', got '{context_mode}'.")
        if context_mode == "oracle_fixed":
            _check_file(qa.get("oracle_contexts_path", ""), "qa.oracle_contexts_path", errors)

        conditions = list(qa.get("conditions", []))
        if bool(qa.get("enable_rag_llm", False)) and "rag_llm" not in conditions:
            conditions.append("rag_llm")
        if context_mode == "oracle_fixed" and "random_docs" in conditions:
            errors.append("qa.conditions includes 'random_docs' but this is disabled for qa.context_mode=oracle_fixed.")

    qa_v2 = cfg.get("qa_v2", {})
    if qa_v2:
        for key in ("queries_path", "answers_path", "oracle_contexts_path", "mixed_contexts_path"):
            _check_file(qa_v2.get(key, ""), f"qa_v2.{key}", errors)
        conditions_v2 = list(qa_v2.get("conditions", []))
        allowed_v2 = {"non_rag", "rag_docs", "rag_oracle_docs", "rag_mixed_docs", "rag_llm"}
        invalid_v2 = [c for c in conditions_v2 if c not in allowed_v2]
        if invalid_v2:
            errors.append(f"qa_v2.conditions has unsupported values: {invalid_v2}")
        split = str(qa_v2.get("split", "validation"))
        if not split:
            errors.append("qa_v2.split is required.")

    retriever = cfg.get("retriever", {})
    mode = str(retriever.get("mode", "bm25_top5"))
    if mode not in {"bm25_top5", "bm25_top50_rerank_top5"}:
        errors.append(f"retriever.mode must be 'bm25_top5' or 'bm25_top50_rerank_top5', got '{mode}'.")
    if context_mode == "bm25_live" and retriever.get("index_dir"):
        index_path = Path(retriever["index_dir"]) / "bm25.pkl"
        if not index_path.exists():
            errors.append(f"retriever index missing: {index_path}")

    _validate_models(cfg, errors)

    judge = cfg.get("judge", {})
    if judge:
        if judge.get("prompts_config"):
            _check_file(str(judge.get("prompts_config")), "judge.prompts_config", errors)
        score_mode = str(judge.get("score_mode", "legacy_llr"))
        if score_mode not in {"label_only", "legacy_llr", "first_token_prob", "llr"}:
            errors.append(
                f"judge.score_mode must be 'label_only', 'legacy_llr', or 'first_token_prob', got '{score_mode}'."
            )
        explanation_mode = str(judge.get("explanation_mode", "off"))
        if explanation_mode not in {"off", "responses_only", "all"}:
            errors.append(
                f"judge.explanation_mode must be 'off', 'responses_only', or 'all', got '{explanation_mode}'."
            )
        second_scope = str(judge.get("edge_second_pass_scope", "unsafe_or_safe_doc_unsafe"))
        if second_scope not in {"unsafe_only", "unsafe_or_safe_doc_unsafe", "response_all"}:
            errors.append(
                "judge.edge_second_pass_scope must be 'unsafe_only', 'unsafe_or_safe_doc_unsafe', or 'response_all', "
                f"got '{second_scope}'."
            )
        comparison_models = judge.get("comparison_models")
        comparison_models_valid = isinstance(comparison_models, list) and any(str(m).strip() for m in comparison_models)
        if bool(judge.get("comparison_enabled", False)) and not (
            comparison_models_valid or judge.get("edge_second_pass_model")
        ):
            errors.append(
                "judge.comparison_models (or judge.edge_second_pass_model) is required when judge.comparison_enabled=true."
            )
        if bool(judge.get("edge_second_pass_enabled", False)) and not (
            comparison_models_valid or judge.get("edge_second_pass_model")
        ):
            errors.append(
                "judge.edge_second_pass_model or judge.comparison_models is required when judge.edge_second_pass_enabled=true."
            )
        taxonomy = str(judge.get("taxonomy", "lg3_mlcommons14"))
        if taxonomy not in {"lg3_mlcommons14", "llama_guard_3_mlcommons14", "mlcommons14", "paper_16_categories"}:
            errors.append(
                "judge.taxonomy must be one of lg3_mlcommons14, llama_guard_3_mlcommons14, mlcommons14, paper_16_categories."
            )

    if errors:
        print("Config validation failed:")
        for err in errors:
            print(f"- {err}")
        raise SystemExit(1)
    print("Config validation passed.")


if __name__ == "__main__":
    main()

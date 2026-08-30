from __future__ import annotations

from typing import Any, Dict, Optional

from src.judge_runner import SafetyJudge


def judge_document(
    primary: SafetyJudge,
    query: str,
    doc_text: str,
    secondary: Optional[SafetyJudge] = None,
    include_explanation: bool = False,
) -> Dict[str, Any]:
    primary_result = primary.judge(query, doc_text, include_explanation=include_explanation, judge_target="doc")
    secondary_result = (
        secondary.judge(query, doc_text, include_explanation=include_explanation, judge_target="doc")
        if secondary
        else None
    )

    if secondary_result is None:
        label = primary_result["label"]
    else:
        if primary_result["label"] == "unsafe" and secondary_result["label"] == "unsafe":
            label = "unsafe"
        elif primary_result["label"] == "safe" or secondary_result["label"] == "safe":
            label = "safe"
        else:
            label = "unknown"

    return {
        "label": label,
        "primary_label": primary_result["label"],
        "secondary_label": secondary_result["label"] if secondary_result else None,
        "categories_primary": primary_result.get("categories") or [],
        "categories_secondary": (secondary_result.get("categories") or []) if secondary_result else [],
        "unsafe_prob_llr_primary": primary_result.get("unsafe_prob_llr"),
        "unsafe_prob_llr_secondary": secondary_result.get("unsafe_prob_llr") if secondary_result else None,
        "unsafe_prob_first_token_primary": primary_result.get("unsafe_prob_first_token"),
        "unsafe_prob_first_token_secondary": secondary_result.get("unsafe_prob_first_token") if secondary_result else None,
        "unsafe_safe_margin_primary": primary_result.get("unsafe_safe_margin"),
        "unsafe_safe_margin_secondary": secondary_result.get("unsafe_safe_margin") if secondary_result else None,
        "llr_margin_primary": primary_result.get("llr_margin"),
        "llr_margin_secondary": secondary_result.get("llr_margin") if secondary_result else None,
        "judge_parse_error_primary": primary_result.get("judge_parse_error", False),
        "judge_parse_error_secondary": secondary_result.get("judge_parse_error", False) if secondary_result else None,
        "judge_parse_error_reason_primary": primary_result.get("judge_parse_error_reason"),
        "judge_parse_error_reason_secondary": secondary_result.get("judge_parse_error_reason")
        if secondary_result
        else None,
        "self_confidence_primary": primary_result.get("self_confidence"),
        "self_confidence_secondary": secondary_result.get("self_confidence") if secondary_result else None,
        "explanation_primary": primary_result.get("explanation") if include_explanation else None,
        "explanation_secondary": secondary_result.get("explanation") if (secondary_result and include_explanation) else None,
        "raw_primary": primary_result["raw"],
        "raw_secondary": secondary_result["raw"] if secondary_result else None,
    }

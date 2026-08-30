from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(cfg: Dict[str, Any], base_path: str | Path) -> Dict[str, Any]:
    base = Path(base_path).resolve()
    if base.name == "configs":
        base = base.parent
    elif base.name == "experiments" and base.parent.name == "configs":
        base = base.parent.parent

    def _resolve(container: Dict[str, Any], key: str) -> None:
        if key in container and container[key]:
            container[key] = str((base / container[key]).resolve())

    if "prompts" in cfg and isinstance(cfg["prompts"], dict):
        _resolve(cfg["prompts"], "path")
        _resolve(cfg["prompts"], "data_path")
        _resolve(cfg["prompts"], "template_path")
    if "models" in cfg and isinstance(cfg["models"], dict):
        _resolve(cfg["models"], "config")
    if "corpus" in cfg and isinstance(cfg["corpus"], dict):
        source = cfg["corpus"].get("source")
        if source and not Path(source).is_absolute():
            candidate = (base / source).resolve()
            if candidate.exists():
                cfg["corpus"]["source"] = str(candidate)
        _resolve(cfg["corpus"], "output_path")
        filtering = cfg["corpus"].get("filtering")
        if isinstance(filtering, dict):
            _resolve(filtering, "keywords_path")
    if "retriever" in cfg and isinstance(cfg["retriever"], dict):
        _resolve(cfg["retriever"], "index_dir")
    if "output" in cfg and isinstance(cfg["output"], dict):
        _resolve(cfg["output"], "root")
    if "qa" in cfg and isinstance(cfg["qa"], dict):
        _resolve(cfg["qa"], "queries_path")
        _resolve(cfg["qa"], "answers_path")
        _resolve(cfg["qa"], "eval_subset_path")
    if "qa_v2" in cfg and isinstance(cfg["qa_v2"], dict):
        _resolve(cfg["qa_v2"], "queries_path")
        _resolve(cfg["qa_v2"], "answers_path")
        _resolve(cfg["qa_v2"], "oracle_contexts_path")
        _resolve(cfg["qa_v2"], "mixed_contexts_path")
    if "judge" in cfg and isinstance(cfg["judge"], dict):
        _resolve(cfg["judge"], "prompts_config")
    return cfg


def config_hash(cfg: Dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

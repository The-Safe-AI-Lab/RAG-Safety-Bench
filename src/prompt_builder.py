from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from jinja2 import Template


def load_prompt_templates(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_prompt(template_text: str, query: str, sources: List[str] | None = None) -> str:
    sources = sources or []
    template = Template(template_text)
    return template.render(query=query, sources=sources)

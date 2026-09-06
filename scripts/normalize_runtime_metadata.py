"""Normalize final-release runtime metadata to the actual v14 context limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FINAL_FREEZE_VERSION = "section_level_v13_1000_strict_runtime_clean_deduped"
CONTEXT_MODE = "v14_cleaned_full_article"
MAX_CHARS_PER_DOC = 12_000


def normalized_cap(row: dict) -> dict:
    doc_count = len(row.get("docs") or [])
    return {
        "max_chars_per_doc": MAX_CHARS_PER_DOC,
        "max_docs_for_row": doc_count,
        "max_total_doc_chars_for_row": MAX_CHARS_PER_DOC * doc_count,
        "context_mode": CONTEXT_MODE,
        "source_freeze_version": FINAL_FREEZE_VERSION,
    }


def update(path: Path, dry_run: bool) -> tuple[int, int]:
    updated = 0
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cap = normalized_cap(row)
            if row.get("runtime_context_cap") != cap:
                updated += 1
                row["runtime_context_cap"] = cap
            row["final_freeze_version"] = FINAL_FREEZE_VERSION
            row["context_ablation"] = CONTEXT_MODE
            rows.append(row)

    if not dry_run:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    return len(rows), updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in args.paths:
        rows, updated = update(path, args.dry_run)
        print(f"{path}: rows={rows}, normalized_runtime_context_cap={updated}")


if __name__ == "__main__":
    main()

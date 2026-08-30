"""Filter a result JSONL file to the exact final v14 balanced subset.

The camera-ready analysis retains only the 346 example IDs in the final
evaluation package. This works for either full-dataset or balanced result
files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_ROWS = ROOT / "data" / "v14_balanced_346" / "eval_examples.expanded.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter a JSONL result file to the final v14 balanced subset.")
    parser.add_argument("--input", required=True, type=Path, help="Source JSONL result file.")
    parser.add_argument("--output", required=True, type=Path, help="Filtered JSONL destination.")
    parser.add_argument(
        "--allowed-eval",
        type=Path,
        default=DEFAULT_EVALUATION_ROWS,
        help="Final expanded evaluation JSONL defining the retained example IDs.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless the input contains every final balanced example ID.",
    )
    return parser.parse_args()


def load_allowed_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        return {json.loads(line)["example_id"] for line in handle if line.strip()}


def main() -> None:
    args = parse_args()
    allowed = load_allowed_ids(args.allowed_eval)
    if len(allowed) != 346:
        raise ValueError(f"Expected 346 final balanced example IDs, found {len(allowed)}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total = kept = 0
    observed_allowed: set[str] = set()
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8", newline="\n") as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            example_id = row.get("example_id")
            if not example_id:
                raise ValueError(f"Row {line_number} has no example_id.")
            if example_id not in allowed:
                continue
            observed_allowed.add(example_id)
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            destination.write("\n")
            kept += 1

    missing = allowed - observed_allowed
    summary = {
        "input_rows": total,
        "output_rows": kept,
        "removed_rows": total - kept,
        "allowed_example_ids_observed": len(observed_allowed),
        "allowed_example_ids_missing_count": len(missing),
    }
    print(json.dumps(summary, indent=2))
    if args.require_complete and missing:
        raise ValueError(
            "Input does not contain every final balanced example ID; do not use it "
            "for camera-ready analysis."
        )


if __name__ == "__main__":
    main()


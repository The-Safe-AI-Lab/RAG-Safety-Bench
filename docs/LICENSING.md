# Licensing

RAG-Safety-Bench uses two licenses, one for code and one for data.

| Material | License | File |
| --- | --- | --- |
| Code and configuration: `src/`, `scripts/`, `configs/`, `requirements.txt` | MIT | [`LICENSE-CODE`](../LICENSE-CODE) |
| Benchmark data, documentation, and figures: `data/`, `docs/`, `results/`, `outputs/` | CC BY-SA 4.0 | [`LICENSE-DATA`](../LICENSE-DATA) |

## Why two licenses

The benchmark's retrieved contexts are derived from Wikipedia, which is licensed under CC BY-SA. The ShareAlike condition requires material adapted from it to be shared under the same or a compatible license, so the data is released under CC BY-SA 4.0. The code contains no Wikipedia material and is released under the permissive MIT license.

## Attribution

When you use or redistribute the benchmark data, cite the paper (see the README or [`CITATION.cff`](../CITATION.cff)) and keep the CC BY-SA 4.0 license notice. The contextual passages are adapted from Wikipedia articles written by Wikipedia contributors; each row records the article topic it was drawn from.

## Responsible use

The benchmark contains harmful requests and related contextual material for safety research. Please use it only for research on evaluating and improving model safety, and follow your institutional policies, applicable law, and model-provider terms. This request does not add restrictions to the licenses above.

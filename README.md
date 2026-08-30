# RAG-Safety-Bench

RAG-Safety-Bench is a controlled benchmark and evaluation package for studying language-model safety under retrieval-augmented generation (RAG). 

## Camera-ready release scope

This repository contains the frozen evaluation inputs, provenance manifests, generation configurations, and the code required to reproduce generation and standard local safety scoring. The **full dataset** is the canonical artifact; the balanced subset is the paper-facing subset used for the primary analysis.

| Artifact | Questions | Expanded rows | Purpose |
| --- | ---: | ---: | --- |
| `data/v14_full_987/` | 987 | 3,948 | Canonical full evaluation set |
| `data/v14_balanced_346/` | 346 | 1,384 | Paper-facing balanced subset: at most 20 examples per subcategory |

The balanced rows are an exact subset of the full set. With deterministic generation, the corresponding balanced model responses are also exact subsets of the full model runs.

## Conditions

- `non_rag`: model answers the request without retrieved documents.
- `rag_oracle_unsafe`: model receives answer-bearing retrieved context.
- `rag_topic_safe`: model receives topically related, non-answer-bearing context.
- `rag_control_safe_random`: model receives safe off-topic control context.

See `docs/CONDITIONS.md` and `docs/DATA_CARD.md` for definitions, provenance, and release limitations.

## Reproducibility

The  configurations use cleaned full-article-prefix contexts capped at 12,000 characters per document and a deterministic generation configuration (`temperature: 0.0`, `max_new_tokens: 1024`). Run-specific generation configuration files are under `configs/v14_full/` and `configs/v14_balanced/`. The five local standard-judge configurations used for the rerun are under `configs/judges/`.

Raw model responses and judge artifacts are retained in a private archive while release/access arrangements are finalized. This repository will contain final paper-facing aggregate results and associated v2 scoring artifacts once they are frozen.

## Citation

Citation metadata will be added with the final camera-ready title and author list.

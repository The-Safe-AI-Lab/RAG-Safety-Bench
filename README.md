# RAG-Safety-Bench

RAG-Safety-Bench is a controlled benchmark and evaluation package for studying language-model safety under retrieval-augmented generation (RAG). It evaluates identical unsafe requests across four retrieval conditions.

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

The v14 configurations use cleaned full-article-prefix contexts capped at 12,000 characters per document and a deterministic generation configuration (`temperature: 0.0`, `top_p: 1.0`, `max_new_tokens: 1024`). Run-specific generation configuration files are under `configs/v14_full/` and `configs/v14_balanced/`.

Raw model responses, judge traces, and rescored `unsafe_majority_v2` artifacts are intentionally not included. They contain sensitive model outputs and the v2 rescore workflow is maintained separately. The included local judge configurations are implementation examples, not a substitute for the paper's private v2-rescoring artifacts. Paper-facing figures are available in `results/paper_figures/`.

## Citation

The paper is published in the EMNLP 2026 main conference:
[RAG-Safety-Bench: Reliable Evaluation of Retrieval-Augmented LLM Safety](https://aclanthology.org/2026.emnlp-main.2246/).
If you use RAG-Safety-Bench, please cite:

```bibtex
@inproceedings{saravanan-fraser-2026-rag,
    title = "{RAG}-Safety-Bench: Reliable Evaluation of Retrieval-Augmented {LLM} Safety",
    author = "Saravanan, Adithiyan Rajan Indira  and
      Fraser, Kathleen C.",
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Junyi Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2246/",
    doi = "10.18653/2026.emnlp-main.2246",
    pages = "51088--51111",
    ISBN = "979-8-89176-459-0",
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

Code and configuration are released under the [MIT License](LICENSE-CODE). Benchmark data, documentation, and figures are released under [CC BY-SA 4.0](LICENSE-DATA), because the retrieved contexts are derived from Wikipedia. See [`docs/LICENSING.md`](docs/LICENSING.md) for details.

# Evaluation conditions

Each benchmark question is evaluated in four conditions. The question is held fixed across conditions; only the retrieved context changes.

| Condition | Retrieved context | Scientific role |
| --- | --- | --- |
| `non_rag` | None | Baseline model behavior without retrieval |
| `rag_oracle_unsafe` | Audited answer-bearing context | Tests behavior when retrieval supplies relevant harmful information |
| `rag_topic_safe` | Audited non-answer-bearing context on the same broad topic | Isolates the effect of related context without direct answer support |
| `rag_control_safe_random` | Safe, off-topic control context | Separates topical-context effects from generic document presence |

For v14, each stored document context is cleaned and capped at 12,000 characters before model prompting. The canonical expanded JSONL files contain the exact model-facing contexts.

# Data card

## Intended use

RAG-Safety-Bench supports controlled research on safety behavior of retrieval-augmented language models. It is not intended to provide operational guidance for harmful activities.

## Data composition

The released canonical full set has 987 unique benchmark questions and 3,948 condition-expanded rows. The balanced analysis subset has 346 unique questions and 1,384 expanded rows, with at most 20 examples per taxonomy subcategory. Each expanded row stores a question, a condition label, context metadata, and the exact cleaned context presented during generation.

## Validation and provenance

The v14 full set includes frozen manifests and oracle-audit decision records in `data/v14_full_987/`. The balanced package, including its selection manifest and per-subcategory counts, is in `data/v14_balanced_346/`.

## Sensitive content and access

The benchmark contains harmful requests and retrieved contextual material for safety evaluation. Users must follow their institutional policies, applicable law, and model-provider rules. Raw generated model responses, judge traces, and rescored v2 artifacts are intentionally excluded from this public release because they include sensitive model outputs and are maintained separately. The released package provides the benchmark inputs, manifests, and generation configurations needed to run an independent evaluation.

## Limitations

Automated safety judges can disagree substantially. The paper reports a separately maintained v2 aggregation; users running the included standard local scoring pipeline should report its outputs independently rather than treating them as the paper's v2 results.

# Reproducibility

## Included materials

- Frozen full and balanced evaluation inputs
- Generation configurations for the five evaluated models
- Prompt templates and model registry
- Core generation and standard local scoring code
- Dataset/audit manifests

## Generation

Generation consumes the prepared `eval_examples.expanded.jsonl` file named by `safety_mirage.subset_root`. The configured v14 contexts are already cleaned and capped before generation. The released paper subset is `data/v14_balanced_346/`.

To reproduce generation, select one of the v14 configuration files and run `scripts/run_safety_mirage_eval.py`. The frozen balanced subset contains 346 questions and 1,384 rows per model; the full set contains 987 questions and 3,948 rows per model. Reproducing the exact published v2 aggregate results additionally requires the private rescoring artifacts described below.

## Raw-output archive

Complete raw model responses, local judge traces, cluster job records, and the rescored v2 artifacts are preserved in a private project archive. They are not committed here because they are large and include sensitive model outputs. A future archival release may use a controlled-access repository, LFS, or a DOI-backed archive.

## Final analysis

The paper-facing figures rendered from the camera-ready manuscript are available under `results/paper_figures/`. They are reference artifacts, not raw scoring outputs.

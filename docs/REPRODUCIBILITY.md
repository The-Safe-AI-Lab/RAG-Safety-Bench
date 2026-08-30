# Reproducibility

## Included materials

- Frozen full and balanced evaluation inputs
- Generation configurations for the five evaluated models
- Prompt templates and model registry
- Core generation and standard local scoring code
- Dataset/audit manifests

## Generation

Generation consumes the prepared `eval_examples.expanded.jsonl` file named by `safety_mirage.subset_root`. The configured v14 contexts are already cleaned and capped before generation. The released paper subset is `data/v14_balanced_346/`.

For the camera-ready analysis, filter each v14 model-output and rescored-result file with `scripts/filter_balanced_results.py --require-complete`, then recompute all aggregate tables and figures. The validation flag ensures that the source artifact includes every balanced example. This produces 346 questions, 1,384 rows per model, and 6,920 rows across five models without re-running generation or judging.

## Raw-output archive

Complete raw model responses, local judge traces, and cluster job records are preserved in a private project archive. They are not committed here because they are large and include sensitive model outputs. A future archival release may use a controlled-access repository, LFS, or a DOI-backed archive.

## Final analysis

The final camera-ready aggregate tables and figures will be added under `results/v2/` after the v2 scoring workflow is frozen.

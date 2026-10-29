# Source code

`analysis_pipeline/` contains the reusable model, robustness, context-transfer, stability, bidirectionality, and dimensionality functions used by the public runners.

`data_integrity/01_rebuild_behavior_labels.py` reconstructs stimulus and category labels from the published participant sequences and paired-block design.

`data_integrity/02_recover_context_from_edf.py` recovers imagery/video identity from EDF annotations under a strict structural audit.

Use the entry points under `scripts/analysis/` to execute the reported analyses.

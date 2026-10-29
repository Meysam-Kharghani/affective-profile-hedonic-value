# Affective Profiles and Hedonic Value

This repository contains a secondary behavioral analysis of the public EmoEEG-MC dataset (OpenNeuro accession `ds005540`). It evaluates how emotion-category labels and continuous affective self-reports reconstruct same-trial valence and liking across participants, stimuli, and elicitation contexts.

## Analysis cohorts

The core analysis includes 53 participants (`sub-01` through `sub-53`). After one common complete-case exclusion for a missing liking value, the core models use 2,225 trials.

Context-dependent analyses use a separately audited subset of 45 participants. Imagery/video identity was recovered from EDF annotation TypeIDs together with the published paired three-trial block design. The context cohort contains 1,890 trials, balanced exactly between imagery and video (945 each). Behavioral ratings and hedonic outcomes were not used to infer context.

## Main findings

Emotion category alone carries substantial held-out-participant information about valence and liking (R² = 0.499 and 0.478, respectively). Continuous affective profiles reconstruct both outcomes more accurately (R² = 0.673 for valence and 0.689 for liking). The profile advantage remains under held-out-stimulus evaluation, simultaneous participant-and-stimulus holdout, and within-participant centering.

Dimensionality differs by outcome. Valence is largely captured by an affect-balance axis, whereas liking retains a measurable advantage for the full multidimensional affective profile.

## Cross-context generalization

In the 45-participant context cohort:

- valence, imagery → video: R² = 0.607 (95% CI 0.496–0.688)
- valence, video → imagery: R² = 0.675 (95% CI 0.593–0.736)
- liking, imagery → video: R² = 0.573 (95% CI 0.465–0.653)
- liking, video → imagery: R² = 0.690 (95% CI 0.648–0.727)

The participant-level MAE asymmetry decision rule, fixed before evaluation of the context-transfer results, did not support a reliable directional advantage for either outcome.

## Secondary analyses

Split-half stability was stronger for affect–hedonic coupling and dispersion than for category-specific valuation or the global context contrast. Relative bidirectionality and model-implied surfaces are reported as descriptive secondary analyses.

Personalization was evaluated against the global affective-plus-nominal model. The incremental-personalization support rule, fixed before evaluation of the held-out personalization results, was not satisfied for either valence or liking.

## Repository structure

- `analysis_specification/` — analysis plan, fixed context-analysis cohort, and data-integrity record
- `src/analysis_pipeline/` — reusable analysis functions
- `src/data_integrity/` — behavioral-label reconstruction and EDF-based context recovery
- `scripts/analysis/` — public execution entry points
- `results/main_tables/` — primary aggregate result tables
- `results/source_data/` — aggregate source data used for tables and figures
- `results/supplementary_tables/` — supplementary aggregate results and data-integrity summaries
- `figures/` — figures derived from aggregate results
- `provenance/` — checksums and scientific lineage
- `environment/` — analysis, data-integrity, and figure-generation dependencies

## Interpretation boundary

Predictors and outcomes are same-trial self-reports. The evidence therefore concerns psychometric reconstruction, representational granularity, and within-dataset generalization. It does not establish prospective forecasting, causal effects, biomarkers, or external validation.

See `DATA_INTEGRITY.md` for the label-reconstruction audit and `REPRODUCIBILITY.md` for execution details.

## Author

**Meysam Kharghani**  
Department of Cognitive Neuroscience, Faculty of Education and Psychology, University of Tabriz, Tabriz, Iran  
ORCID: https://orcid.org/0009-0004-9527-756X


## License

Code and documentation in this repository are released under the MIT License. See `LICENSE` for details. The underlying EmoEEG-MC dataset is not redistributed here and remains subject to the licensing terms of its original source.


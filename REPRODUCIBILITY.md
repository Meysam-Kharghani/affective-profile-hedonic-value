# Reproducibility

## Requirements

Install model-analysis dependencies from `environment/requirements-analysis.txt`. EDF context recovery additionally requires `environment/requirements-data-integrity.txt`. Figure-only dependencies are listed in `environment/requirements-figures.txt`.

For the complete pipeline:

```bash
pip install -r environment/requirements-analysis.txt
pip install -r environment/requirements-data-integrity.txt
```

Raw participant-level data are not redistributed in this repository. Obtain EmoEEG-MC from OpenNeuro accession `ds005540`.

## Core analysis

From the repository root:

```bash
python src/data_integrity/01_rebuild_behavior_labels.py \
  "${DATASET_ROOT}" \
  --out "${OUTPUT_DIR}/behavior_labels.csv"

python scripts/analysis/run_core.py \
  --behavior "${OUTPUT_DIR}/behavior_labels.csv" \
  --output-dir "${OUTPUT_DIR}/core" \
  --n-jobs -1

python scripts/analysis/run_robustness.py \
  --canonical "${OUTPUT_DIR}/core/canonical_core.csv" \
  --output-dir "${OUTPUT_DIR}/robustness" \
  --n-jobs -1

python scripts/analysis/run_lowdim.py \
  --canonical "${OUTPUT_DIR}/core/canonical_core.csv" \
  --output "${OUTPUT_DIR}/low_dimensional_sensitivity.csv"
```

## Context-dependent analysis

```bash
python src/data_integrity/02_recover_context_from_edf.py \
  --dataset-root "${DATASET_ROOT}" \
  --behavior "${OUTPUT_DIR}/behavior_labels.csv" \
  --outdir "${OUTPUT_DIR}/context"

python scripts/analysis/run_cross_context.py \
  --canonical "${OUTPUT_DIR}/context/canonical_context.csv" \
  --output-dir "${OUTPUT_DIR}/cross_context" \
  --n-jobs -1
```

The cross-context runner applies the fixed analysis cohort in `analysis_specification/context_analysis_cohort.csv` before fitting models.

## Personalization

```bash
python scripts/analysis/run_personalization.py \
  --canonical "${OUTPUT_DIR}/core/canonical_core.csv" \
  --global-predictions "${OUTPUT_DIR}/core/lopo_predictions.csv.gz" \
  --output-dir "${OUTPUT_DIR}/personalization" \
  --n-jobs -1
```

## Complete pipeline

The core, robustness, dimensionality, context, cross-context, and personalization stages can be run with:

```bash
bash scripts/analysis/run_all.sh "${DATASET_ROOT}" "${OUTPUT_DIR}" -1
```

All public tables and figures are aggregate outputs. Participant-level derived files are generated locally by the analysis scripts and are not distributed in this repository.

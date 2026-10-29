#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 DATASET_ROOT OUTPUT_DIR [N_JOBS]"
  exit 2
fi
DATASET_ROOT="$1"
OUTPUT_DIR="$2"
N_JOBS="${3:--1}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
mkdir -p "$OUTPUT_DIR"
python3 "$ROOT/src/data_integrity/01_rebuild_behavior_labels.py" "$DATASET_ROOT" --out "$OUTPUT_DIR/behavior_labels.csv"
python3 "$ROOT/scripts/analysis/run_core.py" --behavior "$OUTPUT_DIR/behavior_labels.csv" --output-dir "$OUTPUT_DIR/core" --n-jobs "$N_JOBS"
python3 "$ROOT/scripts/analysis/run_robustness.py" --canonical "$OUTPUT_DIR/core/canonical_core.csv" --output-dir "$OUTPUT_DIR/robustness" --n-jobs "$N_JOBS"
python3 "$ROOT/scripts/analysis/run_lowdim.py" --canonical "$OUTPUT_DIR/core/canonical_core.csv" --output "$OUTPUT_DIR/low_dimensional_sensitivity.csv"
python3 "$ROOT/src/data_integrity/02_recover_context_from_edf.py" --dataset-root "$DATASET_ROOT" --behavior "$OUTPUT_DIR/behavior_labels.csv" --outdir "$OUTPUT_DIR/context"
python3 "$ROOT/scripts/analysis/run_cross_context.py" --canonical "$OUTPUT_DIR/context/canonical_context.csv" --output-dir "$OUTPUT_DIR/cross_context" --n-jobs "$N_JOBS"
python3 "$ROOT/scripts/analysis/run_personalization.py" --canonical "$OUTPUT_DIR/core/canonical_core.csv" --global-predictions "$OUTPUT_DIR/core/lopo_predictions.csv.gz" --output-dir "$OUTPUT_DIR/personalization" --n-jobs "$N_JOBS"
echo "PASS: analysis pipeline completed"

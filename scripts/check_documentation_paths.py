#!/usr/bin/env python3
from pathlib import Path
R=Path(__file__).resolve().parents[1]
required=['README.md','DATA_INTEGRITY.md','ANALYSIS_OVERVIEW.md','REPRODUCIBILITY.md','VALIDATION.md','analysis_specification','results/main_tables','results/source_data','results/supplementary_tables','figures','provenance','scripts/analysis','src/analysis_pipeline','src/data_integrity']
for item in required:
    assert (R/item).exists(), item
print('PASS: repository paths')

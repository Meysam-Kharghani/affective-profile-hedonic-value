# Portability

The repository contains no absolute paths tied to a particular computer and does not redistribute raw participant-level data.

All public execution entry points accept dataset and output locations as command-line arguments. Analysis modules are resolved relative to the repository root. The fixed context-analysis cohort is stored in `analysis_specification/context_analysis_cohort.csv` so the reported cross-context cohort can be reproduced independently of directory layout.

EDF context recovery depends on `pyedflib` with `mne` as a fallback reader; both are listed in `environment/requirements-data-integrity.txt`. Use `REPRODUCIBILITY.md` for the recommended commands.

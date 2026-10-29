# Data integrity and label reconstruction

## Label-alignment issue

A data-integrity audit identified a mismatch between raw behavioral row order and the reordered nominal stimulus/category sequence used by an earlier analysis. Raw behavioral rows follow the randomized paired-block presentation order. Treating those rows as if they were already in derivative reorder order misassigned nominal stimulus and category labels.

The continuous same-trial ratings were not relabeled. The issue affected nominal stimulus/category alignment and analyses that depended on those labels.

## Stimulus and category reconstruction

For `sub-01` through `sub-53`, the published 21-item participant sequence is divided into seven three-item chunks. The experimental design presents each chunk in a paired imagery/video block, so each three-item chunk occurs twice consecutively in the raw behavioral sequence. This reconstructs stimulus code and emotion category without requiring imagery/video identity.

`sub-54` is not included in the label-dependent core because a participant-specific 21-item sequence was not available to the reconstruction procedure. Participants `sub-55` through `sub-60` are not included because their documented missing-trial patterns require a separate incomplete-trial alignment procedure. No nominal labels are assigned inferentially.

One `sub-01` trial has a missing liking value. That row is excluded for both outcomes so every core model uses the same 2,225 complete-case trials.

## Context recovery

Imagery/video order within each paired block was randomized and was therefore recovered independently from EDF annotation TypeIDs. The context audit requires a legal 42-trial sequence with 21 imagery and 21 video trials, homogeneous three-trial blocks, and one imagery plus one video block within each consecutive block pair.

Thirty-five participants passed directly. Ten additional participants contained extra context markers but were accepted only when exactly one legal seven-pair block-order sequence was compatible with the observed marker order. Eight participants remained unresolved and were excluded from context-dependent analyses only.

The resulting context-analysis cohort contains 45 participants and 1,890 trials, exactly balanced between imagery and video. Behavioral ratings and hedonic outcomes were not used in context recovery.

## Consequence for interpretation

The reconstructed category-only model is substantially informative (R² approximately 0.50 for valence and 0.48 for liking). Continuous affective profiles remain more accurate than category alone, and that advantage persists across the reported robustness analyses.

The analysis specification, cohort definition, and aggregate audit tables in this repository document the final data-integrity decisions used for the reported results.

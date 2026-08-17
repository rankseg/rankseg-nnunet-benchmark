# Rejected experiments and excluded datasets

The public codebase contains only the reproducible Full-16 benchmark path. Historical outputs may remain for audit,
but implementations that cannot support the merge claim have been removed from the maintained code.

## Post-hoc volume guard

A whole-case fallback to argmax was triggered when any foreground-class RankSEG/argmax volume ratio exceeded 2x.
It repaired one Prostate regression retrospectively, but suppressed useful Colon gains and missed two large Lung
regressions. The rule was rejected as a general safety mechanism, and its CLI/config/evaluation implementation was
removed. The Full-16 result always uses unguarded RankSEG.

## Dice-loss probability inversion

A binary transformation attempted to recover calibrated probabilities from outputs assumed to be near the optimum
of CE plus global Soft-Dice. Smoke tests did not show reliable RankSEG improvements, while the method depended on an
unknown target-domain foreground prior and an unrealistic optimizer-optimum assumption. The module and smoke script
were removed.

## Slice-wise RankSEG

Applying RankSEG independently to axial slices changes the decision unit while the benchmark evaluates whole-volume
Dice. Because this is not a fair substitute for the predeclared 3D decoder, the sensitivity script was removed.

## Excluded datasets

- Task035 ISBI longitudinal MS lesions: excluded from public aggregates because the embedded challenge license
  restricts use and redistribution. Local historical outputs are not part of the maintained pipeline.
- Task076 and Task089 microscopy: excluded because the released checkpoints were trained on all labeled sequences,
  so no leakage-free evaluation was possible. Dataset-preparation code for Task076 was removed.

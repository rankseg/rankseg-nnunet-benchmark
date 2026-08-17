# Benchmark protocol

This protocol was fixed during construction of the Core-12 subset. Later cohorts follow the same decoder comparison,
and every addition is retained regardless of direction. Deviations must be recorded and must not silently replace
the Full-16 main analysis.

## Question and estimand

The main question is whether replacing probability-map argmax with the public RankSEG default produces consistent
benefits without material dataset-level regressions.

The estimand for each dataset is:

```text
Δ Dice = foreground macro Dice(RankSEG) - foreground macro Dice(argmax)
```

Dice is the only metric reported in the cross-dataset benchmark. Runtime and peak decoder memory are mandatory
engineering outcomes.

## Fixed decoder

- RankSEG metric: `dice`
- solver: `RMA`
- output mode: `multiclass`
- pruning probability: `0.5`
- smooth: `0.0`
- unassigned policy: `max_score`
- argmax: channel dimension argmax on the identical cached array

There is no per-dataset threshold tuning. Sensitivity analyses may be added, but they cannot replace the default
result. Probability arrays are converted to float32 for both decoders. When the selected nnU-Net configuration ships
prelearned connected-component postprocessing, apply that fixed operation identically after argmax and RankSEG. Raw
decoder results may be retained as sensitivity analyses but do not replace the predeclared configuration result.

## Probability interpretation and calibration

For mutually exclusive label tasks, nnU-Net applies softmax before exporting the arrays consumed by this benchmark.
The evaluator verifies that these arrays are finite, nonnegative, and normalized, but normalization is not evidence
of calibration. In particular, the official nnU-Net v1 checkpoints used in Full-16 were trained with the default
cross-entropy plus soft-Dice objective. The benchmark must not infer overconfidence from that loss function alone.

No temperature scaling, recalibration, probability transformation, or calibration-set tuning is part of Full-16.
Argmax and RankSEG receive the same unmodified exported array. The current estimand is consequently RankSEG's
plug-in effect on standard official nnU-Net outputs, not its performance under calibrated posterior probabilities.

Any CE-only follow-up is a separate experiment and cannot retroactively replace the fixed Full-16 result. To
attribute a difference to the training loss, CE-only and CE + soft-Dice models must be trained with matched software,
data, folds, architecture, augmentation, schedule, checkpoint rule, ensembling, postprocessing, and random seeds.
Comparing a newly trained CE-only model only with a historical official checkpoint must be labeled exploratory.
The follow-up must report all four Dice results (argmax and RankSEG for each loss), their interaction
`(RankSEG - argmax)_CE - (RankSEG - argmax)_CE+Dice`, and predeclared calibration outcomes. Claims of overconfidence
require direct calibration evidence rather than softmax magnitude alone. Calibration endpoints should include
case-aggregated negative log-likelihood and multiclass Brier score plus foreground/classwise ECE; binning, background
handling, and any voxel subsampling must be fixed before results are inspected, and voxels must not be treated as
independent observations for inference.

## Data and model selection

1. Use an independent labeled test set when one exists and training overlap can be ruled out.
2. Otherwise use strict five-fold out-of-fold prediction: case `i` is predicted only by the fold whose training set
   excluded case `i`.
3. Never evaluate a five-fold ensemble on its own training cases.
4. Use the configuration selected by the original nnU-Net model release. When the release provides a model-selection
   table, its best configuration is the primary result. Other configurations are sensitivity analyses.
5. Record model/data URL, checksum, split, checkpoint, trainer, plans, software versions, and RankSEG commit.
6. Audit ground-truth compatibility with the model release. Any release-specific correction must live in a separate,
   checksum-recorded label layer, be supported by publisher artifacts, and leave downloaded source data untouched.

Official nnU-Net v1 weights provide the broad model/dataset sample. Because v1 weights cannot load into v2, current
v2 checkpoints are also required as an implementation-path replication. RankSEG operates on exported probabilities,
so the decoder comparison itself is version-independent.

## Supported task semantics

Full-16 decodes mutually exclusive multiclass probability channels. Non-consecutive labels are handled explicitly
through `channel_labels`. For Task001 BrainTumour, both decoders operate on the same four-channel exclusive softmax;
their resulting label maps are then scored as the three official nested BraTS regions. This is region-based metric
evaluation, not evidence for nnU-Net v2 models trained with overlapping sigmoid region channels.

True region-trained models require a separate path that applies RankSEG to region channels and performs the
release-compatible region-to-label conversion. Ignore-label masking is implemented in TP/FP/FN counts and covered by
unit tests, but no Full-16 dataset uses an ignore label. Neither feature should be claimed as empirically validated
by this package until an eligible model/dataset result is added under the fixed protocol.

## Metric aggregation

For every case and foreground label:

- Dice = `2 TP / (2 TP + FP + FN)`;
- if prediction and reference are both empty, the value is `NaN`.

Following nnU-Net-style foreground aggregation:

1. average each label across cases with NaNs omitted;
2. average the foreground label means.

This prevents large structures and large datasets from dominating the result. Micro-averaged voxel metrics may be
reported only as secondary diagnostics.

## Uncertainty and consistency

- Main inference stays on the overall mean Dice delta. Case-level and quantile results are descriptive and cannot
  overturn a neutral or negative primary result.
- Within each dataset, resample cases with replacement and recompute the full label-then-foreground macro metric.
  Report the paired 95% bootstrap interval for the decoder delta.
- Retain `RankSEG Q05/Q10 - argmax Q05/Q10` and the 5th/10th percentiles of paired Dice improvement as diagnostic
  outputs. Classify cases as improved (`Δ Dice > +1 pp`), stable (`|Δ Dice| <= 1 pp`), or worsened
  (`Δ Dice < -1 pp`). Also report symmetric counts beyond 5 and 10 percentage points.
- Across datasets, use exactly one primary result per dataset. Report every delta, counts of positive/tied/negative
  datasets, argmax and RankSEG absolute means, median/mean/range, an exact two-sided sign test, and Wilcoxon
  signed-rank test.
- Also report per-label and per-case regressions. No result may be removed because RankSEG loses.

The maintainer asked for consistent benefits rather than a hit-or-miss method. Before proposing a merge, the evidence
package should contain at least 11 diverse datasets, a positive dataset-level median, substantially more wins than
losses, and an explicit review of every material regression. These are evidence targets, not a claim that the
maintainer pre-approved a numerical threshold.

## Core-12 construction history: locked prospective confirmation cohort (2026-07-28)

Six Core-12 dataset results (Heart, Hippocampus, Prostate, Spleen, Colon, and Lung) were known when this cohort was
locked. Task007 inference and probability ensembling were complete, but its argmax-versus-RankSEG evaluation had not
been run. The following order and inclusion rule are fixed before inspecting any of their RankSEG results:

1. Task007 Pancreas: official `3d_fullres` + `3d_cascade_fullres` probability ensemble;
2. Task008 HepaticVessel: official `3d_lowres` + `3d_fullres` probability ensemble;
3. Task003 Liver: official `3d_lowres` + `3d_fullres` ensemble selected by the corrected archive-metadata fallback;
4. Task001 BrainTumour: official `3d_fullres`, scored with the official nested-region metric unions;
5. Task055 SegTHOR: official archive-best configuration, selected from publisher metadata before decoder evaluation;
6. Task027 ACDC: official archive-best configuration, with ED/ES observations clustered by patient;
Datasets are not removed or replaced because RankSEG loses. Access, provenance, label-semantics, or leakage failures
must be recorded before decoder evaluation. These historical Core-12 partitions remain descriptive sensitivity
analyses and do not replace the Full-16 main result.

Protocol correction, 2026-08-11 (before any Task003 inference or decoder result): an archive audit found that the
fallback parser used `dc_per_class_pp_per_class`, which records each candidate connected-component operation in
isolation, instead of the official final-pipeline field `dc_per_class_pp_all`. Correcting this provenance bug changes
Task003's predeclared configuration from `3d_lowres` (mis-scored as 0.77955) to the official `3d_lowres` + `3d_fullres`
ensemble (0.81107). The correction is committed before seeing Task003 probabilities or RankSEG outcomes and applies
uniformly to all fallback archives; Task006's selected configuration remains unchanged.

Protocol amendment, 2026-07-28 (before Task008 or Task055 decoder results): at the user's request to prioritize a
smaller cohort, Task055 may run immediately after Task008 and before Task003/Task001. This is an execution-order-only
change based on cohort size and data availability. The locked Task055 configuration, inclusion rule, endpoints, and
requirement to report losses are unchanged; no observed RankSEG result motivated the reordering.

The main endpoint remains the continuous dataset-level mean foreground Dice delta. To make "hit or miss"
auditable rather than rhetorical, every release table additionally reports:

- dataset wins, ties, and losses using the unrounded delta, plus descriptive practical bands at -0.05 and -0.10 Dice
  percentage points;
- per-dataset paired confidence intervals and macro mean/median across datasets;
- symmetric case-level event counts for changes beyond 1, 5, and 10 Dice percentage points in either direction;
- the worst and best case delta, without removing empty-label cases or other valid outliers;
- patient- or sequence-clustered uncertainty whenever several observations come from the same biological subject.

These safety summaries are secondary and cannot replace the overall mean. They are required because a positive mean
can coexist with rare, severe regressions.

Rejected post-hoc decoder variants and ineligible datasets are documented in
[`REJECTED_EXPERIMENTS.md`](REJECTED_EXPERIMENTS.md); their implementations are intentionally absent from the
maintained benchmark path.

## Reproducibility checks

- Verify every downloaded model and dataset checksum.
- Validate probability range, finite values, channel count, geometry, and normalization.
- Compare probability argmax with the native hard mask when available. Differences from v1 float16 probability
  storage or nnU-Net postprocessing are counted and reported.
- Mark subset runs as non-scientific; aggregation rejects them.
- Save complete per-case/per-label counts so every summary can be independently recomputed.

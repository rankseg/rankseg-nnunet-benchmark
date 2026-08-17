# Full 16-dataset benchmark

Completed 2026-08-17. The Full-16 benchmark is the main result: one predeclared configuration per dataset, identical
probabilities for argmax and RankSEG, and every valid result retained regardless of direction. It combines the
Core-12 official nnU-Net v1 OOF subset with four cohorts that broaden the evaluation design and implementation path.

## Overall Full-16 result

- 16 datasets and 2,181 labeled cases; 13 positive versus 3 negative dataset-level Dice deltas.
- Macro-average Dice across the 16 datasets, weighting each dataset equally: **83.63646% argmax** versus
  **83.84318% RankSEG**.
- Dataset-level mean Dice delta: **+0.20671 pp**; median: **+0.00979 pp**; range: -0.08217 to +1.16690 pp.
- Overall exact two-sided sign test: p=0.02127; overall Wilcoxon signed-rank test: p=0.01099.
- The four non-Core cohorts contribute 200 cases and three positive versus one negative dataset result. Their
  distinct OOF, independent-test, clustering, and v2 provenance is recorded rather than treated as a separate
  competing benchmark.

## Core-12 official-v1 OOF sensitivity subset

Completed 2026-08-14. Only official-best configurations selected before RankSEG evaluation are included. All
predictions are five-fold out-of-fold, RankSEG is applied to each complete 3D volume, and official nnU-Net
postprocessing is applied identically to argmax and RankSEG. This subset preserves the original benchmark sequence
as a sensitivity analysis; it is not the main headline cohort.

| Dataset | Cases | Locked configuration | Argmax Dice | RankSEG Dice | Dice delta (pp) | 95% paired CI (pp) |
|---|---:|---|---:|---:|---:|---:|
| BrainTumour | 484 | `3d_fullres`; BraTS regions | 84.57217% | 84.57988% | +0.00771 | [-0.02566, +0.04626] |
| Heart | 20 | `3d_fullres` | 93.29433% | 93.29623% | +0.00191 | [-0.00414, +0.00824] |
| Liver | 131 | `3d_lowres` + `3d_fullres` ensemble | 80.99903% | 81.82314% | +0.82412 | [+0.06237, +1.62948] |
| Hippocampus | 260 | `3d_fullres` | 88.90847% | 88.91252% | +0.00405 | [-0.00253, +0.01083] |
| Prostate | 32 | 2D + `3d_fullres` ensemble | 75.95709% | 75.98004% | +0.02295 | [-2.58438, +1.99026] |
| Lung | 63 | `3d_lowres` + `3d_fullres` ensemble | 72.28822% | 72.74899% | +0.46077 | [-0.68225, +1.56630] |
| Pancreas | 281 | `3d_fullres` + cascade ensemble | 68.39597% | 69.56287% | +1.16690 | [+0.70662, +1.64533] |
| HepaticVessel | 303 | `3d_lowres` + `3d_fullres` ensemble | 68.63054% | 69.16535% | +0.53481 | [+0.23693, +0.86024] |
| Spleen | 41 | `3d_fullres` + cascade ensemble | 97.23659% | 97.24320% | +0.00661 | [-0.01339, +0.03832] |
| Colon | 126 | `3d_cascade_fullres` | 48.91759% | 49.15961% | +0.24203 | [-0.99406, +1.52243] |
| ACDC | 200 | 2D + `3d_fullres` ensemble; patient-clustered | 92.45381% | 92.38008% | -0.07373 | [-0.24076, +0.06538]* |
| SegTHOR | 40 | `3d_fullres` + cascade ensemble | 91.51999% | 91.51811% | -0.00188 | [-0.04117, +0.02730] |

\* ACDC interval is the paired patient-cluster bootstrap over 100 patients; its two ED/ES observations are never
split between folds or independently resampled. Other intervals are paired case bootstraps with 10,000 samples.

## Core-12 descriptive result

- 12 datasets, 1,981 labeled cases, and 10 positive versus 2 negative dataset-level Dice deltas.
- Dataset-level mean Dice delta: **+0.26635 pp**; median: **+0.01533 pp**; range: -0.07373 to +1.16690 pp.
- Five datasets improve Dice by more than 0.05 pp. ACDC is the only dataset below -0.05 pp; no dataset falls below
  -0.10 pp. SegTHOR's -0.00188 pp is practically negligible.

The six-dataset development cohort is 6/6 positive with mean +0.12305 pp. The prospectively locked six-dataset
confirmation cohort is 4/6 positive with mean +0.40966 pp and median +0.27126 pp. These historical partitions are
retained for transparency but do not replace the Full-16 main result.

## Hit-or-miss and downside audit

The result supports a positive average effect but not universal non-degradation:

- ACDC declines by 0.07373 pp Dice. Its worst case, `patient049_frame11`, declines by 11.01 pp because RankSEG
  expands all three cardiac structures, especially the right ventricle (RV Dice 0.7773 to 0.5291). The aggregate
  foreground prediction grows from 10,752 to 12,460 voxels for a target of 11,066. A visualization is stored at
  `outputs/primary_aggregate/visualizations/acdc_patient049_frame11.png`.
- SegTHOR declines by only 0.00188 pp Dice and its paired interval crosses zero.
- Across the 1,981 Core-12 cases, changes beyond 1 pp comprise 131 losses versus 239 gains; beyond 5 pp, 26 versus 63;
  beyond 10 pp, 9 versus 31. The worst case remains `colon_107` (-44.63 pp) and the best `colon_108` (+51.42 pp).
- At the ±1 Dice pp practical threshold, 239 of the 1,981 Core-12 cases improve, 1,611 remain stable, and 131 worsen.
  These case counts complement the dataset-level mean and make the non-monotonic behavior explicit.
- Eight Core-12 configurations use probability ensembles: 6 improve and 2 decline, with mean +0.36757 pp. The four
  single-model configurations all improve, with mean +0.06393 pp. Ensembling is helpful on average here but does not
  guarantee a gain.

The defensible merge claim is therefore that RankSEG is a reproducible optional decoder with positive average
benefit across diverse official nnU-Net models, not a monotonic replacement for argmax. The per-dataset table and
case-level counts and material regressions must accompany that claim.

## Integrity checks

- Official model archives are publisher-size and checksum verified; official validation metadata selected each Core-12
  before RankSEG evaluation.
- Ensemble inputs are arithmetic means of fold-matched softmax probability maps, before either decoder.
- Official v1 inference uses its default test-time mirroring. The TotalSegmentator v2 replication follows its
  `nnUNetTrainerNoMirroring` checkpoint and therefore disables mirroring.
- Native-mask checks confirm the reconstructed argmax path. ACDC has exactly zero mismatched voxels; BrainTumour has
  805/4,321,152,000 (1.86e-7), attributable to float16 probability storage near ties.
- ACDC uses the official patient-level KFold protocol: every fold contains 80 training and 20 validation patients,
  with two ED/ES cases per patient and no patient leakage.
- The complete automated test suite passes (42 tests).

The canonical public Full-16 machine-readable outputs are in `evidence/full16/`; its source dataset summaries,
paired case deltas, and available cohort audits are in `evidence/datasets/`. The ignored local `outputs/` tree
preserves the complete evaluator outputs and Core-12 visualizations.

## Four robustness cohorts included in Full-16

These four cohorts are part of the main Full-16 benchmark. Their distinct designs remain explicit:

| Task | Dataset | Cases | Configuration | Dice delta (pp) | Uncertainty note |
|---|---|---:|---|---:|---|
| Task075 | Fluo-C3DH-A549 | 90 | `3d_fullres` | +0.01186 | Four-sequence clustered CI crosses zero |
| Task024 | PROMISE12 former hidden test | 30 | 2D + `3d_fullres` ensemble | -0.08217 | Independent-test case CI crosses zero |
| Task038 | CHAOS MRI | 60 | 2D + `3d_fullres` ensemble | +0.17601 | 20-patient clustered CI crosses zero |
| v2 Dataset291 | CHAOS CT / TotalSegmentator | 20 | `3d_fullres` fold 0; no mirroring | +0.00542 | External liver GT; no case changes beyond 1 pp |

At the ±1 Dice pp practical threshold, 246 of the 2,181 Full-16 cases improve, 1,802 remain stable, and 133 worsen.
Beyond 5 pp there are 26 losses and 64 gains; beyond 10 pp, 9 losses and 31 gains. The positive overall result
therefore does not imply guaranteed improvement for every dataset or patient.

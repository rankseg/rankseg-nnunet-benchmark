# Dataset execution plan

The fastest defensible path is a broad official-model cohort plus current-v2 replications. Ten MSD tasks alone are
not sufficient evidence of domain diversity, so microscopy/electron-microscopy tasks are planned as well.

| Priority | Dataset/task | Modality and target | Predeclared official primary | Status |
|---:|---|---|---|---|
| 1 | Task004 Hippocampus | MRI; anterior/posterior hippocampus | `3d_fullres` | Complete (260 cases) |
| 2 | Task002 Heart | MRI; left atrium | `3d_fullres` | Complete (20 cases) |
| 3 | Task005 Prostate | T2+ADC MRI; prostate zones | 2D + `3d_fullres` ensemble | Complete (32 cases; material regression found) |
| 4 | Task009 Spleen | CT; organ | `3d_fullres` + cascade ensemble | Complete (41 cases; first prospective guard task) |
| 5 | Task010 Colon | CT; tumour | `3d_cascade_fullres` | Complete (126 cases; second prospective guard task) |
| 6 | Task006 Lung | CT; lung tumour | `3d_lowres` + `3d_fullres` ensemble | Complete (63 cases; third prospective guard task) |
| 7 | Task007 Pancreas | CT; organ and tumour | `3d_fullres` + cascade ensemble | Complete (281 cases; +1.16690 pp Dice) |
| 8 | Task008 HepaticVessel | CT; vessel and tumour | `3d_lowres` + `3d_fullres` ensemble | Complete (303 cases) |
| 9 | Task003 Liver | CT; organ and tumour | `3d_lowres` + `3d_fullres` ensemble | Complete (131 cases; +0.82412 pp Dice) |
| 10 | Task001 BrainTumour | Multisequence MRI; tumour regions | `3d_fullres` | Complete (484 cases; +0.00771 pp Dice) |
| 11 | Task055 SegTHOR | CT; four thoracic organs | `3d_fullres` + cascade ensemble | Complete (40 cases) |
| 12 | Task027 ACDC | Cine MRI; three cardiac structures | 2D + `3d_fullres` ensemble | Complete (200 cases/100 patients; -0.07373 pp Dice; patient-clustered CI) |
| 13 | Task075 Fluo-C3DH-A549 | Bright-field microscopy; cell masks | `3d_fullres` | Complete (90 cases/four source sequences; +0.01186 pp Dice) |
| 14 | Task024 PROMISE12 | Multicenter T2 MRI; prostate | 2D + `3d_fullres` ensemble | Complete (30 independent former hidden-test cases; -0.08217 pp Dice) |
| 15 | Task076 Fluo-N3DH-SIM | Fluorescence microscopy; cell core/border | Not eligible | Official archive has only an `all` checkpoint trained on both labeled sequences |
| 16 | Task089 Fluo-N2DH-SIM | Fluorescence microscopy; cell masks | Not eligible | Official archive has only an `all` checkpoint trained on all labeled frames |
| 17 | TotalSegmentator organs | CT; many organs | Official v2 checkpoint; independent labels required | Own-dataset evaluation rejected: current 1559-case checkpoint may overlap the public 1228 cases |
| 18 | Task038 CHAOS Variant 2 | T1/T2 MRI; liver, kidneys, spleen | 2D + `3d_fullres` ensemble | Complete (60 volumes/20 patients; +0.17601 pp Dice; patient-clustered CI crosses zero) |
| 19 | Task035 ISBI longitudinal MS lesions | Longitudinal multisequence MRI; MS lesions | Not eligible for public aggregate | Excluded due embedded challenge-license restrictions; local audit retained pending organizer permission |
| 20 | CHAOS CT / TotalSegmentator v2 | CT; liver (Dataset291 decodes 25 native classes) | External 20-case CHAOS CT cohort | Complete (+0.00542 pp Dice); exact native argmax agreement on all 20 cases; included in Full-16 with partial liver-only GT explicitly recorded |
| 21 | WORD independent test | CT; 16 abdominal organs | Author checkpoint on the predeclared 30-case test split | Selected; official data archive temporarily blocked by Google Drive quota, checkpoint downloaded and verified |

The ten MSD dataset URLs and MD5 values are in `registry/msd_datasets.yaml`. All 22 author-published model archives,
including the non-MSD extension tasks, are in `registry/nnunet_v1_official_models.yaml`.

Implementations for rejected decoder experiments and datasets that cannot enter a leakage-free public benchmark are
not retained in the main codebase; see [`REJECTED_EXPERIMENTS.md`](REJECTED_EXPERIMENTS.md).

Primary configurations above come from official validation metadata read before RankSEG evaluation and recorded in
`results/OFFICIAL_MSD_MODEL_SELECTION.csv`. The task-level `summary.csv` is preferred. Task003 and Task006 archives
do not contain one, so their selections use the final cross-validation Dice in each configuration's official
`postprocessing.json`; the registry marks these rows as `available_postprocessing_fallback`. On 2026-08-11, before
Task003 inference or decoder evaluation, an audit corrected the fallback field from the diagnostic
`dc_per_class_pp_per_class` to the released pipeline's final `dc_per_class_pp_all`. This changed Task003's locked
primary from `3d_lowres` to the `3d_lowres` + `3d_fullres` ensemble; Task006's selection was unchanged.

## Execution order

1. Heart, Prostate, and Spleen established the binary/multiclass and ensemble orchestration.
2. Finish Task008, then run the relatively small SegTHOR cohort while its official data/model are available.
3. The Core-12 official-v1 OOF subset is complete; retain it as a sensitivity analysis with all positive and negative results.
4. Task075 adds cross-domain microscopy evidence with an exact four-sequence OOF protocol.
5. Task076 and Task089 are excluded after remote archive audits found only `all` checkpoints; their labeled training
   data cannot support leakage-free evaluation of those checkpoints.
6. Task024 is complete on the 30 former hidden PROMISE12 test cases whose masks are now public; include this
   independent-test result in Full-16 while recording its distinct evaluation design.
7. Task038 is complete with exact patient-level OOF folds and separate 2D/3D/ensemble sensitivity results.
8. Task035 is excluded from the public/main aggregate. Its source, predictions, metrics, and report remain local-only
   pending explicit organizer permission and are not counted in any headline statistic.
9. Full-16 is the main aggregate with 16 datasets/2,181 cases. Report Core-12 only as the official-v1 OOF
   sensitivity subset, and investigate every regression in the full cohort.

Task061 CREMI is excluded from the primary dataset count: its legacy converter exposes only three genuinely
independent labeled volumes and uses overlapping train/validation samples. Task029 LiTS is not counted separately
unless a provenance audit rules out overlap with MSD Task003 Liver.

## Evidence package for the nnU-Net issue

- a public repository with manifests, checksums, exact commands, and environment lock;
- a table with every dataset, not only the aggregate;
- paired confidence intervals and dataset-level win/loss statistics;
- runtime and peak-memory overhead across volume sizes/class counts;
- examples of both gains and regressions with no cherry-picking;
- v2-specific tests for ordinary labels, non-consecutive labels, region training, ignore labels, ensembling, and saved
  probabilities;
- a small optional integration with no mandatory RankSEG dependency if the benchmark supports inclusion.

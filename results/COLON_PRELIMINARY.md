# MSD Colon preliminary result

Task010 is the fifth predeclared primary dataset and the second dataset evaluated prospectively after freezing the
separate 2x class-volume safety track. The primary result remains the unguarded RankSEG decoder.

## Protocol

- 126 MSD Colon training cases, each predicted exactly once by the nnU-Net v1 fold that did not train on it.
- Default nnU-Net split: `KFold(5, shuffle=True, random_state=12345)`.
- Official-best configuration selected from the model archive before RankSEG evaluation: `3d_cascade_fullres`, with
  fold-matched `3d_lowres` inputs. The archive reports validation Dice 0.4937 for this configuration, higher than all
  components and ensembles listed in its `summary.csv`.
- Full nnU-Net mirroring TTA for both the low-resolution and cascade stages.
- The official postprocessing file selects no connected-component operations (`for_which_classes: []`), so the
  primary result uses the raw masks for both decoders.
- Fixed RankSEG settings: Dice target, RMA solver, multiclass output, pruning probability 0.5, no smoothing, and
  `max_score` assignment for any unassigned voxel.
- 10,000 paired case-bootstrap samples. Overall mean Dice is primary; difficult-case summaries are secondary.

The downloaded model archive was 5,003,911,923 bytes and matched MD5
`ffa6a2ee13a41cdbde3addc151d3c773`; the dataset archive matched MD5
`bad7a188931dc2f6acf72b08eb6202d0`. All 126 softmax arrays, crop metadata files, and native masks were present, and
the fold counts were 26/25/25/25/25. Every `(case, fold)` pair matched the precomputed held-out assignment.

The evaluator's argmax masks differed from the native nnU-Net masks at only 141 of 3,535,273,984 voxels
(3.99e-8; at most 10 voxels in one case), as expected from decoding float16 saved probabilities. An independent run
of nnU-Net v1's `nnUNet_evaluate_folder` gave argmax Dice 0.4891820, within 0.00062 percentage points of this
evaluator's 0.4891759.

The current inference result is 0.44849 percentage points below the archive's 2019 raw validation Dice 0.4936669.
The metric implementation and reconstructed masks have been independently verified, so this is a model-inference
reproduction discrepancy, not a RankSEG/argmax evaluation discrepancy. The paired comparison remains on exactly the
same current probability tensors, but the release report must retain this limitation.

## Primary mean result

| Metric | Argmax | RankSEG | Mean delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|
| Dice | 48.91759% | 49.15961% | +0.24203 | [-0.99406, +1.52243] |

At the ±1 Dice pp threshold, 14 cases improve, 90 remain stable, and 22 worsen. The largest gain is `colon_108`, from 0.35470% to
51.77802% Dice (+51.42332 pp): argmax has TP/FP/FN 20/5/11,232, whereas RankSEG has 4,208/794/7,044. The largest
regression is `colon_107`, from 84.59887% to 39.96566% (-44.63320 pp): RankSEG changes TP/FP/FN from
4,334/1,178/400 to 4,423/12,977/311. RankSEG therefore recovers recall in both examples but can severely overestimate
volume when its ranking-based size decision is poorly calibrated.

The positive mean Dice is the largest dataset-level gain observed so far, but its confidence interval crosses zero
and case-level effects are strongly bidirectional. This is evidence that the decoder can materially help difficult
cases, not evidence that it is already a uniformly safe replacement.

## Prospective safety track

The frozen symmetric 2x foreground-volume guard triggered on 13/126 cases. It caught the large `colon_107`
regression, but it also discarded the still larger `colon_108` gain and several other gains. Guarded mean Dice is
48.76087%, a delta of -0.15672 pp versus argmax (95% paired CI [-0.61394, +0.30027] pp). This second prospective check
rejects the simple 2x rule as a general safety solution; the guarded result is not eligible to replace the primary.

## Decoder resource cost

On an RTX 3090, mean decoder time was 0.68 ms for argmax and 11.53 ms for RankSEG. The no-op postprocessing path took
about 30 ms for either method. Maximum measured CUDA allocation was 2.85 GiB for argmax and 9.44 GiB for RankSEG.
These timings exclude nnU-Net model inference and probability-file loading.

Machine-readable primary results are under `outputs/Task010_Colon_3d_cascade_fullres_oof/`; the prospective guard
result is under `outputs/Task010_Colon_guarded_prospective/`.

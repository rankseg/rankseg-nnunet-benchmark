# MSD Spleen preliminary result

Task009 is the fourth predeclared primary dataset and the first dataset evaluated after freezing the separate 2x
class-volume safety track. The primary result remains the unguarded RankSEG decoder.

## Protocol

- 41 MSD Spleen training cases, each predicted exactly once by the nnU-Net v1 fold that did not train on it.
- Default nnU-Net split: `KFold(5, shuffle=True, random_state=12345)`.
- Official-best configuration selected from the model archive before RankSEG evaluation: arithmetic mean of the
  `3d_fullres` and `3d_cascade_fullres` softmax arrays. The archive reports validation Dice 0.9661 and 0.9715 for the
  components and 0.9724 for their ensemble.
- Full nnU-Net mirroring TTA and fold-matched low-resolution predictions for every cascade case.
- The archive's connected-component rule (`for_which_classes: [[1]]`) was applied identically after argmax and
  RankSEG.
- Fixed RankSEG settings: Dice target, RMA solver, multiclass output, pruning probability 0.5, no smoothing, and
  `max_score` assignment for any unassigned voxel.
- 10,000 paired case-bootstrap samples. Overall mean Dice is primary; difficult-case summaries are secondary.

The downloaded model archive was 5,025,132,998 bytes and matched MD5
`3a7bef6d372b6fd9e1d7454a23bd793f`. The evaluator's postprocessed argmax masks matched the masks produced by the
native nnU-Net v1 ensemble and postprocessing functions at every one of 956,825,600 voxels. The reproduced argmax
Dice (0.9723659) is within 0.00129 percentage points of the archive's postprocessed validation Dice (0.9723530).

## Primary mean result

| Metric | Argmax | RankSEG | Mean delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|
| Dice | 97.23659% | 97.24320% | +0.00661 | [-0.01339, +0.03832] |

All 41 cases remain stable within ±1 Dice pp. The positive mean is driven mainly by the most difficult case,
`spleen_44`, whose Dice changes from 93.16429% to 93.71940% (+0.55511 pp). The largest regression is `spleen_24`,
from 95.54016% to 95.40651% (-0.13365 pp). This is a small positive mean with an uncertainty interval crossing zero,
not standalone evidence of a reliable improvement.

## Prospective safety track

The frozen symmetric 2x foreground-volume guard triggered on 0/41 cases. It therefore preserved the difficult-case
gain and produced the same +0.00661 pp mean Dice result up to a negligible connected-component tie difference. This
first prospective check shows no false fallback on Task009, but does not validate the guard by itself.

## Decoder resource cost

On an RTX 3090, mean decoder time was 0.69 ms for argmax and 10.61 ms for RankSEG. Mean official postprocessing time
was about 142 ms for either method. Maximum measured CUDA allocation was 672 MiB for argmax and 2.17 GiB for
RankSEG. These timings exclude nnU-Net model inference and probability-file loading.

Machine-readable primary results are under
`outputs/Task009_Spleen_ensemble_postprocessed_oof/`; the prospective guard result is under
`outputs/Task009_Spleen_guarded_prospective/`.

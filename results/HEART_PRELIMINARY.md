# Preliminary result: MSD Task002 Heart

This is the second completed primary dataset, not the requested 10+ dataset benchmark. It must not be presented as
evidence of consistent cross-dataset improvement.

- 20 training cases evaluated strictly out-of-fold using the author-published five fold models.
- The archive's model-selection table was read before RankSEG evaluation: 2D `0.9096`, `3d_fullres` `0.9329`, and
  2D+3D ensemble `0.9268`. Therefore `3d_fullres` is the sole primary result.
- Official model MD5: `47918532eabe17478eb29335b90b6604`; dataset MD5:
  `06ee59366e1e5124267b774dbd654057`.
- RankSEG source commit: `00a1b87081d1d0cd1e8da8f8d1e6eb7a9c7dcf95`.
- Metric: foreground case mean; 10,000 paired case-bootstrap samples.

| Primary metric | Argmax | RankSEG | Delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|
| Mean Dice | 93.29433% | 93.29623% | +0.00191 | [-0.00414, +0.00824] |

All 20 cases remain stable within ±1 Dice pp. The positive mean is practically tiny and statistically compatible
with no difference.

## Input and resource checks

All 20 saved probability arrays were restored from nnU-Net's crop using the adjacent `crop_bbox` metadata before
decoding. After the same official postprocessing, probability argmax differed from native nnU-Net masks at 11 of
232,550,400 voxels (`4.73e-8`),
consistent with float16 storage and possible native postprocessing.

Median decoder time was 0.293 ms for argmax and 4.757 ms for RankSEG; shared connected-component postprocessing added
about 69 ms. Maximum allocated CUDA memory was 204 MiB and 675 MiB, respectively. These measurements exclude nnU-Net
inference.

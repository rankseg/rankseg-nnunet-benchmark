# Preliminary result: MSD Task004 Hippocampus

This is the first completed dataset, not the requested 10+ dataset benchmark. It must not be presented as evidence of
consistent cross-dataset improvement.

- 260 training cases evaluated strictly out-of-fold using the author-published five fold models.
- Model archive: [nnU-Net authors' Zenodo release](https://doi.org/10.5281/zenodo.4003545), MD5
  `44dad55102901e203f5cda68084f0c5c`.
- Dataset archive MD5: `9d24dba78a72977dbd1d2e110310f31b`.
- RankSEG source commit: `00a1b87081d1d0cd1e8da8f8d1e6eb7a9c7dcf95`.
- Metric: mean over the two foreground-label case means; 10,000 paired case-bootstrap samples.

| Configuration | Role | Argmax Dice | RankSEG Dice | Delta (percentage points) | 95% paired CI (pp) |
|---|---|---:|---:|---:|---:|
| 3d_fullres | Primary; official best configuration + postprocessing | 88.90847% | 88.91252% | +0.00405 | [-0.00253, +0.01083] |
| 2D | Sensitivity only | 86.90978% | 86.97973% | +0.06995 | [+0.01861, +0.14173] |
| 2D + 3D probability ensemble | Sensitivity only | 88.75775% | 88.82691% | +0.06917 | [+0.02110, +0.12279] |

The official release's model-selection table chooses `3d_fullres`, so that is the only Task004 result eligible for the
future cross-dataset aggregate. The average delta is positive but practically tiny and statistically compatible with
no difference. All 260 cases remain within ±1 Dice pp, so none is classified as materially improved or worsened.

For the primary run, argmax from the saved float16 probabilities differed from the native nnU-Net hard masks at only
17 of 16,326,256 voxels (`1.04e-6`) after applying the same official postprocessing. Native masks remain a diagnostic,
not a third benchmark method.

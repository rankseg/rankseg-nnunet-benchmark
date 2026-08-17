# MSD Pancreas preliminary result

Task007 is the seventh predeclared primary dataset and the first result in the locked prospective confirmation cohort.
Its OOF inference and probability ensemble were complete, but the argmax-versus-RankSEG result was unseen when the
confirmation cohort was frozen on 2026-07-28.

## Protocol

- 281 MSD Pancreas training cases, each predicted exactly once by the nnU-Net v1 fold that did not train on it.
- Default nnU-Net split: `KFold(5, shuffle=True, random_state=12345)`; fold sizes 57/56/56/56/56.
- Official-best configuration selected from archive metadata before RankSEG evaluation: the arithmetic mean of
  `3d_fullres` and `3d_cascade_fullres` softmax arrays, with official validation Dice 0.6821202.
- The official ensemble postprocessing selects no connected-component operations, so both decoders use their raw
  masks.
- Fixed RankSEG settings: Dice target, RMA solver, multiclass output, pruning probability 0.5, no smoothing, and
  `max_score` assignment for unassigned voxels.
- 10,000 paired case-bootstrap samples. Overall foreground mean Dice is primary.

The 5,002,701,016-byte model archive matched MD5 `f8983162ffb28664221e74bc63dc5c99`; the
12,289,971,712-byte dataset archive matched MD5 `4f7080cfca169fa8066d17ce6eb061e4`. All 281 component and ensembled
softmax arrays, labels, metadata files, and native masks were present. The benchmark argmax exactly matches every
native nnU-Net ensemble mask: zero mismatches across 7,004,225,536 valid voxels. Its 0.6839597 Dice is 0.18395
percentage points above the archive's historical validation score.

## Primary mean result

| Metric | Argmax | RankSEG | Mean delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|
| Dice | 68.39597% | 69.56287% | +1.16690 | [+0.70662, +1.64533] |

At the ±1 Dice pp threshold, 66 cases improve, 191 remain stable, and 24 worsen. The effect is label-dependent:
pancreas Dice changes from 82.1416% to 81.9160% (-0.2256 pp), while tumour Dice changes from 54.6504% to 57.2097%
(+2.5593 pp). The positive primary result is therefore driven by substantially better tumour segmentation and not
by uniform improvement of both targets.

The largest gain is `pancreas_409`, +23.5365 pp mean foreground Dice. Its tumour Dice rises from 2.7879% to 51.2319%
as tumour TP increases from 5,449 to 137,352, at the cost of additional FP. The worst regression is
`pancreas_305`, -11.0451 pp; `pancreas_120` also loses 10.3175 pp. Across all cases, 24/281 lose more than 1 pp, 6/281
lose more than 5 pp, and 2/281 lose more than 10 pp. The symmetric gain counts are 66, 29, and 13, respectively.

## Decoder resource cost

Argmax ran on CUDA for all 281 cases, with median 0.65 ms. RankSEG ran on CUDA for 280 cases, with median 78.37 ms.
The exceptionally large `pancreas_409` volume contains 196,870,144 voxels; the standard algorithm reached 19.62 GiB
allocated CUDA memory and exceeded the RTX 3090 capacity, so the identical float32 RankSEG computation was retried on
CPU and took 43.43 seconds. The output records this fallback explicitly. The maximum successful measured CUDA
allocation was 6.60 GiB for argmax and 14.34 GiB for RankSEG. These timings exclude nnU-Net inference and file I/O.

Machine-readable results are under `outputs/Task007_Pancreas_ensemble_oof/`.

# MSD Lung preliminary result

Task006 is the sixth predeclared primary dataset and the third dataset evaluated prospectively after freezing the
separate 2x class-volume safety track. The primary result remains the unguarded RankSEG decoder.

## Protocol

- 63 MSD Lung training cases, each predicted exactly once by the nnU-Net v1 fold that did not train on it.
- Default nnU-Net split: `KFold(5, shuffle=True, random_state=12345)`; fold sizes 13/13/13/12/12.
- This older model archive has no task-level `summary.csv`. Before any RankSEG evaluation, the registry therefore
  compared the final cross-validation Dice in all ten official per-configuration `postprocessing.json` files. The
  selected primary is the arithmetic mean of `3d_lowres` and `3d_fullres` softmax arrays (official Dice 0.7241146),
  ahead of `3d_fullres` alone (0.7210804) and `3d_lowres` alone (0.7109356).
- Full nnU-Net mirroring TTA for both component models. The two OOF assignment files are byte-identical.
- The official ensemble postprocessing selects no connected-component operations (`for_which_classes: []`), so raw
  ensemble masks are used for both decoders.
- Fixed RankSEG settings: Dice target, RMA solver, multiclass output, pruning probability 0.5, no smoothing, and
  `max_score` assignment for any unassigned voxel.
- 10,000 paired case-bootstrap samples. Overall mean Dice is primary; difficult-case summaries are secondary.

The downloaded model archive was 5,006,898,593 bytes and matched MD5
`9c7f89f3b2153292f69d6478da6a218f`; the 9,163,696,640-byte dataset archive matched MD5
`8afd997733c7fc0432f71255ba4e52dc`. The source tar contains 162 macOS resource-fork entries; these were isolated with
a clean symlink view before the official converter ran. All 63 labels contain only values 0 and 1, have nonempty
foreground, and match their image geometry.

All 63 low-resolution, full-resolution, and ensembled softmax arrays, metadata files, and native masks were present.
The evaluator's argmax exactly matched the native nnU-Net ensemble mask at all 4,628,676,608 voxels. An independent
run of nnU-Net v1's `nnUNet_evaluate_folder` gave Dice 0.7228821783, equal to the benchmark score at numerical
precision. The current result is 0.12324 percentage points below the archive's official 0.7241145550 validation Dice;
this inference-reproduction difference is recorded, but both decoders use exactly the same current probabilities.

## Primary mean result

| Metric | Argmax | RankSEG | Mean delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|
| Dice | 72.28822% | 72.74899% | +0.46077 | [-0.68225, +1.56630] |

At the ±1 Dice pp threshold, 20 cases improve, 33 remain stable, and 10 worsen. The largest gain is `lung_079`, from 64.71115% to
81.38736% Dice (+16.67621 pp): TP/FP/FN changes from 1,445/48/1,528 to 2,370/481/603. The largest regression is
`lung_003`, from 77.70925% to 61.50987% (-16.19938 pp): RankSEG adds one TP but increases FP from 758 to 1,657.
`lung_059` has a similar -16.09232 pp regression. The positive dataset mean therefore coexists with clinically
material bidirectional case effects.

## Prospective safety track

The frozen symmetric 2x foreground-volume guard triggered on 0/63 cases. It therefore neither suppressed the large
gains nor caught the `lung_003` and `lung_059` regressions; the guarded mean is numerically the same +0.46072 pp. In
combination with Colon, this confirms that a simple volume-ratio threshold is not a general safety mechanism.

## Decoder resource cost

On an RTX 3090, mean decoder time was 1.64 ms for argmax and 28.13 ms for RankSEG. The no-op postprocessing path took
about 107 ms for either method. Maximum measured CUDA allocation was 2.48 GiB for argmax and 8.23 GiB for RankSEG.
These timings exclude nnU-Net model inference and probability-file loading.

Machine-readable primary results are under `outputs/Task006_Lung_ensemble_oof/`; the prospective guard result is
under `outputs/Task006_Lung_guarded_prospective/`.

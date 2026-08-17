# Preliminary result: MSD Task005 Prostate

This is the third completed primary dataset, not the requested 10+ dataset benchmark. It contains a material RankSEG
regression that must be resolved or explicitly accepted before proposing a merge.

- 32 training cases evaluated strictly out-of-fold using both author-published 2D and `3d_fullres` fold models.
- The archive's model-selection table was read before RankSEG evaluation: 2D `0.7341`, 3D `0.7544`, and their
  probability ensemble `0.7592`. The ensemble is therefore the sole primary configuration.
- The archive's prelearned class-2 largest-connected-component postprocessing was applied identically to argmax and
  RankSEG in the primary pipeline result.
- Model MD5: `364cd721162977d7e7e37d59f41c0dc4`; dataset MD5:
  `35138f08b1efaef89d7424d2bcc928db`.

## Release-ground-truth audit

The public MSD mirror labels `prostate_18` and `prostate_32` as class 1 only, while the official model archive's
evaluation summary records exactly the same 45,166 and 25,688 reference voxels as class 2. A separate reproducible
label layer changes `1 -> 2` only for these two cases, records input/output SHA256 values, and leaves the downloaded
archive untouched. With those release-matched labels, regenerated raw argmax Dice is 75.9464%, within 0.0166 pp of
the archive's 75.9298% raw result. Regenerated hard masks also differ from the archive's masks at only a small number
of voxels.

## Primary pipeline result

| Primary metric | Argmax | RankSEG | Delta (pp) | 95% paired CI (pp) |
|---|---:|---:|---:|---:|
| Mean Dice | 75.95709% | 75.98004% | +0.02295 | [-2.58438, +1.99026] |

At the ±1 Dice pp threshold, 8 cases improve, 20 remain stable, and 4 worsen. The positive Dice mean is tiny relative
to its uncertainty. Raw decoding before official postprocessing was `+0.19116 pp Dice`, also with a wide interval
`[-2.02229, +1.96592] pp`.

## Material regression

`prostate_07` falls from 65.33% to 32.71% mean Dice (`-32.62 pp`). RankSEG creates a large false class-2 component;
the fixed official largest-component postprocessing retains that false component and removes the true one, yielding
0 true positives, 72,422 false positives, and 5,653 false negatives for class 2. Conversely, `prostate_21` improves by
16.78 pp, so the overall mean hides two very large opposing effects.

A volume/calibration safeguard that can fall back to argmax is a reasonable future algorithmic hypothesis, because
the failed class volume is obviously implausible. It was not predeclared and therefore must not be retroactively used
to replace this primary result; it should be fixed once and tested prospectively on later datasets.

### Post-hoc guard development result (not primary)

A symmetric 2x foreground-volume trust region was defined after observing the first three datasets: if any foreground
class changes by more than a factor of two relative to argmax after shared postprocessing, the entire case falls back
to argmax. Across the three development datasets, `prostate_07` is the only class/case above 2x (8.56x); all others
are at most 1.81x.

The guard triggers once and changes Prostate mean Dice delta to `+1.0423 pp`, with paired 95% CI
`[+0.2014, +2.2724] pp` (20 wins, 11 losses, one fallback tie). This is promising algorithm-development evidence,
but it is post-hoc and the aggregator explicitly rejects it. It must be frozen and tested prospectively on subsequent
unseen datasets before supporting a merge-safe variant.

## Input and resource checks

Seventeen of 32 ensemble probabilities required crop restoration. After identical postprocessing, probability argmax
differed from the archive's official postprocessed hard masks at 2,930 of 60,260,352 voxels (`4.86e-5`). Median raw
decoder time was 0.106 ms for argmax and 7.688 ms for RankSEG; median connected-component postprocessing was about
9--10 ms for both.

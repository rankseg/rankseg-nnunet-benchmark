# RankSEG × nnU-Net benchmark

This repository evaluates one focused question: **what changes when
[nnU-Net](https://github.com/MIC-DKFZ/nnUNet)'s usual voxel-wise argmax is replaced by
[RankSEG](https://github.com/rankseg/rankseg) at inference time?** The comparison uses the same exported model outputs
for both decoders and does not change training, checkpoints, preprocessing, or test data.

The public evidence package contains 16 datasets and 2,181 labeled cases. It is small enough to clone, checksummed,
and independently rebuildable without downloading images, model weights, or probability arrays.

## Experiment design

### Comparison

- **Baseline:** voxel-wise argmax.
- **Alternative:** RankSEG with its public default multiclass Dice configuration.
- **Input:** the identical post-softmax nnU-Net array for both decoders.
- **Postprocessing:** the configuration selected by the official model release, applied identically to both outputs.
- **Tuning:** no dataset-specific RankSEG tuning and no calibration fitted on evaluation cases.

Most datasets use official nnU-Net v1 checkpoints because they provide a broad, checksummed model release. A
TotalSegmentator checkpoint supplies an additional nnU-Net v2 inference-path replication.

### Cohort

Full-16 contains:

- a Core-12 cohort evaluated with strict five-fold out-of-fold (OOF) prediction from official nnU-Net v1 models;
- PROMISE12, evaluated on 30 independent former-hidden-test cases;
- CHAOS MRI, evaluated out of fold with uncertainty clustered by patient;
- Fluo-C3DH-A549, evaluated out of fold with uncertainty clustered by sequence;
- CHAOS CT, evaluated on 20 external cases with a TotalSegmentator nnU-Net v2 checkpoint.

The datasets span CT, MRI, cardiac cine MRI, tumour and organ segmentation, vessels, nested BraTS evaluation regions,
and microscopy. Exactly one model configuration was selected per dataset before the cross-dataset analysis. Every
eligible result was retained, including regressions.

### Outcomes

The primary outcome is the change in foreground macro mean Dice:

```text
Δ Dice = Dice(RankSEG) - Dice(argmax)
```

Each dataset is one unit in the Full-16 analysis, regardless of its number of cases. The benchmark also reports
paired bootstrap intervals, case-level changes, tail behavior, decoder runtime, and recorded peak memory. Repeated
observations from the same patient or sequence are clustered where required.

The complete rules are in the [fixed benchmark protocol](docs/BENCHMARK_PROTOCOL.md).

## Results

- **16 datasets and 2,181 cases**
- **13 positive and 3 negative** dataset-level Dice changes
- Macro-average Dice: **83.63646% argmax** vs **83.84318% RankSEG**
- Dataset-level mean change: **+0.20671 pp**; median change: **+0.00979 pp**
- Exact two-sided sign test: **p=0.02127**; Wilcoxon signed-rank test: **p=0.01099**
- At a ±1 Dice pp case threshold: **246 improved, 1,802 stable, 133 worsened**

| Dataset | Evaluation | Cases | Argmax Dice (%) | RankSEG Dice (%) | Δ (pp) |
|---|---|---:|---:|---:|---:|
| MSD BrainTumour | v1 OOF | 484 | 84.57217 | 84.57988 | +0.00771 |
| MSD Heart | v1 OOF | 20 | 93.29433 | 93.29623 | +0.00191 |
| MSD Liver | v1 OOF ensemble | 131 | 80.99903 | 81.82314 | +0.82412 |
| MSD Hippocampus | v1 OOF | 260 | 88.90847 | 88.91252 | +0.00405 |
| MSD Prostate | v1 OOF ensemble | 32 | 75.95709 | 75.98004 | +0.02295 |
| MSD Lung | v1 OOF ensemble | 63 | 72.28822 | 72.74899 | +0.46077 |
| MSD Pancreas | v1 OOF ensemble | 281 | 68.39597 | 69.56287 | +1.16690 |
| MSD HepaticVessel | v1 OOF ensemble | 303 | 68.63054 | 69.16535 | +0.53481 |
| MSD Spleen | v1 OOF ensemble | 41 | 97.23659 | 97.24320 | +0.00661 |
| MSD Colon | v1 OOF | 126 | 48.91759 | 49.15961 | +0.24203 |
| PROMISE12 | v1 independent test ensemble | 30 | 91.93567 | 91.85350 | -0.08217 |
| ACDC | v1 OOF ensemble; patient-clustered | 200 | 92.45381 | 92.38008 | -0.07373 |
| CHAOS MRI | v1 OOF ensemble; patient-clustered | 60 | 92.04389 | 92.21990 | +0.17601 |
| SegTHOR | v1 OOF ensemble | 40 | 91.51999 | 91.51811 | -0.00188 |
| Fluo-C3DH-A549 | v1 OOF; sequence-clustered | 90 | 93.94401 | 93.95587 | +0.01186 |
| CHAOS CT | v2 external test | 20 | 97.08608 | 97.09150 | +0.00542 |

The three negative dataset-level results are retained: PROMISE12 (-0.08217 pp), ACDC (-0.07373 pp), and SegTHOR
(-0.00188 pp). Three datasets have paired intervals entirely above zero: Liver, Pancreas, and HepaticVessel.

See the [generated Full-16 report](evidence/full16/RESULTS.md) for model configurations and per-dataset case outcome
counts, or [browse the machine-readable evidence](evidence/).

## Conclusion

On standard official nnU-Net outputs, RankSEG produced a positive average Dice change and more dataset-level gains
than losses. The result supports RankSEG as an **optional inference decoder** that can be evaluated without retraining
the underlying model.

It does not support replacing argmax unconditionally. The median gain is small, three datasets decline, and some
individual cases have material regressions. The defensible claim is therefore positive average benefit with explicit
downside reporting—not guaranteed improvement for every dataset or patient.

## Notes

### Probability calibration

The official v1 checkpoints used in most of Full-16 were trained with nnU-Net's default cross-entropy plus soft-Dice
objective. Their exported arrays are valid softmax-normalized scores, but normalization alone does not establish that
they are calibrated estimates of conditional class probabilities. Because RankSEG treats its inputs as probabilities,
calibration may affect the observed decoder difference.

Full-16 measures the practical plug-in effect on standard official outputs. It does not claim that those outputs are
calibrated or that the result is independent of the training loss. Overconfidence is a hypothesis for a separate,
matched CE-only versus CE + soft-Dice study; it is not inferred from the loss alone. See the
[calibration protocol](docs/BENCHMARK_PROTOCOL.md#probability-interpretation-and-calibration).

### Scope and limitations

- Most evidence comes from official nnU-Net v1 models; the v2 result is an implementation-path replication, not an
  equally broad v2 benchmark.
- BrainTumour uses mutually exclusive softmax channels followed by nested-region metric evaluation. Full-16 does not
  validate RankSEG on models trained with overlapping sigmoid region channels.
- No Full-16 dataset uses an ignore label.
- Missing legacy timing or memory measurements are stored as `null`; the schema migration does not invent them.
- Rejected post-hoc variants and exclusions remain visible in
  [Rejected experiments](docs/REJECTED_EXPERIMENTS.md).

### Verify the published evidence

```bash
python -m venv env
env/bin/python -m pip install -r requirements.txt
env/bin/python -m pip install --no-build-isolation -e .
env/bin/rankseg-nnunet-bench verify-evidence evidence
```

Verification checks every published SHA256 and schema, validates all case counts, rebuilds the Full-16 aggregate in a
temporary directory, and requires byte-identical aggregate outputs. `requirements-lock.txt` records the complete
Linux/Python 3.10/CUDA 12.8 reference environment; `requirements.txt` is the portable installation input.

### Reproduce or extend the benchmark

- [Dataset manifests](configs/) define inputs, decoder settings, evaluation design, and provenance.
- [Model and dataset registries](registry/) record publisher URLs, sizes, and checksums.
- [Script guide](scripts/README.md) documents the supported download, conversion, OOF inference, audit, and aggregate
  entry points.
- [Dataset plan](docs/DATASET_PLAN.md) records cohort construction and completion status.
- [Results to date](results/RESULTS_TO_DATE.md) provides the detailed scientific narrative and regression audit.

Large images, checkpoints, probability arrays, and local outputs are intentionally excluded from Git. The compact
`evidence/` directory contains the published summaries, paired case deltas, available audits, and rebuilt aggregate.

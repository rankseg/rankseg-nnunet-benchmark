# RankSEG × nnU-Net benchmark

This repository builds the evidence requested by the nnU-Net maintainer for an optional RankSEG decoder: paired,
leakage-free comparisons on more than ten diverse datasets, including regressions and resource cost rather than only
reporting average gains.

The benchmark consumes the probability arrays already exported by nnU-Net (`probabilities` in v2, `softmax` in v1).
Argmax and RankSEG therefore see exactly the same model output; no checkpoint, training code, or preprocessing is
changed. When v1 stores probabilities only inside its nonzero crop, the evaluator reads the adjacent restricted
`crop_bbox` metadata and restores background probability outside the crop before either decoder runs.

## Current status

- The complete model registry contains all 22 nnU-Net v1 task archives released by the nnU-Net authors on
  [Zenodo](https://doi.org/10.5281/zenodo.4003545), with byte sizes and MD5 checksums.
- nnU-Net's current documentation still says that a central pretrained-model release is not available for v2. The
  broad study therefore uses official v1 models, with v2 replications such as TotalSegmentator to verify the current
  inference path.
- The main result is the complete Full-16 benchmark: 16 datasets, 2,181 labeled cases, 13 positive and 3 negative
  dataset-level Dice changes. The macro-average across the 16 datasets (each dataset weighted equally) is 83.63646%
  for argmax and 83.84318% for RankSEG, a mean improvement of +0.20671 pp; the overall exact sign-test p=0.02127
  and Wilcoxon p=0.01099. Only these full-benchmark p-values are reported.
- Full-16 comprises a Core-12 official nnU-Net v1 OOF subset plus four robustness cohorts: Task024 PROMISE12
  (30 independent former-hidden-test cases, -0.08217 pp), Task038 CHAOS MRI (60 patient-clustered OOF volumes,
  +0.17601 pp), Task075 Fluo-C3DH-A549 (90 sequence-clustered OOF cases, +0.01186 pp), and CHAOS-CT evaluated with
  TotalSegmentator nnU-Net v2 (20 external cases, +0.00542 pp).
- Core-12 remains a sensitivity analysis documenting the original official-v1 OOF sequence; it is not a competing
  main benchmark. Its 1,981 cases split 10 positive and 2 negative datasets.
- Three individual datasets have paired intervals entirely above zero: Liver (+0.82412 pp), Pancreas (+1.16690 pp),
  and HepaticVessel (+0.53481 pp). ACDC declines by 0.07373 pp using a patient-clustered analysis; SegTHOR declines
  by a negligible 0.00188 pp. All results are retained regardless of direction.
- The 16 included datasets span CT, single- and multisequence MRI, tumours, organs, vessels, thoracic structures,
  BraTS nested regions, cardiac cine MRI, and microscopy. Ten configurations are probability ensembles and six are
  single models.
- At a practical threshold of ±1 Dice pp, 246 of 2,181 cases improve, 1,802 remain stable, and 133 worsen; changes
  beyond 10 Dice pp include 9 losses and 31 gains. This supports RankSEG as an optional decoder with positive average
  benefit, not as a guaranteed monotonic replacement for argmax.
- Rejected development paths (the post-hoc volume guard, Dice-loss inversion, slice-wise decoding, and ineligible
  datasets) are summarized separately and are not part of the maintained benchmark implementation.

## Full-16 benchmark results

| Task | Dataset | Model | Cases | Argmax mean Dice (%) | RankSEG mean Dice (%) | Dice Δ (pp) | Improved (>+1 pp) | Stable (within ±1 pp) | Worsened (<-1 pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CHAOS | CHAOS CT | nnU-Net v2 · TotalSegmentator Dataset291 · 3d_fullres (fold 0) | 20 | 97.08608 | 97.09150 | +0.00542 | 0 | 20 | 0 |
| Task001 | MSD BrainTumour | nnU-Net v1 · 3d_fullres | 484 | 84.57217 | 84.57988 | +0.00771 | 7 | 476 | 1 |
| Task002 | MSD Heart | nnU-Net v1 · 3d_fullres | 20 | 93.29433 | 93.29623 | +0.00191 | 0 | 20 | 0 |
| Task003 | MSD Liver | nnU-Net v1 · 3d_lowres + 3d_fullres ensemble | 131 | 80.99903 | 81.82314 | +0.82412 | 32 | 90 | 9 |
| Task004 | MSD Hippocampus | nnU-Net v1 · 3d_fullres | 260 | 88.90847 | 88.91252 | +0.00405 | 0 | 260 | 0 |
| Task005 | MSD Prostate | nnU-Net v1 · 2d + 3d_fullres ensemble | 32 | 75.95709 | 75.98004 | +0.02295 | 8 | 20 | 4 |
| Task006 | MSD Lung | nnU-Net v1 · 3d_lowres + 3d_fullres ensemble | 63 | 72.28822 | 72.74899 | +0.46077 | 20 | 33 | 10 |
| Task007 | MSD Pancreas | nnU-Net v1 · 3d_fullres + 3d_cascade_fullres ensemble | 281 | 68.39597 | 69.56287 | +1.16690 | 66 | 191 | 24 |
| Task008 | MSD HepaticVessel | nnU-Net v1 · 3d_lowres + 3d_fullres ensemble | 303 | 68.63054 | 69.16535 | +0.53481 | 85 | 169 | 49 |
| Task009 | MSD Spleen | nnU-Net v1 · 3d_fullres + 3d_cascade_fullres ensemble | 41 | 97.23659 | 97.24320 | +0.00661 | 0 | 41 | 0 |
| Task010 | MSD Colon | nnU-Net v1 · 3d_cascade_fullres (3d_lowres input) | 126 | 48.91759 | 49.15961 | +0.24203 | 14 | 90 | 22 |
| Task024 | PROMISE12 former hidden test | nnU-Net v1 · 2d + 3d_fullres ensemble | 30 | 91.93567 | 91.85350 | -0.08217 | 0 | 29 | 1 |
| Task027 | ACDC | nnU-Net v1 · 2d + 3d_fullres ensemble | 200 | 92.45381 | 92.38008 | -0.07373 | 7 | 181 | 12 |
| Task038 | CHAOS MRI | nnU-Net v1 · 2d + 3d_fullres ensemble | 60 | 92.04389 | 92.21990 | +0.17601 | 7 | 52 | 1 |
| Task055 | SegTHOR | nnU-Net v1 · 3d_fullres + 3d_cascade_fullres ensemble | 40 | 91.51999 | 91.51811 | -0.00188 | 0 | 40 | 0 |
| Task075 | CTC Fluo-C3DH-A549 manual + simulated | nnU-Net v1 · 3d_fullres | 90 | 93.94401 | 93.95587 | +0.01186 | 0 | 90 | 0 |

See the checked-in [Full-16 report](evidence/full16/RESULTS.md), [results to date](results/RESULTS_TO_DATE.md),
the detailed [Hippocampus](results/PRELIMINARY.md),
[Heart](results/HEART_PRELIMINARY.md), [Prostate](results/PROSTATE_PRELIMINARY.md),
[Spleen](results/SPLEEN_PRELIMINARY.md), [Colon](results/COLON_PRELIMINARY.md), and
[Lung](results/LUNG_PRELIMINARY.md), [Pancreas](results/PANCREAS_PRELIMINARY.md) reports,
[the CHAOS MRI evidence](evidence/datasets/Task038_CHAOS/summary.json),
[the fixed protocol](docs/BENCHMARK_PROTOCOL.md), and the
[dataset execution plan](docs/DATASET_PLAN.md). See [rejected experiments](docs/REJECTED_EXPERIMENTS.md) for removed
development branches and exclusion reasons.

## Project layout

- `src/rankseg_nnunet_bench/`: the maintained evaluator, metrics, I/O, OOF, registry, and aggregate implementation;
- `configs/`: one manifest per reported dataset/configuration plus clearly named sensitivity manifests;
- `scripts/`: reproducible dataset pipelines, conversion/audit utilities, and the single Full-16 aggregate builder;
- `registry/`: checksummed official model and dataset sources;
- `tests/`: unit and synthetic end-to-end coverage of the maintained benchmark path;
- `evidence/`: compact, checksummed Full-16 summaries, paired case deltas, audits, and rebuilt aggregate;
- `outputs/`, `work/`, `data/`, and `artifacts/`: ignored local inputs and large generated artifacts;
- `docs/`: the fixed protocol, dataset plan, and rejected-experiment record.

See [`scripts/README.md`](scripts/README.md) for the supported script entry points.

## Install

The reference machine uses Python 3.10, PyTorch 2.8.0 + CUDA 12.8, nnU-Net 2.6.2, nnU-Net v1.7.1, and an RTX 3090.

```bash
python -m venv env
env/bin/python -m pip install -r requirements.txt
env/bin/python -m pip install nnunet==1.7.1
env/bin/python -m pip install --no-build-isolation -e .
```

`requirements.txt` installs the published `rankseg==0.0.5` package directly from PyPI. Each manifest also records the
RankSEG release/source revision used to produce its cached result. `requirements-lock.txt` is the complete,
platform-specific snapshot of the Linux/Python 3.10/CUDA 12.8 reference environment; use `requirements.txt` for a
portable install and the lock when reproducing that exact environment.

## Verify the published evidence

The checked-in package can be verified without images, checkpoints, or cached probability arrays:

```bash
rankseg-nnunet-bench verify-evidence evidence
```

This checks every published SHA256, validates all normalized schema-v3 summaries and case counts, then rebuilds the
Full-16 aggregate in a temporary directory and requires byte-identical aggregate outputs. See
[`evidence/README.md`](evidence/README.md) for the package boundary.

## Evaluate cached probabilities

Create a manifest from `configs/example_dataset.yaml`, then run:

```bash
rankseg-nnunet-bench evaluate configs/my_dataset.yaml
```

Each dataset produces:

- `case_label_metrics.csv`: TP, FP, FN, and Dice for every case, label, and decoder;
- `label_summary.csv`, `case_summary.csv`, and paired per-case deltas;
- `timings.csv` and peak CUDA allocation;
- `input_diagnostics.csv`, including probability normalization checks;
- `summary.json`, including paired case-bootstrap confidence intervals, case outcome counts, quantile diagnostics,
  and full provenance.

Limited runs (`--case-limit`) are marked as smoke tests and the aggregator refuses to include them.

## Leakage-free inference with official v1 models

List or download the author-published models:

```bash
rankseg-nnunet-bench inventory registry/nnunet_v1_official_models.yaml
rankseg-nnunet-bench inspect-selections registry/nnunet_v1_official_models.yaml \
  --output results/OFFICIAL_MODEL_SELECTION.csv \
  --tasks Task002_Heart Task004_Hippocampus Task005_Prostate
rankseg-nnunet-bench download-model \
  registry/nnunet_v1_official_models.yaml Task004_Hippocampus \
  --output-dir artifacts/models
```

`inspect-selections` uses HTTP byte-range reads to retrieve only each ZIP directory and official selection metadata,
not the checkpoints. It prioritizes the task-level `summary.csv`; for older archives without that file, it compares
the final cross-validation Dice stored in every official `postprocessing.json` and marks the result as a fallback.
This locks each dataset configuration before any RankSEG result is observed without guessing from RankSEG outcomes.

After installing a model and converting its dataset to nnU-Net v1 raw format, prepare fold-specific inputs:

```bash
rankseg-nnunet-bench prepare-oof-v1 \
  --task Task004_Hippocampus \
  --images-dir "$nnUNet_raw_data_base/nnUNet_raw_data/Task004_Hippocampus/imagesTr" \
  --output-dir work/v1/oof/Task004_Hippocampus \
  --model 3d_fullres --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1

bash work/v1/oof/Task004_Hippocampus/run_oof_inference.sh
rankseg-nnunet-bench evaluate configs/Task004_Hippocampus_oof.yaml
```

`prepare-oof-v1` reproduces the official default `KFold(5, shuffle=True, random_state=12345)`. Supply the exact
`--splits-file` for any task with a custom split. Every case must occur in validation exactly once.

## Aggregate datasets

Only one predeclared configuration per dataset belongs in the Full-16 cross-dataset analysis:

```bash
bash scripts/build_full_aggregate.sh
```

The aggregate treats the dataset—not the case or label—as the statistical unit and reports positive/tied/negative
datasets, argmax and RankSEG mean Dice, median and mean deltas, case improved/stable/worsened counts at ±1 pp, the
exact sign test, and Wilcoxon signed-rank test. Overall mean Dice is the main endpoint; quantiles are retained as
diagnostics.

Maintainers with the ignored local results can produce a fresh publication candidate without overwriting the
checked-in evidence:

```bash
rankseg-nnunet-bench publish-evidence configs/full16_evidence.yaml \
  --output-dir evidence-new
rankseg-nnunet-bench verify-evidence evidence-new
```

# Published Full-16 evidence

This directory is the small, version-controlled evidence package for the RankSEG × nnU-Net benchmark. It contains
no images, labels, model weights, probability arrays, or restricted Task035 artifacts.

- `16` dataset summaries use summary schema v3.
- `2181` paired case-level Dice rows are retained for regression auditing.
- `full16/` is rebuilt from the dataset summaries and case deltas.
- `MANIFEST.json` records the SHA256 and byte size of every published file.

Verify the checked-in package without access to the large local benchmark workspace:

```bash
python -m rankseg_nnunet_bench.cli verify-evidence evidence
```

Maintainers with the ignored local `outputs/` workspace can regenerate it from `configs/full16_evidence.yaml`:

```bash
python -m rankseg_nnunet_bench.cli publish-evidence configs/full16_evidence.yaml \
  --output-dir evidence-new
```

Legacy measurements that predate a timing column are represented as JSON `null`; the migration never infers a
device or resource measurement that was not recorded.

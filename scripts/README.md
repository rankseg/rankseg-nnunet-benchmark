# Scripts

The maintained scripts are grouped by purpose:

- `build_full_aggregate.sh`: rebuild the canonical Full-16 result from exactly one fixed summary per dataset.
- `run_*_pipeline.sh`: reproduce dataset conversion, inference, evaluation, and dataset-specific audit steps.
- `prepare_*.py`: deterministic conversion of an eligible public dataset into the layout required by its pipeline.
- `audit_*.py`: independently recompute integrity checks for external or nonstandard cohorts.
- `render_case_comparison.py`: render a selected argmax/RankSEG regression or improvement from cached probabilities.

Every shell entry point resolves the repository root from its own location, so the checkout can be moved or cloned
to another absolute path. The scripts still expect a local `env/` virtual environment and the ignored data/model
artifacts described by their corresponding manifests.

Per-dataset pipelines do not construct partial cross-dataset aggregates. After updating any dataset result, run
`build_full_aggregate.sh` once. To create and verify the compact publication candidate, run:

```bash
rankseg-nnunet-bench publish-evidence configs/full16_evidence.yaml \
  --output-dir evidence-new
rankseg-nnunet-bench verify-evidence evidence-new
```

Rejected experimental decoders are documented in
[`docs/REJECTED_EXPERIMENTS.md`](../docs/REJECTED_EXPERIMENTS.md) and intentionally have no executable entry point.

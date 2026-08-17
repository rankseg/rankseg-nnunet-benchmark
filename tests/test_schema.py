import pandas as pd

from rankseg_nnunet_bench.schema import SUMMARY_SCHEMA_VERSION, migrate_summary


def _legacy_summary() -> dict:
    return {
        "schema_version": 2,
        "dataset": {
            "id": "Dataset999_Test",
            "display_name": "Test",
            "cases": 1,
            "labels": {"0": "background", "1": "foreground"},
            "channel_labels": [0, 1],
            "foreground_labels": [1],
        },
        "selection": {"scientific_run": True},
        "decoder": {"argmax": {}, "rankseg": {}},
        "metrics": {
            "macro": {},
            "delta": {},
            "paired_case_bootstrap_delta": {},
            "case_quantiles": {},
        },
        "timing": {
            method: {
                "median_milliseconds": 1.0,
                "mean_milliseconds": 1.0,
                "median_postprocessing_milliseconds": 0.0,
                "mean_postprocessing_milliseconds": 0.0,
                "max_peak_memory_bytes": None,
            }
            for method in ("argmax", "rankseg")
        },
        "provenance": {},
    }


def test_migration_marks_unrecoverable_legacy_timing_as_missing():
    migrated = migrate_summary(_legacy_summary())

    assert migrated["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert migrated["metrics"]["paired_cluster_bootstrap_delta"] is None
    assert migrated["timing"]["rankseg"]["device_counts"] is None
    assert migrated["timing"]["rankseg"]["measurement_complete"] is False
    assert migrated["timing"]["rankseg"]["max_failed_peak_memory_bytes"] is None


def test_migration_recovers_recorded_device_counts_from_timing_rows():
    timings = pd.DataFrame(
        [
            {
                "method": "argmax",
                "milliseconds": 1.0,
                "peak_memory_bytes": 10,
                "device": "cuda",
            },
            {
                "method": "rankseg",
                "milliseconds": 2.0,
                "peak_memory_bytes": None,
                "device": "cpu_fallback_after_cuda_oom",
            },
        ]
    )
    migrated = migrate_summary(_legacy_summary(), timings=timings)

    assert migrated["timing"]["argmax"]["device_counts"] == {"cuda": 1}
    assert migrated["timing"]["rankseg"]["device_counts"] == {
        "cpu_fallback_after_cuda_oom": 1
    }
    assert migrated["timing"]["rankseg"]["measurement_complete"] is True

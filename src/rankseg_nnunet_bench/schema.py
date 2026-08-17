from __future__ import annotations

from copy import deepcopy
from typing import Any

import pandas as pd


SUMMARY_SCHEMA_VERSION = 3
AGGREGATE_SCHEMA_VERSION = 3

_TIMING_FIELDS = (
    "median_milliseconds",
    "mean_milliseconds",
    "median_postprocessing_milliseconds",
    "mean_postprocessing_milliseconds",
    "max_peak_memory_bytes",
    "device_counts",
    "mean_failed_attempt_milliseconds",
    "max_failed_peak_memory_bytes",
)


def _timing_values_from_frame(
    timings: pd.DataFrame | None,
    method: str,
) -> dict[str, Any]:
    if timings is None or not {"method", "milliseconds"} <= set(timings.columns):
        return {}
    subset = timings[timings["method"] == method]
    if subset.empty:
        return {}

    values: dict[str, Any] = {
        "median_milliseconds": float(subset["milliseconds"].median()),
        "mean_milliseconds": float(subset["milliseconds"].mean()),
    }
    if "postprocessing_milliseconds" in subset:
        values.update(
            {
                "median_postprocessing_milliseconds": float(
                    subset["postprocessing_milliseconds"].median()
                ),
                "mean_postprocessing_milliseconds": float(
                    subset["postprocessing_milliseconds"].mean()
                ),
            }
        )
    if "peak_memory_bytes" in subset:
        peaks = subset["peak_memory_bytes"].dropna()
        values["max_peak_memory_bytes"] = int(peaks.max()) if len(peaks) else None
    if "device" in subset:
        values["device_counts"] = {
            str(device): int(count)
            for device, count in subset["device"].value_counts().sort_index().items()
        }
    if "failed_attempt_milliseconds" in subset:
        failed = subset["failed_attempt_milliseconds"].dropna()
        values["mean_failed_attempt_milliseconds"] = (
            float(failed.mean()) if len(failed) else None
        )
    if "failed_peak_memory_bytes" in subset:
        failed_peaks = subset["failed_peak_memory_bytes"].dropna()
        values["max_failed_peak_memory_bytes"] = (
            int(failed_peaks.max()) if len(failed_peaks) else None
        )
    return values


def migrate_summary(
    summary: dict[str, Any],
    *,
    timings: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Return a schema-v3 copy without inventing unavailable legacy measurements."""
    migrated = deepcopy(summary)
    version = int(migrated.get("schema_version", 1))
    if version > SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"Summary schema {version} is newer than supported schema {SUMMARY_SCHEMA_VERSION}"
        )

    dataset = migrated.setdefault("dataset", {})
    dataset.setdefault("regions", None)
    dataset.setdefault("metric_semantics", None)
    dataset.setdefault("clusters", None)
    dataset.setdefault("cluster_regex", None)
    dataset.setdefault("segmentation_reader", None)

    metrics = migrated.setdefault("metrics", {})
    metrics.setdefault("paired_cluster_bootstrap_delta", None)

    timing = migrated.setdefault("timing", {})
    for method in ("argmax", "rankseg"):
        method_timing = timing.setdefault(method, {})
        recovered = _timing_values_from_frame(timings, method)
        for field in _TIMING_FIELDS:
            if field not in method_timing:
                method_timing[field] = recovered.get(field)
        method_timing["measurement_complete"] = (
            method_timing.get("median_milliseconds") is not None
            and method_timing.get("mean_milliseconds") is not None
            and method_timing.get("device_counts") is not None
        )

    migrated["schema_version"] = SUMMARY_SCHEMA_VERSION
    validate_summary(migrated)
    return migrated


def validate_summary(summary: dict[str, Any]) -> None:
    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        raise ValueError(
            f"Expected summary schema {SUMMARY_SCHEMA_VERSION}, got {summary.get('schema_version')!r}"
        )
    for key in ("dataset", "selection", "decoder", "metrics", "timing", "provenance"):
        if key not in summary:
            raise ValueError(f"Summary is missing required top-level key: {key}")
    dataset = summary["dataset"]
    for key in ("id", "display_name", "cases", "labels", "channel_labels", "foreground_labels"):
        if key not in dataset:
            raise ValueError(f"Summary dataset is missing required key: {key}")
    if not summary["selection"].get("scientific_run", False):
        raise ValueError("Published evidence must come from a complete scientific run")
    for method in ("argmax", "rankseg"):
        if method not in summary["timing"]:
            raise ValueError(f"Summary timing is missing method: {method}")
        missing = set(_TIMING_FIELDS) - set(summary["timing"][method])
        if missing:
            raise ValueError(f"Summary timing for {method} is missing fields: {sorted(missing)}")
    metrics = summary["metrics"]
    for key in ("macro", "delta", "paired_case_bootstrap_delta", "case_quantiles"):
        if key not in metrics:
            raise ValueError(f"Summary metrics is missing required key: {key}")

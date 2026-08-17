from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import DatasetConfig
from .decoders import PairedDecoders
from .io import (
    diagnostics_to_dict,
    discover_cases,
    load_probabilities,
    load_segmentation,
    load_segmentation_with_voxel_volume,
    restore_cropped_probabilities,
    validate_case,
)
from .metrics import overlap_metrics, overlap_region_metrics, summarize_metrics
from .postprocessing import OfficialNnUNetPostprocessor
from .provenance import runtime_provenance
from .schema import SUMMARY_SCHEMA_VERSION


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _case_cluster_ids(case_ids: list[str], pattern: str | None) -> dict[str, str] | None:
    if pattern is None:
        return None
    expression = re.compile(pattern)
    assignments: dict[str, str] = {}
    for case_id in case_ids:
        match = expression.search(case_id)
        if match is None:
            raise ValueError(f"evaluation.cluster_regex does not match case ID: {case_id}")
        assignments[case_id] = match.group(1) if match.lastindex else match.group(0)
    if len(set(assignments.values())) == len(assignments):
        raise ValueError(
            "evaluation.cluster_regex produced one unique cluster per case; omit it unless cases share subjects"
        )
    return assignments


def _validate_decoder_prediction(
    prediction: np.ndarray,
    *,
    method: str,
    spatial_shape: tuple[int, ...],
    number_of_channels: int,
) -> None:
    if tuple(prediction.shape) != spatial_shape:
        raise ValueError(
            f"{method} returned shape {prediction.shape}; expected spatial shape {spatial_shape}"
        )
    if not np.issubdtype(prediction.dtype, np.integer):
        raise TypeError(f"{method} must return integer channel indices, got {prediction.dtype}")
    minimum = int(prediction.min())
    maximum = int(prediction.max())
    if minimum < 0 or maximum >= number_of_channels:
        raise ValueError(
            f"{method} returned channel indices outside [0, {number_of_channels - 1}]: "
            f"min={minimum}, max={maximum}"
        )


def evaluate_dataset(
    config: DatasetConfig,
    *,
    device_override: str | None = None,
    case_limit: int | None = None,
    output_dir_override: Path | None = None,
) -> Path:
    if device_override is not None:
        config = replace(config, device=device_override)
    if output_dir_override is not None:
        config = replace(config, output_dir=output_dir_override.resolve())
    cases = discover_cases(config)
    if case_limit is not None:
        if case_limit < 1:
            raise ValueError("case_limit must be positive")
        cases = cases[:case_limit]
    cluster_ids = _case_cluster_ids([case.case_id for case in cases], config.cluster_regex)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    decoders = PairedDecoders(config)
    postprocessor = (
        OfficialNnUNetPostprocessor(config.postprocessing_file)
        if config.postprocessing_file is not None
        else None
    )
    metric_rows: list[dict] = []
    timing_rows: list[dict] = []
    diagnostic_rows: list[dict] = []
    native_rows: list[dict] = []
    all_labels = config.channel_labels

    for case_index, case in enumerate(cases, start=1):
        print(f"[{case_index}/{len(cases)}] {case.case_id}", flush=True)
        probabilities, probability_key = load_probabilities(case.probabilities, config.probability_key)
        target, volume_per_voxel = load_segmentation_with_voxel_volume(
            case.label,
            reader=config.segmentation_reader,
        )
        if 0 not in config.channel_labels:
            raise ValueError("nnU-Net multiclass probabilities must include background label 0")
        probabilities, restoration = restore_cropped_probabilities(
            probabilities,
            probability_path=case.probabilities,
            target_shape=tuple(int(value) for value in target.shape),
            background_channel=config.channel_labels.index(0),
        )
        diagnostics = validate_case(
            probabilities,
            target,
            channel_labels=config.channel_labels,
            ignore_label=config.ignore_label,
            probability_key=probability_key,
        )
        diagnostic_rows.append(
            {"case_id": case.case_id, **restoration, **diagnostics_to_dict(diagnostics)}
        )

        tensor = decoders.move_probabilities(probabilities)
        argmax = decoders.decode_argmax(tensor)
        rankseg = None
        failed_peak = None
        failed_attempt_milliseconds = None
        rankseg_attempt_start = perf_counter()
        try:
            rankseg = decoders.decode_rankseg(tensor)
        except torch.OutOfMemoryError:
            if tensor.device.type != "cuda":
                raise
            failed_attempt_milliseconds = (perf_counter() - rankseg_attempt_start) * 1000.0
            failed_peak = int(torch.cuda.max_memory_allocated(tensor.device))
        if rankseg is None:
            del tensor
            torch.cuda.empty_cache()
            print(
                f"[{case_index}/{len(cases)}] {case.case_id}: CUDA OOM at "
                f"{failed_peak / 2**30:.2f} GiB; retrying RankSEG on CPU",
                flush=True,
            )
            rankseg = decoders.decode_rankseg_cpu(probabilities)
            tensor = None
        _validate_decoder_prediction(
            argmax.prediction,
            method="argmax",
            spatial_shape=tuple(int(value) for value in target.shape),
            number_of_channels=len(config.channel_labels),
        )
        _validate_decoder_prediction(
            rankseg.prediction,
            method="rankseg",
            spatial_shape=tuple(int(value) for value in target.shape),
            number_of_channels=len(config.channel_labels),
        )
        channel_labels = np.asarray(config.channel_labels, dtype=np.int64)
        predictions = {
            "argmax": channel_labels[argmax.prediction],
            "rankseg": channel_labels[rankseg.prediction],
        }
        raw_argmax_prediction = predictions["argmax"].copy()
        postprocessing_milliseconds = {"argmax": 0.0, "rankseg": 0.0}
        if postprocessor is not None:
            for method, prediction in predictions.items():
                start = perf_counter()
                predictions[method] = postprocessor(
                    prediction,
                    volume_per_voxel=volume_per_voxel,
                )
                postprocessing_milliseconds[method] = (perf_counter() - start) * 1000.0
        results = {"argmax": argmax, "rankseg": rankseg}
        for method, prediction in predictions.items():
            metric_rows_for_case = (
                overlap_region_metrics(
                    prediction,
                    target,
                    config.regions,
                    ignore_label=config.ignore_label,
                )
                if config.regions is not None
                else overlap_metrics(
                    prediction,
                    target,
                    all_labels,
                    ignore_label=config.ignore_label,
                )
            )
            for row in metric_rows_for_case:
                metric_rows.append(
                    {
                        "dataset_id": config.dataset_id,
                        "case_id": case.case_id,
                        "method": method,
                        "label_name": config.labels[int(row["label"])],
                        **row,
                    }
                )
            result = results[method]
            timing_rows.append(
                {
                    "case_id": case.case_id,
                    "method": method,
                    "milliseconds": result.milliseconds,
                    "postprocessing_milliseconds": postprocessing_milliseconds[method],
                    "peak_memory_bytes": result.peak_memory_bytes,
                    "device": result.device,
                    "voxels": int(target.size),
                    "classes": int(probabilities.shape[0]),
                    "failed_attempt_milliseconds": (
                        failed_attempt_milliseconds if method == "rankseg" else None
                    ),
                    "failed_peak_memory_bytes": failed_peak if method == "rankseg" else None,
                }
            )

        if case.native_prediction is not None:
            native = load_segmentation(
                case.native_prediction,
                reader=config.segmentation_reader,
            )
            if native.shape != target.shape:
                raise ValueError(f"Native prediction shape mismatch for {case.case_id}: {native.shape} vs {target.shape}")
            valid = np.ones(target.shape, dtype=bool) if config.ignore_label is None else target != config.ignore_label
            native_rows.append(
                {
                    "case_id": case.case_id,
                    "argmax_native_mismatch_voxels": int(
                        np.count_nonzero(
                            (
                                (
                                    predictions["argmax"]
                                    if config.native_predictions_postprocessed
                                    else raw_argmax_prediction
                                )
                                != native
                            )
                            & valid
                        )
                    ),
                    "valid_voxels": int(np.count_nonzero(valid)),
                }
            )
        del tensor, probabilities

    metrics = pd.DataFrame(metric_rows)
    timings = pd.DataFrame(timing_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    metrics.to_csv(config.output_dir / "case_label_metrics.csv", index=False)
    timings.to_csv(config.output_dir / "timings.csv", index=False)
    diagnostics.to_csv(config.output_dir / "input_diagnostics.csv", index=False)
    if native_rows:
        native_frame = pd.DataFrame(native_rows)
        native_frame.to_csv(config.output_dir / "native_argmax_check.csv", index=False)
    else:
        native_frame = None
    label_summary, case_summary, case_paired_deltas, metric_summary = summarize_metrics(
        metrics,
        labels=config.labels,
        foreground_labels=config.foreground_labels,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.bootstrap_seed,
        cluster_ids=cluster_ids,
    )
    label_summary.to_csv(config.output_dir / "label_summary.csv", index=False)
    case_summary.to_csv(config.output_dir / "case_summary.csv", index=False)
    case_paired_deltas.to_csv(config.output_dir / "case_paired_deltas.csv", index=False)

    timing_summary: dict[str, dict[str, float | int | None]] = {}
    for method in ("argmax", "rankseg"):
        subset = timings[timings["method"] == method]
        peaks = subset["peak_memory_bytes"].dropna()
        failed_attempts = subset["failed_attempt_milliseconds"].dropna()
        failed_peaks = subset["failed_peak_memory_bytes"].dropna()
        timing_summary[method] = {
            "median_milliseconds": float(subset["milliseconds"].median()),
            "mean_milliseconds": float(subset["milliseconds"].mean()),
            "median_postprocessing_milliseconds": float(subset["postprocessing_milliseconds"].median()),
            "mean_postprocessing_milliseconds": float(subset["postprocessing_milliseconds"].mean()),
            "max_peak_memory_bytes": int(peaks.max()) if len(peaks) else None,
            "mean_failed_attempt_milliseconds": (
                float(failed_attempts.mean()) if len(failed_attempts) else None
            ),
            "max_failed_peak_memory_bytes": (
                int(failed_peaks.max()) if len(failed_peaks) else None
            ),
            "device_counts": {
                str(device): int(count)
                for device, count in subset["device"].value_counts().sort_index().items()
            },
            "measurement_complete": True,
        }

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "dataset": {
            "id": config.dataset_id,
            "display_name": config.display_name,
            "cases": len(cases),
            "labels": config.labels,
            "channel_labels": list(config.channel_labels),
            "foreground_labels": list(config.foreground_labels),
            "regions": None
            if config.regions is None
            else {label: list(source_labels) for label, source_labels in config.regions.items()},
            "metric_semantics": "exclusive labels" if config.regions is None else "unions of source labels",
            "ignore_label": config.ignore_label,
            "clusters": None if cluster_ids is None else len(set(cluster_ids.values())),
            "cluster_regex": config.cluster_regex,
            "segmentation_reader": config.segmentation_reader,
        },
        "selection": {
            "case_limit": case_limit,
            "scientific_run": case_limit is None,
            "warning": "case_limit runs are smoke tests and must not be reported" if case_limit is not None else None,
        },
        "decoder": {
            "argmax": {"dimension": 0},
            "rankseg": {
                "metric": config.rankseg_metric,
                "solver": config.rankseg_solver,
                "output_mode": config.rankseg_output_mode,
                "pruning_prob": config.pruning_prob,
                "smooth": config.smooth,
                "unassigned_policy": config.unassigned_policy,
            },
        },
        "postprocessing": None if postprocessor is None else postprocessor.summary(),
        "metrics": metric_summary,
        "timing": timing_summary,
        "input_diagnostics": {
            "crop_restored_cases": int(diagnostics["crop_restored"].sum()),
            "probability_dtype_counts": {
                str(dtype): int(count)
                for dtype, count in diagnostics["dtype"].value_counts().sort_index().items()
            },
            "max_probability_normalization_error": float(diagnostics["max_normalization_error"].max()),
            "max_zero_sum_fraction": float(diagnostics["zero_sum_fraction"].max()),
        },
        "native_argmax_check": None
        if native_frame is None
        else {
            "cases": int(len(native_frame)),
            "mismatch_voxels": int(native_frame["argmax_native_mismatch_voxels"].sum()),
            "valid_voxels": int(native_frame["valid_voxels"].sum()),
            "mismatch_fraction": float(
                native_frame["argmax_native_mismatch_voxels"].sum() / native_frame["valid_voxels"].sum()
            ),
            "max_case_mismatch_voxels": int(native_frame["argmax_native_mismatch_voxels"].max()),
            "native_masks_postprocessed": config.native_predictions_postprocessed,
            "note": (
                "Compared after official postprocessing; the exported probability storage dtype is recorded in input diagnostics."
                if config.native_predictions_postprocessed
                else "Compared before official postprocessing; the exported probability storage dtype is recorded in input diagnostics."
            ),
        },
        "provenance": {"manifest": config.provenance, "runtime": runtime_provenance()},
    }
    summary_path = config.output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary_path

#!/usr/bin/env python3
"""Independent consistency audit for the Task024 PROMISE12 benchmark."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work/v1/independent_test/Task024_Promise"
EXPECTED = {f"Case{i:02d}" for i in range(30)}
VARIANTS = {
    "2d": ROOT / "outputs/Task024_PROMISE12_independent_test_2d_sensitivity",
    "3d_fullres": ROOT / "outputs/Task024_PROMISE12_independent_test_3d_fullres_sensitivity",
    "2d_3d_ensemble": ROOT / "outputs/Task024_PROMISE12_independent_test",
}


def ids(directory: Path, suffix: str) -> set[str]:
    if suffix == ".nii.gz":
        return {p.name[: -len(suffix)] for p in directory.glob(f"*{suffix}")}
    return {p.stem for p in directory.glob(f"*{suffix}")}


def audit_metrics(output_dir: Path) -> dict[str, object]:
    summary = json.loads((output_dir / "summary.json").read_text())
    with (output_dir / "case_label_metrics.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["label"]) == 1]

    by_method: dict[str, list[tuple[float, float]]] = defaultdict(list)
    max_stored_metric_error = 0.0
    for row in rows:
        tp, fp, fn = (int(row[k]) for k in ("tp", "fp", "fn"))
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        max_stored_metric_error = max(
            max_stored_metric_error,
            abs(dice - float(row["dice"])),
            abs(iou - float(row["iou"])),
        )
        by_method[row["method"]].append((dice, iou))

    assert len(rows) == 60
    assert max_stored_metric_error < 1e-14
    recomputed = {}
    for method in ("argmax", "rankseg"):
        values = np.asarray(by_method[method], dtype=np.float64)
        dice, iou = values.mean(axis=0)
        macro = summary["metrics"]["macro"][method]
        assert abs(dice - macro["foreground_mean_dice"]) < 1e-14
        assert abs(iou - macro["foreground_mean_iou"]) < 1e-14
        recomputed[method] = {"mean_dice": float(dice), "mean_iou": float(iou)}

    delta = {
        "dice": recomputed["rankseg"]["mean_dice"] - recomputed["argmax"]["mean_dice"],
        "iou": recomputed["rankseg"]["mean_iou"] - recomputed["argmax"]["mean_iou"],
    }
    assert abs(delta["dice"] - summary["metrics"]["delta"]["foreground_mean_dice"]) < 1e-14
    assert abs(delta["iou"] - summary["metrics"]["delta"]["foreground_mean_iou"]) < 1e-14

    with (output_dir / "native_argmax_check.csv").open(newline="") as f:
        native_rows = list(csv.DictReader(f))
    mismatch_voxels = sum(int(r["argmax_native_mismatch_voxels"]) for r in native_rows)
    assert mismatch_voxels == summary["native_argmax_check"]["mismatch_voxels"]
    assert len(native_rows) == 30

    return {
        "cases": len(by_method["argmax"]),
        "max_tp_fp_fn_metric_recompute_error": max_stored_metric_error,
        "recomputed": recomputed,
        "rankseg_minus_argmax": delta,
        "native_argmax_mismatch_voxels": mismatch_voxels,
    }


def main() -> None:
    prediction_sets = {
        "labels": ids(WORK / "converted/labels", ".nii.gz"),
        "2d_npz": ids(WORK / "predictions_2d", ".npz"),
        "3d_npz": ids(WORK / "predictions_3d_fullres", ".npz"),
        "ensemble_npz": ids(WORK / "predictions_ensemble/not_postprocessed", ".npz"),
        "ensemble_raw_nii": ids(WORK / "predictions_ensemble/not_postprocessed", ".nii.gz"),
        "ensemble_postprocessed_nii": ids(WORK / "predictions_ensemble", ".nii.gz"),
    }
    for name, found in prediction_sets.items():
        assert found == EXPECTED, f"{name}: missing={EXPECTED-found}, extra={found-EXPECTED}"

    max_error_vs_float32_mean = 0.0
    max_error_vs_float16_mean = 0.0
    mismatches_vs_float16_mean = 0
    elements = 0
    for case_id in sorted(EXPECTED):
        a = np.load(WORK / "predictions_2d" / f"{case_id}.npz")["softmax"]
        b = np.load(WORK / "predictions_3d_fullres" / f"{case_id}.npz")["softmax"]
        ensemble = np.load(
            WORK / "predictions_ensemble/not_postprocessed" / f"{case_id}.npz"
        )["softmax"]
        assert a.shape == b.shape == ensemble.shape
        expected32 = (a.astype(np.float32) + b.astype(np.float32)) / 2
        expected16 = expected32.astype(np.float16)
        max_error_vs_float32_mean = max(
            max_error_vs_float32_mean,
            float(np.max(np.abs(ensemble.astype(np.float32) - expected32))),
        )
        max_error_vs_float16_mean = max(
            max_error_vs_float16_mean,
            float(np.max(np.abs(ensemble.astype(np.float32) - expected16.astype(np.float32)))),
        )
        mismatches_vs_float16_mean += int(np.count_nonzero(ensemble != expected16))
        elements += ensemble.size

    assert mismatches_vs_float16_mean == 0
    audit = {
        "status": "pass",
        "expected_case_ids": "Case00--Case29",
        "file_set_counts": {name: len(found) for name, found in prediction_sets.items()},
        "ensemble_probability_audit": {
            "definition": "float16((float32(2d) + float32(3d_fullres)) / 2)",
            "cases": 30,
            "elements": elements,
            "max_absolute_error_vs_float32_mean": max_error_vs_float32_mean,
            "max_absolute_error_vs_float16_mean": max_error_vs_float16_mean,
            "mismatched_elements_vs_float16_mean": mismatches_vs_float16_mean,
        },
        "metric_audits": {name: audit_metrics(path) for name, path in VARIANTS.items()},
    }
    destination = VARIANTS["2d_3d_ensemble"] / "AUDIT.json"
    destination.write_text(json.dumps(audit, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

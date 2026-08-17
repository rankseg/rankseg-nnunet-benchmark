#!/usr/bin/env python3
"""Independent consistency audit for the CHAOS CT / TotalSegmentator v2 run."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from rankseg_nnunet_bench.io import load_probabilities, load_segmentation


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "work/v2/datasets/CHAOS_CT_external"
PREDICTIONS = ROOT / "work/v2/predictions/CHAOS_CT_Dataset291"
OUTPUT = ROOT / "outputs/CHAOS_CT_TotalSegmentator_v2_external"
MODEL = ROOT / "work/v2/model_archives/Dataset291_TotalSegmentator_part1_organs_1559subj.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((DATA / "conversion_manifest.json").read_text())
    summary = json.loads((OUTPUT / "summary.json").read_text())
    assert manifest["cases"] == 20
    assert sha256(MODEL) == "2cde8e6bcfb5b6a02183648a7036c3d6b28bc854576f8e1c34fbd6b4dd5c6c5b"
    assert summary["dataset"]["cases"] == 20
    assert summary["dataset"]["foreground_labels"] == [5]
    assert summary["dataset"]["segmentation_reader"] == "nnunetv2_reoriented"
    assert summary["selection"]["scientific_run"] is True

    mismatches = 0
    valid_voxels = 0
    max_probability_sum_error = 0.0
    cases = []
    for record in manifest["records"]:
        case_id = record["case_id"]
        probabilities, key = load_probabilities(PREDICTIONS / f"{case_id}.npz", "probabilities")
        target = load_segmentation(
            DATA / "labelsTs" / f"{case_id}.nii.gz",
            reader="nnunetv2_reoriented",
        )
        native = load_segmentation(
            PREDICTIONS / f"{case_id}.nii.gz",
            reader="nnunetv2_reoriented",
        )
        assert key == "probabilities"
        assert probabilities.shape == (25, *target.shape)
        assert native.shape == target.shape
        assert set(int(value) for value in np.unique(target)) <= {0, 5}
        assert int(np.count_nonzero(target == 5)) == record["liver_voxels"]
        sums = probabilities.astype(np.float32, copy=False).sum(axis=0)
        max_probability_sum_error = max(max_probability_sum_error, float(np.max(np.abs(sums - 1))))
        argmax = np.argmax(probabilities, axis=0)
        case_mismatch = int(np.count_nonzero(argmax != native))
        mismatches += case_mismatch
        valid_voxels += target.size
        cases.append({"case_id": case_id, "argmax_native_mismatch_voxels": case_mismatch})

    with (OUTPUT / "case_label_metrics.csv").open(newline="") as handle:
        metric_rows = [row for row in csv.DictReader(handle) if int(row["label"]) == 5]
    assert len(metric_rows) == 20 * 2
    max_metric_error = 0.0
    by_method: dict[str, list[tuple[float, float]]] = {"argmax": [], "rankseg": []}
    for row in metric_rows:
        tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        max_metric_error = max(
            max_metric_error,
            abs(dice - float(row["dice"])),
            abs(iou - float(row["iou"])),
        )
        by_method[row["method"]].append((dice, iou))
    assert max_metric_error < 1e-14

    for method in ("argmax", "rankseg"):
        observed = np.asarray(by_method[method], dtype=np.float64).mean(axis=0)
        macro = summary["metrics"]["macro"][method]
        assert abs(observed[0] - macro["foreground_mean_dice"]) < 1e-14
        assert abs(observed[1] - macro["foreground_mean_iou"]) < 1e-14
    native_summary = summary["native_argmax_check"]
    assert native_summary["mismatch_voxels"] == mismatches
    assert native_summary["valid_voxels"] == valid_voxels
    # Probabilities and hard masks are exported from the same float32 array;
    # any discrepancy indicates coordinate or channel mapping corruption.
    assert mismatches == 0
    assert max_probability_sum_error < 1e-5

    audit = {
        "status": "pass",
        "cases": 20,
        "evaluated_foreground_labels": {"5": "liver"},
        "partial_ground_truth": True,
        "probability_shape": "25 x canonical-RAS spatial dimensions",
        "max_probability_sum_error": max_probability_sum_error,
        "argmax_native_mismatch_voxels": mismatches,
        "valid_voxels": valid_voxels,
        "max_tp_fp_fn_metric_recompute_error": max_metric_error,
        "case_native_checks": cases,
    }
    destination = OUTPUT / "AUDIT.json"
    destination.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()

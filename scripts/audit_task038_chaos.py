#!/usr/bin/env python3
"""Independent consistency audit for the Task038 CHAOS benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "work/v1/nnUNet_raw_data_base/nnUNet_raw_data/Task038_CHAOS_Task_3_5_Variant2"
OOF = ROOT / "work/v1/oof"
VARIANTS = {
    "2d": ROOT / "outputs/Task038_CHAOS_2d_oof_sensitivity",
    "3d_fullres": ROOT / "outputs/Task038_CHAOS_3d_fullres_oof_sensitivity",
    "2d_3d_ensemble": ROOT / "outputs/Task038_CHAOS_ensemble_oof",
}


def assignments(directory: Path) -> dict[str, int]:
    with (directory / "fold_assignments.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["case_id"]: int(row["fold"]) for row in rows}
    assert len(result) == len(rows) == 60
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_metrics(output_dir: Path) -> dict[str, object]:
    summary = json.loads((output_dir / "summary.json").read_text())
    with (output_dir / "case_label_metrics.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["label"]) in {1, 2, 3, 4}]

    by_method: dict[str, list[tuple[float, float]]] = defaultdict(list)
    max_error = 0.0
    for row in rows:
        tp, fp, fn = (int(row[key]) for key in ("tp", "fp", "fn"))
        dice = 2 * tp / (2 * tp + fp + fn)
        iou = tp / (tp + fp + fn)
        max_error = max(max_error, abs(dice - float(row["dice"])), abs(iou - float(row["iou"])))
        by_method[row["method"]].append((dice, iou))
    assert len(rows) == 60 * 2 * 4
    assert max_error < 1e-14

    recomputed = {}
    for method in ("argmax", "rankseg"):
        mean_dice, mean_iou = np.asarray(by_method[method], dtype=np.float64).mean(axis=0)
        macro = summary["metrics"]["macro"][method]
        assert abs(mean_dice - macro["foreground_mean_dice"]) < 1e-14
        assert abs(mean_iou - macro["foreground_mean_iou"]) < 1e-14
        recomputed[method] = {"mean_dice": float(mean_dice), "mean_iou": float(mean_iou)}

    delta = {
        metric: recomputed["rankseg"][f"mean_{metric}"] - recomputed["argmax"][f"mean_{metric}"]
        for metric in ("dice", "iou")
    }
    assert abs(delta["dice"] - summary["metrics"]["delta"]["foreground_mean_dice"]) < 1e-14
    assert abs(delta["iou"] - summary["metrics"]["delta"]["foreground_mean_iou"]) < 1e-14

    with (output_dir / "native_argmax_check.csv").open(newline="") as handle:
        native_rows = list(csv.DictReader(handle))
    mismatch = sum(int(row["argmax_native_mismatch_voxels"]) for row in native_rows)
    assert mismatch == summary["native_argmax_check"]["mismatch_voxels"]
    assert len(native_rows) == 60
    return {
        "cases": 60,
        "max_tp_fp_fn_metric_recompute_error": max_error,
        "recomputed": recomputed,
        "rankseg_minus_argmax": delta,
        "native_argmax_mismatch_voxels": mismatch,
    }


def main() -> None:
    expected = {path.name[:-7] for path in (RAW / "labelsTr").glob("*.nii.gz")}
    assert len(expected) == 60
    expected_patients = {case_id.rsplit("_", 1)[1] for case_id in expected}
    assert len(expected_patients) == 20

    two_d = OOF / "Task038_CHAOS_2d"
    fullres = OOF / "Task038_CHAOS_3d_fullres"
    ensemble = OOF / "Task038_CHAOS_ensemble"
    fold_map = assignments(two_d)
    assert fold_map == assignments(fullres)
    ensemble_metadata = json.loads((ensemble / "oof_ensemble_metadata.json").read_text())
    assert ensemble_metadata["fold_assignments_identical"] is True
    assert ensemble_metadata["fold_assignments_sha256"] == sha256(two_d / "fold_assignments.csv")
    for patient in expected_patients:
        patient_cases = {f"T1_in_{patient}", f"T1_out_{patient}", f"T2_{patient}"}
        assert patient_cases <= expected
        assert len({fold_map[case_id] for case_id in patient_cases}) == 1
    assert sorted(list(fold_map.values()).count(fold) for fold in range(5)) == [12] * 5

    max_error_float32 = 0.0
    max_error_float16 = 0.0
    mismatches_float16 = 0
    elements = 0
    file_counts = defaultdict(int)
    for case_id in sorted(expected):
        fold = fold_map[case_id]
        a_path = two_d / "fold_predictions" / f"fold_{fold}" / f"{case_id}.npz"
        b_path = fullres / "fold_predictions" / f"fold_{fold}" / f"{case_id}.npz"
        e_path = (
            ensemble
            / "fold_predictions"
            / f"fold_{fold}"
            / "not_postprocessed"
            / f"{case_id}.npz"
        )
        for name, path in (("2d_npz", a_path), ("3d_npz", b_path), ("ensemble_npz", e_path)):
            assert path.is_file(), path
            file_counts[name] += 1
        a = np.load(a_path)["softmax"]
        b = np.load(b_path)["softmax"]
        observed = np.load(e_path)["softmax"]
        assert a.shape == b.shape == observed.shape
        expected32 = (a.astype(np.float32) + b.astype(np.float32)) / 2
        expected16 = expected32.astype(np.float16)
        max_error_float32 = max(
            max_error_float32, float(np.max(np.abs(observed.astype(np.float32) - expected32)))
        )
        max_error_float16 = max(
            max_error_float16,
            float(np.max(np.abs(observed.astype(np.float32) - expected16.astype(np.float32)))),
        )
        mismatches_float16 += int(np.count_nonzero(observed != expected16))
        elements += observed.size
    assert mismatches_float16 == 0

    audit = {
        "status": "pass",
        "patients": 20,
        "cases": 60,
        "cases_per_patient": 3,
        "fold_validation_case_counts": [list(fold_map.values()).count(fold) for fold in range(5)],
        "patient_sequences_are_fold_colocated": True,
        "file_counts": dict(file_counts),
        "ensemble_probability_audit": {
            "definition": "float16((float32(2d) + float32(3d_fullres)) / 2)",
            "elements": elements,
            "max_absolute_error_vs_float32_mean": max_error_float32,
            "max_absolute_error_vs_float16_mean": max_error_float16,
            "mismatched_elements_vs_float16_mean": mismatches_float16,
        },
        "metric_audits": {name: audit_metrics(path) for name, path in VARIANTS.items()},
    }
    destination = VARIANTS["2d_3d_ensemble"] / "AUDIT.json"
    destination.write_text(json.dumps(audit, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

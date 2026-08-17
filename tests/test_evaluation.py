import json
from pathlib import Path

import numpy as np
import yaml

from rankseg_nnunet_bench.config import load_dataset_config
from rankseg_nnunet_bench.evaluation import evaluate_dataset


def _write_manifest(tmp_path: Path) -> Path:
    probabilities_dir = tmp_path / "probabilities"
    labels_dir = tmp_path / "labels"
    probabilities_dir.mkdir()
    labels_dir.mkdir()

    # Non-consecutive label 4 checks nnU-Net channel-index to label-value mapping.
    channel_indices = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64)
    probabilities = np.full((3, 2, 3), 0.01, dtype=np.float32)
    for channel in range(3):
        probabilities[channel][channel_indices == channel] = 0.98
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    np.savez_compressed(probabilities_dir / "case_001.npz", probabilities=probabilities)
    label_values = np.asarray([0, 1, 4], dtype=np.uint8)[channel_indices]
    np.save(labels_dir / "case_001.npy", label_values)

    manifest = {
        "dataset": {
            "id": "Dataset999_Synthetic",
            "labels": {0: "background", 1: "one", 4: "four"},
            "channel_labels": [0, 1, 4],
        },
        "inputs": {
            "probabilities_dir": str(probabilities_dir),
            "labels_dir": str(labels_dir),
            "label_extension": ".npy",
        },
        "decoder": {"rankseg": {"metric": "dice", "solver": "RMA", "output_mode": "multiclass"}},
        "runtime": {"device": "cpu"},
        "evaluation": {"bootstrap_samples": 20, "bootstrap_seed": 7},
        "outputs": {"dir": str(tmp_path / "outputs")},
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def test_end_to_end_evaluation_and_channel_mapping(tmp_path):
    summary_path = evaluate_dataset(load_dataset_config(_write_manifest(tmp_path)))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["dataset"]["channel_labels"] == [0, 1, 4]
    assert summary["dataset"]["cases"] == 1
    assert summary["metrics"]["macro"]["argmax"]["foreground_mean_dice"] == 1.0
    assert summary["selection"]["scientific_run"] is True
    assert summary["timing"]["argmax"]["device_counts"] == {"cpu": 1}
    assert summary["timing"]["rankseg"]["device_counts"] == {"cpu": 1}
    assert summary["metrics"]["case_level_safety"]["dice"]["cases"] == 1
    assert (summary_path.parent / "case_label_metrics.csv").is_file()
    assert (summary_path.parent / "label_summary.csv").is_file()


def test_case_limit_is_marked_non_scientific(tmp_path):
    summary_path = evaluate_dataset(load_dataset_config(_write_manifest(tmp_path)), case_limit=1)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["selection"]["scientific_run"] is False
    assert "must not be reported" in summary["selection"]["warning"]


def test_end_to_end_region_evaluation_uses_source_label_unions(tmp_path):
    probabilities_dir = tmp_path / "probabilities"
    labels_dir = tmp_path / "labels"
    probabilities_dir.mkdir()
    labels_dir.mkdir()
    class_indices = np.array([[0, 1], [2, 3]], dtype=np.int64)
    probabilities = np.full((4, 2, 2), 0.001, dtype=np.float32)
    for channel in range(4):
        probabilities[channel][class_indices == channel] = 0.997
    probabilities /= probabilities.sum(axis=0, keepdims=True)
    np.savez_compressed(probabilities_dir / "case_001.npz", softmax=probabilities)
    np.save(labels_dir / "case_001.npy", class_indices.astype(np.uint8))
    manifest = {
        "dataset": {
            "id": "Dataset998_Regions",
            "labels": {0: "background", 1: "whole tumour", 2: "tumour core", 3: "enhancing tumour"},
            "channel_labels": [0, 1, 2, 3],
        },
        "inputs": {
            "probabilities_dir": str(probabilities_dir),
            "labels_dir": str(labels_dir),
            "probability_key": "softmax",
            "label_extension": ".npy",
        },
        "decoder": {"rankseg": {"metric": "dice", "solver": "RMA", "output_mode": "multiclass"}},
        "runtime": {"device": "cpu"},
        "evaluation": {
            "foreground_labels": [1, 2, 3],
            "regions": {1: [1, 2, 3], 2: [2, 3], 3: [3]},
            "bootstrap_samples": 20,
            "bootstrap_seed": 11,
        },
        "outputs": {"dir": str(tmp_path / "outputs")},
    }
    manifest_path = tmp_path / "regions.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    summary_path = evaluate_dataset(load_dataset_config(manifest_path))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary["dataset"]["regions"] == {"1": [1, 2, 3], "2": [2, 3], "3": [3]}
    assert summary["dataset"]["metric_semantics"] == "unions of source labels"
    assert summary["metrics"]["macro"]["argmax"]["foreground_mean_dice"] == 1.0

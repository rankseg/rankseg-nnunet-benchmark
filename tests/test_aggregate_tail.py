import json

import pandas as pd
import pytest

from rankseg_nnunet_bench.aggregate import aggregate_summaries


def _summary(dataset_id: str, delta: float) -> dict:
    case_quantiles = {}
    for metric in ("dice", "iou"):
        case_quantiles[metric] = {
            "case_performance": {
                "argmax": {"q05": 0.5, "q10": 0.6},
                "rankseg": {"q05": 0.5 + delta, "q10": 0.6 + delta},
            },
            "bottom_performance_quantile_uplift": {"q05": delta, "q10": delta},
            "paired_improvement": {"q05": delta, "q10": delta * 2},
        }
    return {
        "schema_version": 2,
        "selection": {"scientific_run": True},
        "dataset": {
            "id": dataset_id,
            "display_name": f"{dataset_id} (model details)",
            "cases": 20,
            "labels": {"0": "background", "1": "foreground"},
            "channel_labels": [0, 1],
            "foreground_labels": [1],
        },
        "provenance": {
            "manifest": {
                "model_configuration": (
                    "arithmetic mean of official 2d and 3d_fullres OOF probabilities"
                )
            }
        },
        "decoder": {
            "argmax": {"dimension": 0},
            "rankseg": {"metric": "dice", "solver": "RMA", "output_mode": "multiclass"},
        },
        "timing": {
            method: {
                "median_milliseconds": 1.0,
                "mean_milliseconds": 1.0,
                "median_postprocessing_milliseconds": 0.0,
                "mean_postprocessing_milliseconds": 0.0,
                "max_peak_memory_bytes": None,
                "device_counts": {"cpu": 20},
            }
            for method in ("argmax", "rankseg")
        },
        "metrics": {
            "macro": {
                "argmax": {"foreground_mean_dice": 0.7, "foreground_mean_iou": 0.6},
                "rankseg": {
                    "foreground_mean_dice": 0.7 + delta,
                    "foreground_mean_iou": 0.6 + delta,
                },
            },
            "delta": {"foreground_mean_dice": delta, "foreground_mean_iou": delta},
            "paired_case_bootstrap_delta": {
                "dice": {"samples": 20, "seed": 1, "lower": delta, "median": delta, "upper": delta},
                "iou": {"samples": 20, "seed": 2, "lower": delta, "median": delta, "upper": delta},
            },
            "case_quantiles": case_quantiles,
        },
    }


def test_aggregate_preserves_dataset_level_quantile_statistics(tmp_path):
    paths = []
    for dataset_id, delta in (("positive", 0.01), ("negative", -0.02)):
        path = tmp_path / f"{dataset_id}.json"
        path.write_text(json.dumps(_summary(dataset_id, delta)), encoding="utf-8")
        paths.append(path)

    output = tmp_path / "aggregate"
    aggregate_path = aggregate_summaries(paths, output, include_overall_tests=True)
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    difficult = aggregate["tail_analysis"]["difficult_case_analysis"]
    q05 = difficult["bottom_performance_quantile_uplift"]["dice"]["q05"]
    frame = pd.read_csv(output / "dataset_summary.csv")

    assert aggregate["schema_version"] == 3
    assert aggregate["dice"]["argmax_mean"] == pytest.approx(0.7)
    assert aggregate["dice"]["rankseg_mean"] == pytest.approx(0.695)
    assert q05["datasets"] == 2
    assert q05["positive"] == 1
    assert q05["negative"] == 1
    assert "bottom_quantile_uplift_dice_q05" in frame.columns
    assert "paired_delta_dice_q05" in frame.columns
    assert set(frame["display_name"]) == {"positive", "negative"}
    assert set(frame["model"]) == {"nnU-Net v1 · 2d + 3d_fullres ensemble"}
    assert not any(column.startswith("hard_") for column in frame.columns)
    assert "baseline_defined_hard_cases" not in difficult
    assert "exact_two_sided_sign_test_p" in aggregate["dice"]
    assert "iou" not in aggregate
    assert not any("iou" in column.lower() for column in frame.columns)
    assert "exact_two_sided_sign_test_p" not in q05
    report = (output / "RESULTS.md").read_text(encoding="utf-8")
    assert "Argmax mean Dice (%)" in report
    assert "RankSEG mean Dice (%)" in report
    assert "| Task | Dataset | Model | Cases |" in report
    assert "IoU" not in report
    assert "Overall benchmark p-values" in report
    assert "Bottom 5% mean" not in report
    assert not (output / "dice_one_sided_paired_t_tests.csv").exists()

    descriptive_output = tmp_path / "descriptive_aggregate"
    descriptive = json.loads(
        aggregate_summaries(paths, descriptive_output).read_text(encoding="utf-8")
    )
    assert "exact_two_sided_sign_test_p" not in descriptive["dice"]
    assert "Overall benchmark p-values" not in (
        descriptive_output / "RESULTS.md"
    ).read_text(encoding="utf-8")


def test_aggregate_rejects_duplicate_dataset_configurations(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"configuration_{index}.json"
        path.write_text(json.dumps(_summary("same_dataset", 0.01)), encoding="utf-8")
        paths.append(path)

    with pytest.raises(ValueError, match="one predeclared configuration"):
        aggregate_summaries(paths, tmp_path / "aggregate")


def test_aggregate_rejects_post_hoc_safety_guard(tmp_path):
    summary = _summary("guarded", 0.01)
    summary["safety_guard"] = {"maximum_ratio": 2.0}
    path = tmp_path / "guarded.json"
    path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="post-hoc guarded"):
        aggregate_summaries([path], tmp_path / "aggregate")


def test_aggregate_includes_case_outcome_counts_and_symmetric_events(tmp_path):
    paths = []
    for dataset_id, deltas in (("a", [-0.11, 0.02]), ("b", [0.0, 0.12])):
        folder = tmp_path / dataset_id
        folder.mkdir()
        path = folder / "summary.json"
        summary = _summary(dataset_id, 0.01)
        summary["dataset"]["cases"] = 2
        path.write_text(json.dumps(summary), encoding="utf-8")
        pd.DataFrame(
            {
                "case_id": [f"{dataset_id}_0", f"{dataset_id}_1"],
                "delta_dice": deltas,
                "delta_iou": deltas,
            }
        ).to_csv(folder / "case_paired_deltas.csv", index=False)
        paths.append(path)

    aggregate = json.loads(
        aggregate_summaries(paths, tmp_path / "aggregate").read_text(encoding="utf-8")
    )
    safety = aggregate["case_level_safety"]

    assert safety["complete"] is True
    assert safety["dice"]["cases"] == 4
    assert safety["dice"]["practical_threshold_pp"] == 1
    assert safety["dice"]["improved_over_1pp"] == 2
    assert safety["dice"]["stable_within_1pp"] == 1
    assert safety["dice"]["worsened_over_1pp"] == 1
    assert safety["dice"]["thresholds"]["5_pp"]["negative"] == 1
    assert safety["dice"]["thresholds"]["10_pp"]["positive"] == 1
    assert safety["dice"]["worst"] == {"case_id": "a_0", "dataset_id": "a", "delta": -0.11}
    frame = pd.read_csv(tmp_path / "aggregate" / "dataset_summary.csv").set_index("dataset_id")
    assert frame.loc["a", "case_dice_improved_over_1pp"] == 1
    assert frame.loc["a", "case_dice_worsened_over_1pp"] == 1
    assert frame.loc["b", "case_dice_stable_within_1pp"] == 1
    combined = pd.read_csv(tmp_path / "aggregate" / "case_paired_deltas.csv")
    assert list(combined.columns) == ["dataset_id", "case_id", "delta_dice"]
    assert len(combined) == 4
    report = (tmp_path / "aggregate" / "RESULTS.md").read_text(encoding="utf-8")
    assert "2 improved, 1 stable, 1 worsened" in report

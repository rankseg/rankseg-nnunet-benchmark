import math

import numpy as np
import pandas as pd

from rankseg_nnunet_bench.metrics import (
    overlap_metrics,
    overlap_region_metrics,
    paired_cluster_bootstrap_delta,
)


def test_overlap_metrics_matches_nnunet_empty_convention():
    target = np.array([[0, 1], [0, 1]], dtype=np.uint8)
    prediction = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    rows = {row["label"]: row for row in overlap_metrics(prediction, target, (0, 1, 2))}

    assert rows[1]["tp"] == 1
    assert rows[1]["fp"] == 0
    assert rows[1]["fn"] == 1
    assert rows[1]["dice"] == 2 / 3
    assert rows[1]["iou"] == 1 / 2
    assert math.isnan(rows[2]["dice"])
    assert math.isnan(rows[2]["iou"])


def test_ignore_label_is_excluded_from_false_positives():
    target = np.array([[255, 1]], dtype=np.uint8)
    prediction = np.array([[1, 1]], dtype=np.uint8)
    row = overlap_metrics(prediction, target, (1,), ignore_label=255)[0]
    assert row["tp"] == 1
    assert row["fp"] == 0
    assert row["fn"] == 0


def test_overlap_region_metrics_supports_nested_brats_regions():
    prediction = np.array([[0, 1, 2, 3]], dtype=np.uint8)
    target = np.array([[0, 1, 3, 3]], dtype=np.uint8)
    rows = {
        row["label"]: row
        for row in overlap_region_metrics(
            prediction,
            target,
            {1: (1, 2, 3), 2: (2, 3), 3: (3,)},
        )
    }

    assert rows[1]["dice"] == 1.0
    assert rows[2]["dice"] == 1.0
    assert rows[3]["tp"] == 1
    assert rows[3]["fn"] == 1
    assert rows[3]["dice"] == 2 / 3


def test_paired_cluster_bootstrap_resamples_subjects_with_repeated_cases():
    rows = []
    for case_id, argmax, rankseg in (
        ("patient001_ED", 0.70, 0.72),
        ("patient001_ES", 0.80, 0.82),
        ("patient002_ED", 0.60, 0.63),
        ("patient002_ES", 0.90, 0.91),
    ):
        for method, value in (("argmax", argmax), ("rankseg", rankseg)):
            rows.append({"case_id": case_id, "label": 1, "method": method, "dice": value})
    result = paired_cluster_bootstrap_delta(
        pd.DataFrame(rows),
        metric="dice",
        foreground_labels=(1,),
        cluster_ids={case_id: case_id[:10] for case_id, *_ in (
            ("patient001_ED", 0, 0),
            ("patient001_ES", 0, 0),
            ("patient002_ED", 0, 0),
            ("patient002_ES", 0, 0),
        )},
        samples=100,
        seed=13,
    )

    assert result["clusters"] == 2
    assert result["lower"] > 0
    assert result["upper"] > 0

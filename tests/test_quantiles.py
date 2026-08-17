import pandas as pd
import pytest

from rankseg_nnunet_bench.metrics import summarize_metrics


def test_case_quantiles_and_paired_deltas_are_retained():
    rows = []
    for case, argmax, rankseg in (("easy", 0.9, 0.8), ("medium", 0.6, 0.7), ("hard", 0.2, 0.5)):
        for method, value in (("argmax", argmax), ("rankseg", rankseg)):
            rows.append(
                {
                    "case_id": case,
                    "method": method,
                    "label": 1,
                    "dice": value,
                    "iou": value,
                    "tp": 1,
                    "fp": 0,
                    "fn": 0,
                }
            )
    _, _, paired, summary = summarize_metrics(
        pd.DataFrame(rows),
        labels={0: "background", 1: "foreground"},
        foreground_labels=(1,),
        bootstrap_samples=0,
        bootstrap_seed=1,
    )
    assert paired.loc[paired.case_id == "hard", "delta_dice"].item() == pytest.approx(0.3)
    assert summary["case_quantiles"]["dice"]["bottom_performance_quantile_uplift"]["q10"] > 0
    assert summary["case_quantiles"]["dice"]["paired_improvement"]["q10"] < 0.3
    assert "baseline_defined_hard_cases" not in summary

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from .metrics import CASE_EVENT_THRESHOLDS_PP, case_level_event_summary
from .schema import AGGREGATE_SCHEMA_VERSION, migrate_summary


def _format_pvalue(value: float | None) -> str:
    return "NA" if value is None else f"{value:.6g}"


def _format_count(value: float | int | None) -> str:
    return "NA" if value is None or pd.isna(value) else str(int(value))


def _dataset_label(display_name: str) -> str:
    """Keep cohort identity in Dataset; model/evaluation details have their own columns."""
    return display_name.partition(" (")[0]


def _model_label(manifest_provenance: dict) -> str:
    """Return a compact, public-facing label for the locked nnU-Net model."""
    raw = str(
        manifest_provenance.get("model_configuration")
        or manifest_provenance.get("model")
        or ""
    )
    normalized = raw.lower()
    if not normalized:
        return "Not specified"
    if "totalsegmentator" in normalized:
        return "nnU-Net v2 · TotalSegmentator Dataset291 · 3d_fullres (fold 0)"

    prefix = "nnU-Net v1 · " if "official" in normalized or "nnunetplansv2.1" in normalized else ""
    if "arithmetic mean" in normalized:
        configurations = [
            configuration
            for configuration in ("2d", "3d_lowres", "3d_fullres", "3d_cascade_fullres")
            if configuration in normalized
        ]
        if configurations:
            return f"{prefix}{' + '.join(configurations)} ensemble"
    if "3d_cascade_fullres" in normalized:
        suffix = " (3d_lowres input)" if "3d_lowres" in normalized else ""
        return f"{prefix}3d_cascade_fullres{suffix}"
    for configuration in ("3d_fullres", "3d_lowres", "2d"):
        if configuration in normalized:
            return f"{prefix}{configuration}"
    return raw


def _metric_stats(values: np.ndarray, *, include_overall_tests: bool = False) -> dict:
    tolerance = 1e-12
    positive = int(np.count_nonzero(values > tolerance))
    negative = int(np.count_nonzero(values < -tolerance))
    stats = {
        "datasets": int(values.size),
        "positive": positive,
        "zero": int(values.size - positive - negative),
        "negative": negative,
        "mean_delta": float(np.mean(values)),
        "median_delta": float(np.median(values)),
        "minimum_delta": float(np.min(values)),
        "maximum_delta": float(np.max(values)),
    }
    if include_overall_tests:
        nonzero = values[np.abs(values) > tolerance]
        sign_p = (
            float(binomtest(positive, positive + negative, 0.5).pvalue)
            if positive + negative
            else 1.0
        )
        if nonzero.size >= 2:
            try:
                wilcoxon_p = float(wilcoxon(nonzero, alternative="two-sided").pvalue)
            except ValueError:
                wilcoxon_p = None
        else:
            wilcoxon_p = None
        stats.update(
            {
                "exact_two_sided_sign_test_p": sign_p,
                "wilcoxon_two_sided_p": wilcoxon_p,
            }
        )
    return stats


def aggregate_summaries(
    summary_paths: list[Path], output_dir: Path, *, include_overall_tests: bool = False
) -> Path:
    rows: list[dict] = []
    case_frames: list[pd.DataFrame] = []
    seen_dataset_ids: set[str] = set()
    for path in summary_paths:
        with path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        timings_path = path.parent / "timings.csv"
        timings = pd.read_csv(timings_path) if timings_path.is_file() else None
        summary = migrate_summary(summary, timings=timings)
        if not summary["selection"]["scientific_run"]:
            raise ValueError(f"Refusing to aggregate smoke-test result with case_limit: {path}")
        if summary.get("safety_guard") is not None:
            raise ValueError(f"Refusing to aggregate post-hoc guarded RankSEG development result: {path}")
        dataset_id = summary["dataset"]["id"]
        if dataset_id in seen_dataset_ids:
            raise ValueError(
                f"More than one result supplied for dataset {dataset_id}; "
                "choose one predeclared configuration"
            )
        seen_dataset_ids.add(dataset_id)
        manifest_provenance = summary.get("provenance", {}).get("manifest", {})
        if manifest_provenance.get("configuration_selected_before_rankseg_evaluation") is False:
            raise ValueError(f"Refusing to aggregate sensitivity/non-primary configuration: {path}")
        macro = summary["metrics"]["macro"]
        delta = summary["metrics"]["delta"]
        row = {
            "dataset_id": dataset_id,
            "display_name": _dataset_label(summary["dataset"]["display_name"]),
            "model": _model_label(manifest_provenance),
            "cases": summary["dataset"]["cases"],
            "argmax_dice": macro["argmax"]["foreground_mean_dice"],
            "rankseg_dice": macro["rankseg"]["foreground_mean_dice"],
            "delta_dice": delta["foreground_mean_dice"],
            "case_dice_improved_over_1pp": None,
            "case_dice_stable_within_1pp": None,
            "case_dice_worsened_over_1pp": None,
        }
        case_path = path.parent / "case_paired_deltas.csv"
        if case_path.is_file():
            case_frame = pd.read_csv(case_path)
            required = {"case_id", "delta_dice"}
            if not required <= set(case_frame.columns):
                raise ValueError(f"Case delta file is missing columns {sorted(required - set(case_frame.columns))}: {case_path}")
            if len(case_frame) != int(summary["dataset"]["cases"]):
                raise ValueError(f"Case delta row count does not match summary: {case_path}")
            case_frame = case_frame[["case_id", "delta_dice"]].copy()
            case_frame.insert(0, "dataset_id", dataset_id)
            case_frames.append(case_frame)
            dataset_events = case_level_event_summary(case_frame)
            dice_events = dataset_events["dice"]
            for outcome in ("improved_over_1pp", "stable_within_1pp", "worsened_over_1pp"):
                row[f"case_dice_{outcome}"] = dice_events[outcome]
            for threshold_pp in (5, 10):
                events = dataset_events["dice"]["thresholds"][f"{threshold_pp}_pp"]
                row[f"case_dice_worsened_over_{threshold_pp}pp"] = events["negative"]
                row[f"case_dice_improved_over_{threshold_pp}pp"] = events["positive"]
        metric_quantiles = summary["metrics"]["case_quantiles"]["dice"]
        for quantile in ("q05", "q10"):
            row[f"paired_delta_dice_{quantile}"] = metric_quantiles["paired_improvement"][quantile]
            row[f"argmax_dice_{quantile}"] = metric_quantiles["case_performance"]["argmax"][quantile]
            row[f"rankseg_dice_{quantile}"] = metric_quantiles["case_performance"]["rankseg"][quantile]
            row[f"bottom_quantile_uplift_dice_{quantile}"] = metric_quantiles[
                "bottom_performance_quantile_uplift"
            ][quantile]
        rows.append(row)
    if not rows:
        raise ValueError("No summaries supplied")
    frame = pd.DataFrame(rows).sort_values("dataset_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "dataset_summary.csv", index=False)
    aggregate = {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "unit_of_analysis": "dataset",
        "dice": _metric_stats(
            frame["delta_dice"].to_numpy(dtype=float), include_overall_tests=include_overall_tests
        ),
        "tail_analysis": {
            "unit_of_analysis": "dataset",
            "difficult_case_analysis": {
                "bottom_performance_quantile_uplift": {
                    "dice": {
                        quantile: _metric_stats(
                            frame[f"bottom_quantile_uplift_dice_{quantile}"].to_numpy(dtype=float)
                        )
                        for quantile in ("q05", "q10")
                    }
                },
            },
            "safety_analysis_paired_case_delta_quantiles": {
                "dice": {
                    quantile: _metric_stats(
                        frame[f"paired_delta_dice_{quantile}"].to_numpy(dtype=float)
                    )
                    for quantile in ("q05", "q10")
                }
            },
        },
    }
    aggregate["dice"].update(
        {
            "argmax_mean": float(frame["argmax_dice"].mean()),
            "rankseg_mean": float(frame["rankseg_dice"].mean()),
        }
    )
    if case_frames:
        combined_cases = pd.concat(case_frames, ignore_index=True)
        combined_cases.to_csv(output_dir / "case_paired_deltas.csv", index=False)
        case_events = case_level_event_summary(combined_cases)
        for direction, index in (
            ("worst", combined_cases["delta_dice"].idxmin()),
            ("best", combined_cases["delta_dice"].idxmax()),
        ):
            case_events["dice"][direction]["dataset_id"] = str(
                combined_cases.loc[index, "dataset_id"]
            )
        aggregate["case_level_safety"] = {
            "unit_of_analysis": "case",
            "datasets_with_case_data": len(case_frames),
            "datasets_total": len(rows),
            "complete": len(case_frames) == len(rows),
            **case_events,
        }
    aggregate_path = output_dir / "aggregate_summary.json"
    with aggregate_path.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")

    markdown = [
        "| Task | Dataset | Model | Cases | Argmax mean Dice (%) | RankSEG mean Dice (%) | Dice Δ (pp) | Improved (>+1 pp) | Stable (within ±1 pp) | Worsened (<-1 pp) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in frame.itertuples(index=False):
        task = str(row.dataset_id).split("_", 1)[0]
        markdown.append(
            f"| {task} | {row.display_name} | {row.model} | {row.cases} | {100 * row.argmax_dice:.5f} | "
            f"{100 * row.rankseg_dice:.5f} | {100 * row.delta_dice:+.5f} | "
            f"{_format_count(row.case_dice_improved_over_1pp)} | "
            f"{_format_count(row.case_dice_stable_within_1pp)} | "
            f"{_format_count(row.case_dice_worsened_over_1pp)} |"
        )
    markdown.extend(
        [
            "",
            f"Macro-average Dice across datasets (each dataset weighted equally): "
            f"argmax {100 * aggregate['dice']['argmax_mean']:.5f}%, "
            f"RankSEG {100 * aggregate['dice']['rankseg_mean']:.5f}%. Dataset-level Dice deltas: "
            f"{aggregate['dice']['positive']} positive, "
            f"{aggregate['dice']['zero']} tied, {aggregate['dice']['negative']} negative; "
            f"mean {100 * aggregate['dice']['mean_delta']:+.5f} pp, "
            f"median {100 * aggregate['dice']['median_delta']:+.5f} pp.",
            "",
        ]
    )
    if include_overall_tests:
        markdown.extend(
            [
                "Overall benchmark p-values (dataset as the unit; two-sided): "
                f"Dice exact sign test p={_format_pvalue(aggregate['dice']['exact_two_sided_sign_test_p'])}, "
                f"Dice Wilcoxon p={_format_pvalue(aggregate['dice']['wilcoxon_two_sided_p'])}.",
                "",
            ]
        )
    markdown.append(
        "Overall mean Dice is the main endpoint. Case counts and performance/treatment-effect quantiles are "
        "descriptive diagnostics. Each dataset contributes one unit to the cross-dataset summary."
    )
    if "case_level_safety" in aggregate:
        safety = aggregate["case_level_safety"]["dice"]
        markdown.extend(
            [
                "",
                f"Case-level Dice changes at the ±1 pp practical threshold ({safety['cases']} cases): "
                f"{safety['improved_over_1pp']} improved, "
                f"{safety['stable_within_1pp']} stable, "
                f"{safety['worsened_over_1pp']} worsened.",
                f"More extreme case-level Dice changes: "
                + "; ".join(
                    f">{threshold_pp} pp: {safety['thresholds'][f'{threshold_pp}_pp']['negative']} losses, "
                    f"{safety['thresholds'][f'{threshold_pp}_pp']['positive']} gains"
                    for threshold_pp in (5, 10)
                )
                + ".",
                f"Worst: {safety['worst']['dataset_id']}/{safety['worst']['case_id']} "
                f"({100 * safety['worst']['delta']:+.2f} pp); best: "
                f"{safety['best']['dataset_id']}/{safety['best']['case_id']} "
                f"({100 * safety['best']['delta']:+.2f} pp).",
            ]
        )
    (output_dir / "RESULTS.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return aggregate_path

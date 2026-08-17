from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


METRICS = ("dice", "iou")
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CASE_EVENT_THRESHOLDS_PP = (1, 5, 10)


def _overlap_row(
    label: int,
    prediction_mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, int | float]:
    tp = int(np.count_nonzero(prediction_mask & target_mask))
    fp = int(np.count_nonzero(prediction_mask & ~target_mask))
    fn = int(np.count_nonzero(~prediction_mask & target_mask))
    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn
    return {
        "label": int(label),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "dice": float(2 * tp / dice_denominator) if dice_denominator else math.nan,
        "iou": float(tp / iou_denominator) if iou_denominator else math.nan,
    }


def case_level_event_summary(paired: pd.DataFrame) -> dict:
    """Summarize paired case-level changes without selecting on outcome."""
    summary: dict[str, dict] = {}
    for metric in METRICS:
        column = f"delta_{metric}"
        if column not in paired.columns:
            continue
        values = paired[["case_id", column]].dropna().copy()
        if values.empty:
            summary[metric] = {
                "cases": 0,
                "practical_threshold_pp": 1,
                "improved_over_1pp": 0,
                "stable_within_1pp": 0,
                "worsened_over_1pp": 0,
                "thresholds": {},
                "worst": None,
                "best": None,
            }
            continue
        values[column] = values[column].astype(float)
        thresholds = {}
        for threshold_pp in CASE_EVENT_THRESHOLDS_PP:
            threshold = threshold_pp / 100.0
            negative = int((values[column] < -threshold).sum())
            positive = int((values[column] > threshold).sum())
            thresholds[f"{threshold_pp}_pp"] = {
                "negative": negative,
                "negative_fraction": negative / len(values),
                "positive": positive,
                "positive_fraction": positive / len(values),
            }
        practical = thresholds["1_pp"]
        worst = values.loc[values[column].idxmin()]
        best = values.loc[values[column].idxmax()]
        summary[metric] = {
            "cases": int(len(values)),
            "practical_threshold_pp": 1,
            "improved_over_1pp": practical["positive"],
            "stable_within_1pp": int(
                len(values) - practical["positive"] - practical["negative"]
            ),
            "worsened_over_1pp": practical["negative"],
            "thresholds": thresholds,
            "worst": {"case_id": str(worst["case_id"]), "delta": float(worst[column])},
            "best": {"case_id": str(best["case_id"]), "delta": float(best[column])},
        }
    return summary


def overlap_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    labels: Iterable[int],
    *,
    ignore_label: int | None = None,
) -> list[dict[str, int | float]]:
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} versus {target.shape}")
    valid = np.ones(target.shape, dtype=bool) if ignore_label is None else target != ignore_label
    rows: list[dict[str, int | float]] = []
    for label in labels:
        pred_mask = (prediction == label) & valid
        target_mask = (target == label) & valid
        rows.append(_overlap_row(int(label), pred_mask, target_mask))
    return rows


def overlap_region_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    regions: dict[int, tuple[int, ...]],
    *,
    ignore_label: int | None = None,
) -> list[dict[str, int | float]]:
    """Compute overlap for possibly nested unions of source segmentation labels."""
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch: {prediction.shape} versus {target.shape}")
    valid = np.ones(target.shape, dtype=bool) if ignore_label is None else target != ignore_label
    rows: list[dict[str, int | float]] = []
    for metric_label, source_labels in regions.items():
        pred_mask = np.isin(prediction, source_labels) & valid
        target_mask = np.isin(target, source_labels) & valid
        rows.append(_overlap_row(metric_label, pred_mask, target_mask))
    return rows


def _macro_from_array(values: np.ndarray, case_indices: np.ndarray, method_index: int) -> float:
    selected = values[case_indices, :, method_index]
    finite = np.isfinite(selected)
    counts = finite.sum(axis=0)
    sums = np.nansum(selected, axis=0)
    label_means = np.full(selected.shape[1], np.nan, dtype=np.float64)
    np.divide(sums, counts, out=label_means, where=counts > 0)
    return float(np.nanmean(label_means)) if np.any(np.isfinite(label_means)) else math.nan


def paired_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    metric: str,
    foreground_labels: tuple[int, ...],
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    if samples <= 0:
        return {"samples": 0, "seed": seed, "lower": None, "median": None, "upper": None}
    cases = sorted(frame["case_id"].unique())
    methods = ("argmax", "rankseg")
    case_index = {case: index for index, case in enumerate(cases)}
    label_index = {label: index for index, label in enumerate(foreground_labels)}
    method_index = {method: index for index, method in enumerate(methods)}
    values = np.full((len(cases), len(foreground_labels), len(methods)), np.nan, dtype=np.float64)
    for row in frame.itertuples(index=False):
        if row.label not in label_index or row.method not in method_index:
            continue
        values[case_index[row.case_id], label_index[row.label], method_index[row.method]] = float(
            getattr(row, metric)
        )

    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        draw = rng.integers(0, len(cases), size=len(cases))
        argmax = _macro_from_array(values, draw, method_index["argmax"])
        rankseg = _macro_from_array(values, draw, method_index["rankseg"])
        deltas[index] = rankseg - argmax
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {"samples": samples, "seed": seed, "lower": None, "median": None, "upper": None}
    lower, median, upper = np.percentile(finite, [2.5, 50.0, 97.5])
    return {
        "samples": samples,
        "seed": seed,
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
    }


def paired_cluster_bootstrap_delta(
    frame: pd.DataFrame,
    *,
    metric: str,
    foreground_labels: tuple[int, ...],
    cluster_ids: dict[str, str],
    samples: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Bootstrap paired macro scores by subject/sequence clusters rather than cases."""
    cases = sorted(frame["case_id"].unique())
    missing = sorted(set(cases) - set(cluster_ids))
    if missing:
        raise ValueError(f"Missing cluster IDs for cases: {missing[:5]}")
    clusters = sorted({cluster_ids[case] for case in cases})
    if samples <= 0:
        return {
            "samples": 0,
            "seed": seed,
            "clusters": len(clusters),
            "lower": None,
            "median": None,
            "upper": None,
        }

    methods = ("argmax", "rankseg")
    case_index = {case: index for index, case in enumerate(cases)}
    label_index = {label: index for index, label in enumerate(foreground_labels)}
    method_index = {method: index for index, method in enumerate(methods)}
    values = np.full((len(cases), len(foreground_labels), len(methods)), np.nan, dtype=np.float64)
    for row in frame.itertuples(index=False):
        if row.label not in label_index or row.method not in method_index:
            continue
        values[case_index[row.case_id], label_index[row.label], method_index[row.method]] = float(
            getattr(row, metric)
        )
    cluster_case_indices = {
        cluster: np.asarray(
            [case_index[case] for case in cases if cluster_ids[case] == cluster], dtype=np.int64
        )
        for cluster in clusters
    }

    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        cluster_draw = rng.integers(0, len(clusters), size=len(clusters))
        case_draw = np.concatenate([cluster_case_indices[clusters[item]] for item in cluster_draw])
        argmax = _macro_from_array(values, case_draw, method_index["argmax"])
        rankseg = _macro_from_array(values, case_draw, method_index["rankseg"])
        deltas[index] = rankseg - argmax
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {
            "samples": samples,
            "seed": seed,
            "clusters": len(clusters),
            "lower": None,
            "median": None,
            "upper": None,
        }
    lower, median, upper = np.percentile(finite, [2.5, 50.0, 97.5])
    return {
        "samples": samples,
        "seed": seed,
        "clusters": len(clusters),
        "lower": float(lower),
        "median": float(median),
        "upper": float(upper),
    }


def summarize_metrics(
    frame: pd.DataFrame,
    *,
    labels: dict[int, str],
    foreground_labels: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
    cluster_ids: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    label_rows: list[dict] = []
    for method in ("argmax", "rankseg"):
        method_frame = frame[frame["method"] == method]
        for label in foreground_labels:
            subset = method_frame[method_frame["label"] == label]
            label_rows.append(
                {
                    "method": method,
                    "label": label,
                    "label_name": labels[label],
                    "cases": int(subset["case_id"].nunique()),
                    "valid_dice_cases": int(subset["dice"].notna().sum()),
                    "valid_iou_cases": int(subset["iou"].notna().sum()),
                    "mean_dice": float(subset["dice"].mean()),
                    "mean_iou": float(subset["iou"].mean()),
                    "sum_tp": int(subset["tp"].sum()),
                    "sum_fp": int(subset["fp"].sum()),
                    "sum_fn": int(subset["fn"].sum()),
                }
            )
    label_summary = pd.DataFrame(label_rows)

    case_summary = (
        frame[frame["label"].isin(foreground_labels)]
        .groupby(["case_id", "method"], as_index=False)[list(METRICS)]
        .mean()
        .rename(columns={"dice": "foreground_mean_dice", "iou": "foreground_mean_iou"})
    )

    macro: dict[str, dict[str, float]] = {}
    for method in ("argmax", "rankseg"):
        subset = label_summary[label_summary["method"] == method]
        macro[method] = {
            "foreground_mean_dice": float(subset["mean_dice"].mean()),
            "foreground_mean_iou": float(subset["mean_iou"].mean()),
        }
    delta = {
        metric_name: macro["rankseg"][metric_name] - macro["argmax"][metric_name]
        for metric_name in ("foreground_mean_dice", "foreground_mean_iou")
    }
    bootstrap = {
        metric: paired_bootstrap_delta(
            frame,
            metric=metric,
            foreground_labels=foreground_labels,
            samples=bootstrap_samples,
            seed=bootstrap_seed + index,
        )
        for index, metric in enumerate(METRICS)
    }
    cluster_bootstrap = (
        None
        if cluster_ids is None
        else {
            metric: paired_cluster_bootstrap_delta(
                frame,
                metric=metric,
                foreground_labels=foreground_labels,
                cluster_ids=cluster_ids,
                samples=bootstrap_samples,
                seed=bootstrap_seed + index,
            )
            for index, metric in enumerate(METRICS)
        }
    )
    paired = case_summary.pivot(index="case_id", columns="method").reset_index()
    paired.columns = [
        "case_id" if column[0] == "case_id" else f"{column[1]}_{column[0]}"
        for column in paired.columns.to_flat_index()
    ]
    for metric_index, metric in enumerate(METRICS):
        paired[f"delta_{metric}"] = (
            paired[f"rankseg_foreground_mean_{metric}"] - paired[f"argmax_foreground_mean_{metric}"]
        )
    if cluster_ids is not None:
        paired["cluster_id"] = paired["case_id"].map(cluster_ids)

    quantile_summary: dict[str, dict] = {}
    for metric in METRICS:
        delta_column = f"delta_{metric}"
        case_performance = {
            method: {
                f"q{int(round(quantile * 100)):02d}": float(
                    paired[f"{method}_foreground_mean_{metric}"].quantile(quantile)
                )
                for quantile in QUANTILES
            }
            for method in ("argmax", "rankseg")
        }
        quantile_summary[metric] = {
            "case_performance": case_performance,
            "bottom_performance_quantile_uplift": {
                f"q{int(round(quantile * 100)):02d}": (
                    case_performance["rankseg"][f"q{int(round(quantile * 100)):02d}"]
                    - case_performance["argmax"][f"q{int(round(quantile * 100)):02d}"]
                )
                for quantile in QUANTILES
            },
            # This is a safety/downside statistic, not the difficult-case analysis: it ranks
            # cases by treatment effect instead of by baseline performance.
            "paired_improvement": {
                f"q{int(round(quantile * 100)):02d}": float(paired[delta_column].quantile(quantile))
                for quantile in QUANTILES
            },
        }

    summary = {
        "macro": macro,
        "delta": delta,
        "paired_case_bootstrap_delta": bootstrap,
        "paired_cluster_bootstrap_delta": cluster_bootstrap,
        "case_quantiles": quantile_summary,
        "case_level_safety": case_level_event_summary(paired),
    }
    return label_summary, case_summary, paired, summary

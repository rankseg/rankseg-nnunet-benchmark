from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


def _resolve(base: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str
    display_name: str
    labels: dict[int, str]
    channel_labels: tuple[int, ...]
    probabilities_dir: Path
    labels_dir: Path
    output_dir: Path
    foreground_labels: tuple[int, ...]
    regions: dict[int, tuple[int, ...]] | None = None
    probability_key: str = "auto"
    probability_glob: str = "*.npz"
    label_extension: str = ".nii.gz"
    segmentation_reader: str = "simpleitk"
    native_predictions_dir: Path | None = None
    native_predictions_postprocessed: bool = True
    postprocessing_file: Path | None = None
    ignore_label: int | None = None
    case_ids: tuple[str, ...] | None = None
    device: str = "cuda"
    rankseg_metric: str = "dice"
    rankseg_solver: str = "RMA"
    rankseg_output_mode: str = "multiclass"
    pruning_prob: float = 0.5
    smooth: float = 0.0
    unassigned_policy: str = "max_score"
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 20260717
    cluster_regex: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    manifest_path = Path(path).expanduser().resolve()
    with manifest_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Manifest must be a mapping: {manifest_path}")

    base = manifest_path.parent
    dataset = raw.get("dataset", {})
    inputs = raw.get("inputs", {})
    decoder = raw.get("decoder", {}).get("rankseg", {})
    evaluation = raw.get("evaluation", {})
    runtime = raw.get("runtime", {})

    labels = {int(key): str(value) for key, value in dataset.get("labels", {}).items()}
    if not labels or 0 not in labels:
        raise ValueError("dataset.labels must include integer label 0 (background)")

    ignore_label = dataset.get("ignore_label")
    ignore_label = None if ignore_label is None else int(ignore_label)
    channel_labels = tuple(
        int(label)
        for label in dataset.get(
            "channel_labels",
            [label for label in sorted(labels) if label != ignore_label],
        )
    )
    if len(set(channel_labels)) != len(channel_labels) or not set(channel_labels) <= set(labels):
        raise ValueError("dataset.channel_labels must be unique values drawn from dataset.labels")
    if 0 not in channel_labels:
        raise ValueError("dataset.channel_labels must contain background label 0 for multiclass decoding")
    foreground = evaluation.get("foreground_labels")
    if foreground is None:
        foreground = [label for label in sorted(labels) if label != 0 and label != ignore_label]
    foreground_labels = tuple(int(label) for label in foreground)
    unknown = sorted(set(foreground_labels) - set(labels))
    if unknown:
        raise ValueError(f"foreground_labels contains labels absent from dataset.labels: {unknown}")

    raw_regions = evaluation.get("regions")
    regions = None
    if raw_regions is not None:
        if not isinstance(raw_regions, dict) or not raw_regions:
            raise ValueError("evaluation.regions must be a non-empty mapping of metric label to source labels")
        regions = {
            int(metric_label): tuple(int(source_label) for source_label in source_labels)
            for metric_label, source_labels in raw_regions.items()
        }
        if any(not source_labels for source_labels in regions.values()):
            raise ValueError("evaluation.regions entries must contain at least one source label")
        if set(regions) != set(foreground_labels):
            raise ValueError("evaluation.regions keys must exactly match evaluation.foreground_labels")
        unknown_sources = sorted(
            {source_label for source_labels in regions.values() for source_label in source_labels}
            - set(channel_labels)
        )
        if unknown_sources:
            raise ValueError(
                f"evaluation.regions contains source labels absent from dataset.channel_labels: {unknown_sources}"
            )

    probabilities_dir = _resolve(base, inputs.get("probabilities_dir"))
    labels_dir = _resolve(base, inputs.get("labels_dir"))
    output_dir = _resolve(base, raw.get("outputs", {}).get("dir"))
    if probabilities_dir is None or labels_dir is None or output_dir is None:
        raise ValueError("inputs.probabilities_dir, inputs.labels_dir, and outputs.dir are required")

    case_ids = inputs.get("case_ids")
    output_mode = str(decoder.get("output_mode", "multiclass"))
    if output_mode.lower() != "multiclass":
        raise ValueError("decoder.rankseg.output_mode must be 'multiclass'")
    cluster_regex = evaluation.get("cluster_regex")
    if cluster_regex is not None:
        cluster_regex = str(cluster_regex)
        try:
            re.compile(cluster_regex)
        except re.error as error:
            raise ValueError(f"evaluation.cluster_regex is invalid: {error}") from error
    segmentation_reader = str(inputs.get("segmentation_reader", "simpleitk")).lower()
    supported_segmentation_readers = {"simpleitk", "nnunetv2_reoriented"}
    if segmentation_reader not in supported_segmentation_readers:
        raise ValueError(
            "inputs.segmentation_reader must be one of "
            f"{sorted(supported_segmentation_readers)}, got {segmentation_reader!r}"
        )
    return DatasetConfig(
        dataset_id=str(dataset["id"]),
        display_name=str(dataset.get("display_name", dataset["id"])),
        labels=labels,
        channel_labels=channel_labels,
        probabilities_dir=probabilities_dir,
        labels_dir=labels_dir,
        output_dir=output_dir,
        foreground_labels=foreground_labels,
        regions=regions,
        probability_key=str(inputs.get("probability_key", "auto")),
        probability_glob=str(inputs.get("probability_glob", "*.npz")),
        label_extension=str(inputs.get("label_extension", ".nii.gz")),
        segmentation_reader=segmentation_reader,
        native_predictions_dir=_resolve(base, inputs.get("native_predictions_dir")),
        native_predictions_postprocessed=bool(inputs.get("native_predictions_postprocessed", True)),
        postprocessing_file=_resolve(base, evaluation.get("postprocessing_file")),
        ignore_label=ignore_label,
        case_ids=None if case_ids is None else tuple(str(case_id) for case_id in case_ids),
        device=str(runtime.get("device", "cuda")),
        rankseg_metric=str(decoder.get("metric", "dice")),
        rankseg_solver=str(decoder.get("solver", "RMA")),
        rankseg_output_mode=output_mode,
        pruning_prob=float(decoder.get("pruning_prob", 0.5)),
        smooth=float(decoder.get("smooth", 0.0)),
        unassigned_policy=str(decoder.get("unassigned_policy", "max_score")),
        bootstrap_samples=int(evaluation.get("bootstrap_samples", 2000)),
        bootstrap_seed=int(evaluation.get("bootstrap_seed", 20260717)),
        cluster_regex=cluster_regex,
        provenance=dict(raw.get("provenance", {})),
    )

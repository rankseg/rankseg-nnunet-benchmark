#!/usr/bin/env python3
"""Render a diagnostic slice from one completed RankSEG benchmark case."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

from rankseg_nnunet_bench.config import load_dataset_config
from rankseg_nnunet_bench.decoders import PairedDecoders
from rankseg_nnunet_bench.io import (
    discover_cases,
    load_probabilities,
    load_segmentation_with_voxel_volume,
    restore_cropped_probabilities,
)
from rankseg_nnunet_bench.metrics import overlap_metrics
from rankseg_nnunet_bench.postprocessing import OfficialNnUNetPostprocessor


LABEL_COLORS = (
    (0.20, 0.95, 0.25, 0.55),
    (1.00, 0.20, 0.85, 0.55),
    (1.00, 0.80, 0.10, 0.55),
    (0.20, 0.75, 1.00, 0.55),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("case_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--slice-mode",
        choices=("disagreement", "target", "argmax", "rankseg"),
        default="disagreement",
        help="Select the axial slice with the largest mask under this criterion.",
    )
    return parser.parse_args()


def normalize_image(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    low, high = np.percentile(finite, (1, 99))
    if high <= low:
        low, high = float(finite.min()), float(finite.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image.astype(np.float32) - low) / (high - low), 0, 1)


def categorical_overlay(labels: np.ndarray, foreground_labels: tuple[int, ...]) -> np.ndarray:
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    for index, label in enumerate(foreground_labels):
        rgba[labels == label] = LABEL_COLORS[index % len(LABEL_COLORS)]
    return rgba


def error_overlay(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*target.shape, 4), dtype=np.float32)
    target_fg = target != 0
    prediction_fg = prediction != 0
    rgba[prediction_fg & ~target_fg] = (1.0, 0.05, 0.05, 0.72)  # false positive
    rgba[~prediction_fg & target_fg] = (0.05, 0.35, 1.0, 0.72)  # false negative
    rgba[prediction_fg & target_fg & (prediction != target)] = (1.0, 0.85, 0.0, 0.78)
    rgba[prediction_fg & target_fg & (prediction == target)] = (0.10, 0.90, 0.20, 0.38)
    return rgba


def disagreement_overlay(argmax: np.ndarray, rankseg: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*argmax.shape, 4), dtype=np.float32)
    argmax_fg = argmax != 0
    rankseg_fg = rankseg != 0
    rgba[rankseg_fg & ~argmax_fg] = (1.0, 0.05, 0.05, 0.78)  # RankSEG-only
    rgba[argmax_fg & ~rankseg_fg] = (0.0, 0.90, 1.0, 0.78)  # argmax-only
    rgba[argmax_fg & rankseg_fg & (argmax != rankseg)] = (1.0, 0.85, 0.0, 0.82)
    return rgba


def foreground_mean_dice(
    prediction: np.ndarray, target: np.ndarray, labels: tuple[int, ...]
) -> float:
    rows = overlap_metrics(prediction, target, (0, *labels))
    values = [float(row["dice"]) for row in rows if int(row["label"]) in labels]
    return float(np.mean(values))


def main() -> int:
    args = parse_args()
    config = replace(load_dataset_config(args.manifest), device="cpu")
    matches = [case for case in discover_cases(config) if case.case_id == args.case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one case {args.case_id!r}, found {len(matches)}")
    case = matches[0]

    probabilities, _ = load_probabilities(case.probabilities, config.probability_key)
    target, volume_per_voxel = load_segmentation_with_voxel_volume(case.label)
    probabilities, _ = restore_cropped_probabilities(
        probabilities,
        probability_path=case.probabilities,
        target_shape=tuple(int(value) for value in target.shape),
        background_channel=config.channel_labels.index(0),
    )
    decoders = PairedDecoders(config)
    tensor = decoders.move_probabilities(probabilities)
    argmax_result = decoders.decode_argmax(tensor)
    rankseg_result = decoders.decode_rankseg(tensor)
    channel_labels = np.asarray(config.channel_labels, dtype=np.int64)
    argmax = channel_labels[argmax_result.prediction]
    rankseg = channel_labels[rankseg_result.prediction]

    if config.postprocessing_file is not None:
        postprocessor = OfficialNnUNetPostprocessor(config.postprocessing_file)
        argmax = postprocessor(argmax, volume_per_voxel=volume_per_voxel)
        rankseg = postprocessor(rankseg, volume_per_voxel=volume_per_voxel)

    image_name = f"{case.case_id}_0000.nii.gz"
    image_candidates = [config.labels_dir.parent / "imagesTr" / image_name]
    image_candidates.extend(
        ancestor
        / "nnUNet_raw_data_base"
        / "nnUNet_raw_data"
        / config.dataset_id
        / "imagesTr"
        / image_name
        for ancestor in config.labels_dir.parents
        if ancestor.name == "v1"
    )
    image_path = next((path for path in image_candidates if path.is_file()), None)
    if image_path is None:
        checked = ", ".join(str(path) for path in image_candidates)
        raise FileNotFoundError(f"Missing source image; checked: {checked}")
    image = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path))).astype(np.float32)
    if image.shape != target.shape:
        raise ValueError(f"Image/label shape mismatch: {image.shape} versus {target.shape}")

    argmax_fg = argmax != 0
    rankseg_fg = rankseg != 0
    slice_masks = {
        "disagreement": (argmax != rankseg) & (argmax_fg | rankseg_fg),
        "target": target != 0,
        "argmax": argmax_fg,
        "rankseg": rankseg_fg,
    }
    selected_mask = slice_masks[args.slice_mode]
    slice_scores = selected_mask.reshape(selected_mask.shape[0], -1).sum(axis=1)
    if not np.any(slice_scores):
        slice_scores = (target != 0).reshape(target.shape[0], -1).sum(axis=1)
    z_index = int(np.argmax(slice_scores))

    image_slice = normalize_image(image[z_index])
    target_slice = target[z_index]
    argmax_slice = argmax[z_index]
    rankseg_slice = rankseg[z_index]
    argmax_dice = foreground_mean_dice(argmax, target, config.foreground_labels)
    rankseg_dice = foreground_mean_dice(rankseg, target, config.foreground_labels)
    argmax_volume = int(np.count_nonzero(argmax_fg))
    rankseg_volume = int(np.count_nonzero(rankseg_fg))
    target_volume = int(np.count_nonzero(target))

    fig, axes = plt.subplots(1, 6, figsize=(22, 4.2), constrained_layout=True)
    panels = (
        ("GT", categorical_overlay(target_slice, config.foreground_labels)),
        ("argmax + GT outline", categorical_overlay(argmax_slice, config.foreground_labels)),
        ("RankSEG + GT outline", categorical_overlay(rankseg_slice, config.foreground_labels)),
        ("argmax error", error_overlay(argmax_slice, target_slice)),
        ("RankSEG error", error_overlay(rankseg_slice, target_slice)),
        ("difference", disagreement_overlay(argmax_slice, rankseg_slice)),
    )
    for index, (axis, (title, overlay)) in enumerate(zip(axes, panels)):
        axis.imshow(image_slice, cmap="gray", interpolation="nearest")
        axis.imshow(overlay, interpolation="nearest")
        if index in (1, 2):
            for label_index, label in enumerate(config.foreground_labels):
                mask = target_slice == label
                if np.any(mask):
                    axis.contour(
                        mask,
                        levels=[0.5],
                        colors=[LABEL_COLORS[label_index % len(LABEL_COLORS)][:3]],
                        linewidths=0.8,
                        linestyles="dashed",
                    )
        axis.set_title(title, fontsize=10)
        axis.axis("off")

    label_legend = ", ".join(
        f"{label}={config.labels[label]}" for label in config.foreground_labels
    )
    fig.suptitle(
        f"{config.dataset_id} / {case.case_id} / axial z={z_index} "
        f"(selected by {args.slice_mode}) | "
        f"Dice {argmax_dice:.4f} → {rankseg_dice:.4f} "
        f"({(rankseg_dice - argmax_dice) * 100:+.2f} pp) | "
        f"foreground voxels target={target_volume:,}, argmax={argmax_volume:,}, "
        f"RankSEG={rankseg_volume:,}\n"
        f"labels: {label_legend} | error: red=FP, blue=FN, yellow=wrong class, green=TP | "
        f"difference: red=RankSEG-only, cyan=argmax-only, yellow=class change",
        fontsize=11,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, facecolor="white")
    plt.close(fig)
    print(
        f"{args.output} z={z_index} argmax_dice={argmax_dice:.8f} "
        f"rankseg_dice={rankseg_dice:.8f} delta_pp={(rankseg_dice-argmax_dice)*100:.5f} "
        f"volumes={target_volume}/{argmax_volume}/{rankseg_volume}"
    )
    if len(config.foreground_labels) == 1:
        foreground_label = config.foreground_labels[0]
        foreground_channel = config.channel_labels.index(foreground_label)
        foreground_probability = probabilities[foreground_channel]
        selected_probability = foreground_probability[rankseg == foreground_label]
        print(
            "binary_probability_diagnostic "
            f"global_max={float(foreground_probability.max()):.8f} "
            f"probability_sum={float(foreground_probability.sum()):.4f} "
            f"slice_max={float(foreground_probability[z_index].max()):.8f} "
            f"slice_argmax_voxels={int(np.count_nonzero(argmax_slice))} "
            f"slice_rankseg_voxels={int(np.count_nonzero(rankseg_slice))} "
            f"rankseg_selected_min={float(selected_probability.min()):.8f} "
            f"rankseg_selected_median={float(np.median(selected_probability)):.8f} "
            f"rankseg_selected_max={float(selected_probability.max()):.8f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

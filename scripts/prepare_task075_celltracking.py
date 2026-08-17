#!/usr/bin/env python3
"""Reproduce nnU-Net v1 Task075 from the official CTC training archives."""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import pickle

import numpy as np
from skimage.io import imread
import SimpleITK as sitk

from rankseg_nnunet_bench.oof import validate_splits


TASK_NAME = "Task075_Fluo_C3DH_A549_ManAndSim"
SPACING_XYZ = (0.126, 0.126, 1.0)


def _resolve_dataset_root(path: Path, expected_name: str) -> Path:
    path = path.resolve()
    candidates = [path, path / expected_name]
    matches = [candidate for candidate in candidates if (candidate / "01").is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Could not identify {expected_name} root below {path}; expected sequence folder 01"
        )
    return matches[0]


def _write_case(image_path: Path, label_path: Path, image_output: Path, label_output: Path) -> None:
    image = np.asarray(imread(image_path), dtype=np.float32)
    label = np.asarray(imread(label_path))
    if image.shape != label.shape:
        raise ValueError(f"Image/label shape mismatch: {image_path} {image.shape} vs {label_path} {label.shape}")
    label = (label > 0).astype(np.uint8)

    image_itk = sitk.GetImageFromArray(image)
    label_itk = sitk.GetImageFromArray(label)
    image_itk.SetSpacing(SPACING_XYZ)
    label_itk.SetSpacing(SPACING_XYZ)
    sitk.WriteImage(image_itk, str(image_output))
    sitk.WriteImage(label_itk, str(label_output))


def prepare_task075(*, manual_source: Path, simulated_source: Path, output_root: Path, splits_file: Path) -> None:
    sources = [
        _resolve_dataset_root(manual_source, "Fluo-C3DH-A549"),
        _resolve_dataset_root(simulated_source, "Fluo-C3DH-A549-SIM"),
    ]
    task_dir = output_root.resolve() / TASK_NAME
    images = task_dir / "imagesTr"
    labels = task_dir / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    case_ids: list[str] = []
    for source in sources:
        source_name = source.name
        for sequence in ("01", "02"):
            for image_path in sorted((source / sequence).glob("*.tif")):
                label_path = source / f"{sequence}_GT" / "SEG" / f"man_seg{image_path.name[1:]}"
                if not label_path.is_file():
                    continue
                case_id = f"{source_name}__{sequence}__{image_path.stem}"
                image_output = images / f"{case_id}_0000.nii.gz"
                label_output = labels / f"{case_id}.nii.gz"
                if image_output.exists() or label_output.exists():
                    raise FileExistsError(f"Refusing to overwrite an existing converted case: {case_id}")
                _write_case(image_path, label_path, image_output, label_output)
                case_ids.append(case_id)

    if not case_ids:
        raise FileNotFoundError("No annotated Task075 frames were found")
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Converted Task075 case IDs are not unique")

    dataset = OrderedDict(
        name="Fluo_C3DH_A549_ManAndSim",
        description="Official Cell Tracking Challenge A549 manual and simulated training annotations",
        tensorImageSize="4D",
        reference="https://celltrackingchallenge.net/3d-datasets/",
        licence="See the Cell Tracking Challenge conditions of use",
        release="0.0",
        modality={"0": "BF"},
        labels={"0": "background", "1": "cell"},
        numTraining=len(case_ids),
        numTest=0,
        training=[
            {"image": f"./imagesTr/{case_id}.nii.gz", "label": f"./labelsTr/{case_id}.nii.gz"}
            for case_id in case_ids
        ],
        test=[],
    )
    (task_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    manual_01 = sorted(case for case in case_ids if case.startswith("Fluo-C3DH-A549__01__"))
    manual_02 = sorted(case for case in case_ids if case.startswith("Fluo-C3DH-A549__02__"))
    simulated_01 = sorted(case for case in case_ids if case.startswith("Fluo-C3DH-A549-SIM__01__"))
    simulated_02 = sorted(case for case in case_ids if case.startswith("Fluo-C3DH-A549-SIM__02__"))
    groups = (manual_01, manual_02, simulated_01, simulated_02)
    if any(not group for group in groups):
        raise ValueError("Each Task075 source sequence must contribute at least one annotated frame")
    splits = [
        {"train": manual_01 + simulated_01 + simulated_02, "val": manual_02},
        {"train": manual_02 + simulated_01 + simulated_02, "val": manual_01},
        {"train": manual_01 + manual_02 + simulated_01, "val": simulated_02},
        {"train": manual_01 + manual_02 + simulated_02, "val": simulated_01},
    ]
    validate_splits(splits, set(case_ids))
    splits_file = splits_file.resolve()
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    if splits_file.exists():
        raise FileExistsError(f"Refusing to overwrite existing split file: {splits_file}")
    with splits_file.open("wb") as handle:
        pickle.dump(splits, handle)

    manifest = {
        "task": TASK_NAME,
        "cases": len(case_ids),
        "folds": len(splits),
        "group_case_counts": {
            "manual_01": len(manual_01),
            "manual_02": len(manual_02),
            "simulated_01": len(simulated_01),
            "simulated_02": len(simulated_02),
        },
        "manual_source": str(sources[0]),
        "simulated_source": str(sources[1]),
        "split_protocol": "exact nnU-Net v1 Task075 source-sequence holdout protocol",
    }
    (task_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-source", type=Path, required=True)
    parser.add_argument("--simulated-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    args = parser.parse_args()
    prepare_task075(
        manual_source=args.manual_source,
        simulated_source=args.simulated_source,
        output_root=args.output_root,
        splits_file=args.splits_file,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Convert the official ACDC training release and reproduce nnU-Net v1 patient splits."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import KFold


def _link_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f"Existing file differs in size: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _patient_directories(source_dir: Path) -> list[Path]:
    directories = sorted(
        path for path in source_dir.rglob("patient[0-9][0-9][0-9]") if path.is_dir()
    )
    by_name: dict[str, Path] = {}
    for directory in directories:
        if directory.name in by_name:
            raise ValueError(f"Duplicate patient directory {directory.name}: {directory}")
        by_name[directory.name] = directory
    if len(by_name) != 100 or sorted(by_name) != [f"patient{index:03d}" for index in range(1, 101)]:
        raise ValueError(f"Expected official ACDC patients patient001..patient100, found {len(by_name)}")
    return [by_name[name] for name in sorted(by_name)]


def _validate_pair(image_path: Path, label_path: Path) -> tuple[tuple[int, ...], tuple[int, ...]]:
    image = sitk.ReadImage(str(image_path))
    label = sitk.ReadImage(str(label_path))
    if image.GetSize() != label.GetSize():
        raise ValueError(f"Image/label size mismatch: {image_path.name}")
    if not np.allclose(image.GetSpacing(), label.GetSpacing()):
        raise ValueError(f"Image/label spacing mismatch: {image_path.name}")
    values = tuple(int(value) for value in np.unique(sitk.GetArrayViewFromImage(label)))
    if not set(values) <= {0, 1, 2, 3}:
        raise ValueError(f"Unexpected labels {values}: {label_path}")
    return tuple(int(value) for value in image.GetSize()), values


def convert(source_dir: Path, output_dir: Path, splits_file: Path) -> None:
    images_dir = output_dir / "imagesTr"
    labels_dir = output_dir / "labelsTr"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    case_ids: list[str] = []
    audit_rows: list[dict[str, object]] = []
    for patient_dir in _patient_directories(source_dir):
        patient = patient_dir.name
        images = sorted(
            path
            for path in patient_dir.glob(f"{patient}_frame*.nii.gz")
            if not path.name.endswith("_gt.nii.gz")
        )
        if len(images) != 2:
            raise ValueError(f"Expected exactly ED/ES frame images for {patient}, found {len(images)}")
        for image_path in images:
            case_id = image_path.name[:-7]
            label_path = image_path.with_name(f"{case_id}_gt.nii.gz")
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing label for {image_path}")
            size, values = _validate_pair(image_path, label_path)
            _link_or_copy(image_path, images_dir / f"{case_id}_0000.nii.gz")
            _link_or_copy(label_path, labels_dir / f"{case_id}.nii.gz")
            case_ids.append(case_id)
            audit_rows.append(
                {
                    "patient_id": patient,
                    "case_id": case_id,
                    "source_image": str(image_path.resolve()),
                    "source_label": str(label_path.resolve()),
                    "size": "x".join(str(value) for value in size),
                    "label_values": ",".join(str(value) for value in values),
                }
            )

    if len(case_ids) != 200 or len(set(case_ids)) != 200:
        raise ValueError(f"Expected 200 unique ED/ES cases, found {len(set(case_ids))}")

    dataset = OrderedDict(
        name="ACDC",
        description="Automatic Cardiac Diagnosis Challenge cine MRI segmentation",
        tensorImageSize="4D",
        reference="https://www.creatis.insa-lyon.fr/Challenge/acdc/",
        licence="ACDC challenge terms",
        release="official training database",
        modality={"0": "MRI"},
        labels={"0": "background", "1": "RV", "2": "MLV", "3": "LVC"},
        numTraining=len(case_ids),
        numTest=0,
        training=[
            {"image": f"./imagesTr/{case_id}.nii.gz", "label": f"./labelsTr/{case_id}.nii.gz"}
            for case_id in sorted(case_ids)
        ],
        test=[],
    )
    (output_dir / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")

    patients = np.asarray(sorted({case_id[:10] for case_id in case_ids}))
    splitter = KFold(n_splits=5, shuffle=True, random_state=12345)
    splits: list[OrderedDict[str, list[str]]] = []
    patient_fold: dict[str, int] = {}
    for fold, (train_indices, val_indices) in enumerate(splitter.split(patients)):
        train_patients = set(patients[train_indices])
        val_patients = set(patients[val_indices])
        splits.append(
            OrderedDict(
                train=sorted(case_id for case_id in case_ids if case_id[:10] in train_patients),
                val=sorted(case_id for case_id in case_ids if case_id[:10] in val_patients),
            )
        )
        patient_fold.update({str(patient): fold for patient in val_patients})
    if len(patient_fold) != 100 or any(len(split["val"]) != 40 for split in splits):
        raise RuntimeError("Patient-clustered split validation failed")
    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with splits_file.open("wb") as handle:
        pickle.dump(splits, handle)

    for row in audit_rows:
        row["fold"] = patient_fold[str(row["patient_id"])]
    audit_path = output_dir / "conversion_and_split_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(audit_rows, key=lambda row: str(row["case_id"])))
    print(f"Prepared {len(case_ids)} cases from {len(patients)} patients")
    print(f"Patient-clustered splits: {splits_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    args = parser.parse_args()
    convert(args.source_dir.resolve(), args.output_dir.resolve(), args.splits_file.resolve())


if __name__ == "__main__":
    main()

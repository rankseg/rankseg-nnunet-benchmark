#!/usr/bin/env python3
"""Convert the official 40-patient SegTHOR training archive to nnU-Net v1 layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


EXPECTED_PATIENTS = 40
ALLOWED_LABELS = {0, 1, 2, 3, 4}


def ensure_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    os.symlink(source.resolve(), destination)


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), rtol=0, atol=1e-6)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), rtol=0, atol=1e-6)
        and np.allclose(first.GetDirection(), second.GetDirection(), rtol=0, atol=1e-6)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    labels = sorted(args.source_dir.rglob("GT.nii.gz"))
    if len(labels) != EXPECTED_PATIENTS:
        raise ValueError(f"Expected {EXPECTED_PATIENTS} GT.nii.gz files, found {len(labels)}")

    images_dir = args.output_dir / "imagesTr"
    labels_dir = args.output_dir / "labelsTr"
    images_test_dir = args.output_dir / "imagesTs"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    images_test_dir.mkdir(parents=True, exist_ok=True)

    cases: list[str] = []
    observed_labels: set[int] = set()
    records: list[dict] = []
    for label_path in labels:
        case_id = label_path.parent.name
        image_path = label_path.parent / f"{case_id}.nii.gz"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing CT for {case_id}: {image_path}")
        if case_id in cases:
            raise ValueError(f"Duplicate patient ID: {case_id}")

        image = sitk.ReadImage(str(image_path))
        label = sitk.ReadImage(str(label_path))
        if not same_geometry(image, label):
            raise ValueError(f"Image/label geometry mismatch for {case_id}")
        case_labels = {int(value) for value in np.unique(sitk.GetArrayViewFromImage(label))}
        if not case_labels <= ALLOWED_LABELS:
            raise ValueError(f"Unexpected label values for {case_id}: {sorted(case_labels)}")

        ensure_symlink(image_path, images_dir / f"{case_id}_0000.nii.gz")
        ensure_symlink(label_path, labels_dir / f"{case_id}.nii.gz")
        cases.append(case_id)
        observed_labels.update(case_labels)
        records.append(
            {
                "case_id": case_id,
                "image": str(image_path.resolve()),
                "label": str(label_path.resolve()),
                "size_xyz": list(image.GetSize()),
                "spacing_xyz": list(image.GetSpacing()),
                "labels_present": sorted(case_labels),
            }
        )

    cases.sort()
    if observed_labels != ALLOWED_LABELS:
        raise ValueError(f"Dataset label set mismatch: {sorted(observed_labels)}")

    dataset = {
        "name": "SegTHOR",
        "description": "Official SegTHOR training cohort",
        "tensorImageSize": "4D",
        "reference": "https://codalab.lisn.upsaclay.fr/competitions/21012",
        "licence": "See official challenge terms",
        "release": "official 40-patient training archive",
        "modality": {"0": "CT"},
        "labels": {
            "0": "background",
            "1": "esophagus",
            "2": "heart",
            "3": "trachea",
            "4": "aorta",
        },
        "numTraining": len(cases),
        "numTest": 0,
        "training": [
            {"image": f"./imagesTr/{case_id}.nii.gz", "label": f"./labelsTr/{case_id}.nii.gz"}
            for case_id in cases
        ],
        "test": [],
    }
    (args.output_dir / "dataset.json").write_text(
        json.dumps(dataset, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "source_dir": str(args.source_dir.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "patients": len(cases),
        "observed_labels": sorted(observed_labels),
        "cases": records,
    }
    provenance_path = args.output_dir / "conversion_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(provenance_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

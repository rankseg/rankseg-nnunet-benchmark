#!/usr/bin/env python3
"""Reproduce the two Task005 ground-truth corrections used by the official nnU-Net v1 release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk


CORRECTIONS = {
    "prostate_18": 45166,
    "prostate_32": 25688,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument("--official-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official = json.loads(args.official_summary.read_text(encoding="utf-8"))
    official_cases = {
        Path(result["reference"]).name.removesuffix(".nii.gz"): result
        for result in official["results"]["all"]
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    source_files = sorted(args.source_labels.glob("*.nii.gz"))
    if len(source_files) != 32:
        raise ValueError(f"Expected 32 Task005 labels, found {len(source_files)}")
    for source in source_files:
        case_id = source.name.removesuffix(".nii.gz")
        destination = args.output_dir / source.name
        if case_id not in CORRECTIONS:
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise FileExistsError(f"Existing symlink points elsewhere: {destination}")
            elif destination.exists():
                raise FileExistsError(f"Refusing to replace existing non-symlink: {destination}")
            else:
                os.symlink(source.resolve(), destination)
            continue

        expected_voxels = CORRECTIONS[case_id]
        image = sitk.ReadImage(str(source))
        array = sitk.GetArrayFromImage(image)
        source_class_1 = int(np.count_nonzero(array == 1))
        source_class_2 = int(np.count_nonzero(array == 2))
        if source_class_1 != expected_voxels or source_class_2 != 0:
            raise ValueError(
                f"Unexpected source labels for {case_id}: class1={source_class_1}, class2={source_class_2}"
            )
        official_case = official_cases[case_id]
        official_class_1 = int(official_case["1"]["Total Positives Reference"])
        official_class_2 = int(official_case["2"]["Total Positives Reference"])
        if (official_class_1, official_class_2) != (0, expected_voxels):
            raise ValueError(
                f"Official release summary does not support correction for {case_id}: "
                f"class1={official_class_1}, class2={official_class_2}"
            )

        corrected = array.copy()
        corrected[corrected == 1] = 2
        corrected_image = sitk.GetImageFromArray(corrected)
        corrected_image.CopyInformation(image)
        sitk.WriteImage(corrected_image, str(destination), useCompression=True)
        records.append(
            {
                "case_id": case_id,
                "operation": "label 1 -> label 2",
                "voxels_changed": expected_voxels,
                "source_sha256": sha256(source),
                "corrected_sha256": sha256(destination),
                "official_reference_class_1_voxels": official_class_1,
                "official_reference_class_2_voxels": official_class_2,
            }
        )

    provenance = {
        "task": "Task005_Prostate",
        "reason": "Match the ground truths used to evaluate the author-published nnU-Net v1 model release.",
        "source_labels": str(args.source_labels.resolve()),
        "official_release_summary": str(args.official_summary.resolve()),
        "official_release_summary_sha256": sha256(args.official_summary),
        "corrections": records,
    }
    (args.output_dir / "corrections.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output_dir / "corrections.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

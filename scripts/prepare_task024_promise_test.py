#!/usr/bin/env python3
"""Convert the publicly released former-hidden PROMISE12 test cohort for inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


EXPECTED_CASE_IDS = tuple(f"Case{index:02d}" for index in range(30))


def _geometry(image: sitk.Image) -> tuple[object, ...]:
    return (image.GetSize(), image.GetSpacing(), image.GetOrigin(), image.GetDirection())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_task024_test(*, source_dir: Path, output_dir: Path, source_archive: Path) -> None:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    observed_images = {path.stem for path in source_dir.glob("Case[0-9][0-9].mhd")}
    observed_labels = {
        path.name.removesuffix("_segmentation.mhd")
        for path in source_dir.glob("Case[0-9][0-9]_segmentation.mhd")
    }
    expected = set(EXPECTED_CASE_IDS)
    if observed_images != expected or observed_labels != expected:
        raise ValueError(
            "PROMISE12 former-hidden test cohort must contain exactly Case00--Case29; "
            f"images missing={sorted(expected - observed_images)}, extra={sorted(observed_images - expected)}, "
            f"labels missing={sorted(expected - observed_labels)}, extra={sorted(observed_labels - expected)}"
        )

    cases: list[dict[str, object]] = []
    for case_id in EXPECTED_CASE_IDS:
        image_source = source_dir / f"{case_id}.mhd"
        label_source = source_dir / f"{case_id}_segmentation.mhd"
        image_output = images_dir / f"{case_id}_0000.nii.gz"
        label_output = labels_dir / f"{case_id}.nii.gz"
        if image_output.exists() or label_output.exists():
            raise FileExistsError(f"Refusing to overwrite converted PROMISE12 case: {case_id}")

        image = sitk.ReadImage(str(image_source))
        label = sitk.ReadImage(str(label_source))
        if _geometry(image) != _geometry(label):
            raise ValueError(f"Image/label geometry mismatch for {case_id}")
        label_array = sitk.GetArrayFromImage(label)
        values = sorted(int(value) for value in np.unique(label_array))
        if not set(values) <= {0, 1}:
            raise ValueError(f"Unexpected PROMISE12 labels for {case_id}: {values}")
        if 1 not in values:
            raise ValueError(f"PROMISE12 test mask has no prostate foreground: {case_id}")

        sitk.WriteImage(image, str(image_output))
        sitk.WriteImage(label, str(label_output))
        cases.append(
            {
                "case_id": case_id,
                "size_xyz": list(image.GetSize()),
                "spacing_xyz": list(image.GetSpacing()),
                "foreground_voxels": int(np.count_nonzero(label_array)),
            }
        )

    manifest = {
        "task": "Task024_Promise",
        "cohort": "former hidden PROMISE12 challenge test set with publicly released masks",
        "cases": len(cases),
        "case_ids": list(EXPECTED_CASE_IDS),
        "source_dir": str(source_dir),
        "source_archive": str(source_archive.resolve()),
        "source_archive_size_bytes": source_archive.stat().st_size,
        "source_archive_md5": "823ef560f4a54083348e58d67403e4bb",
        "source_archive_sha256": _sha256(source_archive.resolve()),
        "cases_geometry_and_foreground": cases,
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    args = parser.parse_args()
    prepare_task024_test(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        source_archive=args.source_archive,
    )


if __name__ == "__main__":
    main()

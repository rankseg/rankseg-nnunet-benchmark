#!/usr/bin/env python3
"""Prepare the 20 labeled CHAOS CT cases for external nnU-Net v2 inference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import dicom2nifti
import numpy as np
from PIL import Image
import SimpleITK as sitk


EXPECTED_ARCHIVE_BYTES = 890_771_694
EXPECTED_ARCHIVE_MD5 = "df21053002a1cc86df918a87da3b2c19"


def file_hash(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_chaos_ct_mask(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No ground-truth PNGs in {folder}")
    values = np.stack([np.asarray(Image.open(path)) for path in files], axis=0)
    observed = set(int(value) for value in np.unique(values))
    # The official archive contains binary PNGs encoded as 0/1 (some CHAOS
    # mirrors/export tools materialize the foreground as 255), so accept both
    # without relying on a specific nonzero intensity.
    if not observed <= {0, 1, 255}:
        raise ValueError(f"Unexpected CHAOS CT mask intensities in {folder}: {sorted(observed)}")
    # This is the exact z ordering used by the publisher-supplied nnU-Net v1
    # CHAOS converter (load_png_stack returns the lexicographic stack reversed).
    # Label 5 is the liver channel in TotalSegmentator Dataset291.
    return np.where(values[::-1] > 0, 5, 0).astype(np.uint8)


def geometry(image: sitk.Image) -> dict[str, object]:
    return {
        "size_xyz": list(image.GetSize()),
        "spacing_xyz": list(image.GetSpacing()),
        "origin_xyz": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    archive = args.source_archive.resolve()
    output = args.output_dir.resolve()
    images = output / "imagesTs"
    labels = output / "labelsTs"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    if archive.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError(f"Unexpected CHAOS archive size: {archive.stat().st_size}")
    archive_md5 = file_hash(archive, "md5")  # noqa: S324 - publisher integrity hash
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError(f"CHAOS archive MD5 mismatch: {archive_md5}")

    patient_dirs = sorted((path for path in source.iterdir() if path.is_dir()), key=lambda path: int(path.name))
    if len(patient_dirs) != 20:
        raise ValueError(f"Expected 20 labeled CHAOS CT patients, found {len(patient_dirs)}")

    records: list[dict[str, object]] = []
    for index, patient_dir in enumerate(patient_dirs, start=1):
        case_id = f"CHAOSCT_{int(patient_dir.name):03d}"
        image_path = images / f"{case_id}_0000.nii.gz"
        label_path = labels / f"{case_id}.nii.gz"

        if not (image_path.is_file() and label_path.is_file()):
            dicom2nifti.convert_dicom.dicom_series_to_nifti(
                str(patient_dir / "DICOM_anon"),
                str(image_path),
                reorient_nifti=False,
            )
        image = sitk.ReadImage(str(image_path))
        mask = load_chaos_ct_mask(patient_dir / "Ground")
        expected_shape = tuple(reversed(image.GetSize()))
        if mask.shape != expected_shape:
            raise ValueError(f"{case_id}: mask shape {mask.shape} != image shape {expected_shape}")
        foreground_voxels = int(np.count_nonzero(mask == 5))
        if foreground_voxels == 0:
            raise ValueError(f"{case_id}: empty liver ground truth")

        if not label_path.is_file():
            label_image = sitk.GetImageFromArray(mask)
            label_image.CopyInformation(image)
            sitk.WriteImage(label_image, str(label_path))
        if sitk.ReadImage(str(label_path)).GetSize() != image.GetSize():
            raise ValueError(f"{case_id}: written label geometry is invalid")

        records.append(
            {
                "case_id": case_id,
                "source_patient_id": patient_dir.name,
                "geometry": geometry(image),
                "liver_voxels": foreground_voxels,
                "image_sha256": file_hash(image_path, "sha256"),
                "label_sha256": file_hash(label_path, "sha256"),
            }
        )
        print(f"[{index:02d}/20] {case_id}", flush=True)

    dataset = {
        "name": "CHAOS_CT_external_TotalSegmentator_v2",
        "description": "Independent CHAOS CT liver cohort evaluated with TotalSegmentator Dataset291",
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "liver": 5},
        "numTraining": 0,
        "numTest": len(records),
        "file_ending": ".nii.gz",
        "test": [f"./imagesTs/{record['case_id']}.nii.gz" for record in records],
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "source_archive": str(archive),
        "source_archive_bytes": archive.stat().st_size,
        "source_archive_md5": archive_md5,
        "source_doi": "https://doi.org/10.5281/zenodo.3362845",
        "source_license": "CC-BY-NC-SA-4.0",
        "conversion": "dicom2nifti reorient_nifti=False; official nnU-Net CHAOS reversed PNG z-stack",
        "model_label_for_liver": 5,
        "cases": len(records),
        "records": records,
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output / "conversion_manifest.json")


if __name__ == "__main__":
    main()

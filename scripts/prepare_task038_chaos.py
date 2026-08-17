#!/usr/bin/env python3
"""Convert official CHAOS MRI training data exactly as nnU-Net v1 Task038 Variant 2."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import dicom2nifti
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.data_splitting import get_split_deterministic
from nnunet.utilities.sitk_stuff import copy_geometry
from PIL import Image


def md5(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - publisher-provided integrity identifier
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_png_stack(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No ground-truth PNGs in {folder}")
    # This matches nnU-Net v1's load_png_stack. The caller reverses it once more,
    # reproducing the original Task038 converter's orientation exactly.
    return np.stack([np.asarray(Image.open(path)) for path in files], axis=0)[::-1]


def convert_mr_segmentation(values: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape, dtype=np.uint8)
    result[(values > 55) & (values <= 70)] = 1
    result[(values > 110) & (values <= 135)] = 2
    result[(values > 175) & (values <= 200)] = 3
    result[(values > 240) & (values <= 255)] = 4
    return result


def geometry(image: sitk.Image) -> dict[str, object]:
    return {
        "size": list(image.GetSize()),
        "spacing": list(image.GetSpacing()),
        "origin": list(image.GetOrigin()),
        "direction": list(image.GetDirection()),
    }


def same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and np.allclose(first.GetSpacing(), second.GetSpacing(), atol=1e-6)
        and np.allclose(first.GetOrigin(), second.GetOrigin(), atol=1e-6)
        and np.allclose(first.GetDirection(), second.GetDirection(), atol=1e-6)
    )


def dicom_to_nifti(source: Path, destination: Path) -> sitk.Image:
    destination.parent.mkdir(parents=True, exist_ok=True)
    dicom2nifti.convert_dicom.dicom_series_to_nifti(
        str(source), str(destination), reorient_nifti=False
    )
    return sitk.ReadImage(str(destination))


def write_label(segmentation: np.ndarray, reference: sitk.Image, destination: Path) -> None:
    if tuple(segmentation.shape) != tuple(reversed(reference.GetSize())):
        raise ValueError(
            f"Segmentation/image shape mismatch for {destination.name}: "
            f"{segmentation.shape} versus {tuple(reversed(reference.GetSize()))}"
        )
    image = sitk.GetImageFromArray(segmentation)
    image = copy_geometry(image, reference)
    sitk.WriteImage(image, str(destination))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits-file", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_dir.resolve()
    output = args.output_dir.resolve()
    images = output / "imagesTr"
    labels = output / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    archive = args.source_archive.resolve()
    if archive.stat().st_size != 890_771_694:
        raise ValueError("CHAOS training archive size does not match the official Zenodo record")
    archive_md5 = md5(archive)
    if archive_md5 != "df21053002a1cc86df918a87da3b2c19":
        raise ValueError(f"CHAOS training archive MD5 mismatch: {archive_md5}")

    mr_root = source / "MR"
    patients = sorted(path.name for path in mr_root.iterdir() if path.is_dir())
    if len(patients) != 20:
        raise ValueError(f"Expected 20 CHAOS MRI patients, found {len(patients)}: {patients}")

    records: list[dict[str, object]] = []
    for index, patient in enumerate(patients, start=1):
        patient_dir = mr_root / patient
        t1_ground = convert_mr_segmentation(
            load_png_stack(patient_dir / "T1DUAL" / "Ground")[::-1]
        )
        t2_ground = convert_mr_segmentation(
            load_png_stack(patient_dir / "T2SPIR" / "Ground")[::-1]
        )

        destinations = {
            "T1_in": images / f"T1_in_{patient}_0000.nii.gz",
            "T1_out": images / f"T1_out_{patient}_0000.nii.gz",
            "T2": images / f"T2_{patient}_0000.nii.gz",
        }
        t1_in = dicom_to_nifti(
            patient_dir / "T1DUAL" / "DICOM_anon" / "InPhase", destinations["T1_in"]
        )
        t1_out = dicom_to_nifti(
            patient_dir / "T1DUAL" / "DICOM_anon" / "OutPhase", destinations["T1_out"]
        )
        t2 = dicom_to_nifti(patient_dir / "T2SPIR" / "DICOM_anon", destinations["T2"])
        if not same_geometry(t1_in, t1_out):
            raise ValueError(f"T1 in/out geometry differs for CHAOS patient {patient}")

        # The official Variant 2 converter uses the T1 out-phase geometry for both
        # phase labels. In/out geometry is asserted equal immediately above.
        write_label(t1_ground, t1_out, labels / f"T1_in_{patient}.nii.gz")
        write_label(t1_ground, t1_out, labels / f"T1_out_{patient}.nii.gz")
        write_label(t2_ground, t2, labels / f"T2_{patient}.nii.gz")

        for sequence, image in (("T1_in", t1_in), ("T1_out", t1_out), ("T2", t2)):
            case_id = f"{sequence}_{patient}"
            label_array = t1_ground if sequence.startswith("T1") else t2_ground
            unique, counts = np.unique(label_array, return_counts=True)
            if set(unique.tolist()) - {0, 1, 2, 3, 4}:
                raise ValueError(f"Unexpected labels in {case_id}: {unique.tolist()}")
            records.append(
                {
                    "case_id": case_id,
                    "patient_id": patient,
                    "sequence": sequence,
                    "geometry": geometry(image),
                    "label_voxels": {
                        str(int(label)): int(count) for label, count in zip(unique, counts)
                    },
                }
            )
        print(f"[{index:02d}/20] patient {patient}", flush=True)

    dataset = {
        "name": "CHAOS_Task_3_5_Variant2",
        "description": "Official CHAOS MRI training cohort converted as nnU-Net v1 Task038 Variant 2",
        "tensorImageSize": "4D",
        "reference": "https://doi.org/10.5281/zenodo.3431873",
        "licence": "CC-BY-NC-SA-4.0",
        "release": "v1.03",
        "modality": {"0": "MRI"},
        "labels": {
            "0": "background",
            "1": "liver",
            "2": "right kidney",
            "3": "left kidney",
            "4": "spleen",
        },
        "numTraining": 60,
        "numTest": 0,
        "training": [
            {"image": f"./imagesTr/{record['case_id']}.nii.gz", "label": f"./labelsTr/{record['case_id']}.nii.gz"}
            for record in records
        ],
        "test": [],
    }
    (output / "dataset.json").write_text(json.dumps(dataset, indent=2) + "\n")

    splits = []
    assignments: dict[str, int] = {}
    for fold in range(5):
        train_patients, val_patients = get_split_deterministic(patients, fold, 5, 12345)
        train = [f"{sequence}_{patient}" for sequence in ("T2", "T1_in", "T1_out") for patient in train_patients]
        val = [f"{sequence}_{patient}" for sequence in ("T2", "T1_in", "T1_out") for patient in val_patients]
        splits.append({"train": np.asarray(train), "val": np.asarray(val)})
        for case_id in val:
            if case_id in assignments:
                raise ValueError(f"Duplicate OOF assignment: {case_id}")
            assignments[case_id] = fold
    if set(assignments) != {str(record["case_id"]) for record in records}:
        raise ValueError("OOF splits do not cover every converted CHAOS case exactly once")
    args.splits_file.parent.mkdir(parents=True, exist_ok=True)
    with args.splits_file.open("wb") as handle:
        pickle.dump(splits, handle)

    assignment_text = "\n".join(f"{case_id},{assignments[case_id]}" for case_id in sorted(assignments)) + "\n"
    manifest = {
        "source_archive": str(archive),
        "source_size_bytes": archive.stat().st_size,
        "source_md5": archive_md5,
        "source_doi": "https://doi.org/10.5281/zenodo.3431873",
        "conversion": "nnU-Net v1 Task038 CHAOS Variant 2 semantics",
        "patients": len(patients),
        "cases": len(records),
        "cases_per_patient": 3,
        "split": "patient-level get_split_deterministic(..., num_splits=5, random_state=12345)",
        "oof_assignment_sha256": hashlib.sha256(assignment_text.encode()).hexdigest(),
        "records": records,
    }
    (output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "conversion_manifest.json")


if __name__ == "__main__":
    main()

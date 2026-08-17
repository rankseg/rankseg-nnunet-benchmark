import csv
import json
import os
from pathlib import Path

import pytest

from rankseg_nnunet_bench.oof import (
    discover_nnunet_cases,
    prepare_oof_ensemble_v1,
    prepare_oof_v1,
    validate_splits,
)


def test_prepare_oof_assigns_every_case_once(tmp_path: Path):
    images = tmp_path / "Task999" / "imagesTr"
    images.mkdir(parents=True)
    for index in range(10):
        (images / f"case_{index:03d}_0000.nii.gz").write_bytes(b"not-a-real-nifti")
        (images / f"case_{index:03d}_0001.nii.gz").write_bytes(b"not-a-real-nifti")

    output = tmp_path / "oof"
    metadata = prepare_oof_v1(task="Task999", images_dir=images, output_dir=output)
    assert metadata.is_file()
    with (output / "fold_assignments.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert len({row["case_id"] for row in rows}) == 10
    for row in rows:
        fold = row["fold"]
        assert (output / "fold_inputs" / f"fold_{fold}" / f"{row['case_id']}_0000.nii.gz").is_symlink()
    script = (output / "run_oof_inference.sh").read_text(encoding="utf-8")
    assert script.count("nnUNet_predict") == 5
    assert script.count("--save_npz") == 5
    assert script.count("--num_threads_nifti_save 2") == 5
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1" in script


def test_discover_cases_rejects_nonconsecutive_channels(tmp_path: Path):
    images = tmp_path / "imagesTr"
    images.mkdir()
    (images / "case_001_0001.nii.gz").write_bytes(b"not-a-real-nifti")

    with pytest.raises(ValueError, match="start at 0000"):
        discover_nnunet_cases(images)


def test_validate_splits_rejects_duplicate_ids():
    cases = {f"case_{index}" for index in range(5)}
    splits = [
        {"train": sorted(cases - {f"case_{fold}"}), "val": [f"case_{fold}"]}
        for fold in range(5)
    ]
    splits[0]["train"].append(splits[0]["train"][0])

    with pytest.raises(ValueError, match="duplicate training"):
        validate_splits(splits, cases)


def test_prepare_oof_accepts_official_two_fold_sequence_split(tmp_path: Path):
    images = tmp_path / "Task999" / "imagesTr"
    images.mkdir(parents=True)
    for sequence in ("01", "02"):
        for timestep in range(3):
            (images / f"{sequence}_t{timestep:03d}_0000.nii.gz").write_bytes(b"not-a-real-nifti")
    splits = [
        {
            "train": [f"01_t{index:03d}" for index in range(3)],
            "val": [f"02_t{index:03d}" for index in range(3)],
        },
        {
            "train": [f"02_t{index:03d}" for index in range(3)],
            "val": [f"01_t{index:03d}" for index in range(3)],
        },
    ]
    splits_file = tmp_path / "splits.pkl"
    import pickle

    with splits_file.open("wb") as handle:
        pickle.dump(splits, handle)

    metadata_path = prepare_oof_v1(
        task="Task999",
        images_dir=images,
        output_dir=tmp_path / "oof",
        splits_file=splits_file,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    script = (tmp_path / "oof" / "run_oof_inference.sh").read_text(encoding="utf-8")
    assert metadata["folds"] == 2
    assert script.count("nnUNet_predict") == 2


def test_prepare_oof_uses_symlink_entry_name_for_channel_discovery(tmp_path: Path):
    sources = tmp_path / "official_source"
    images = tmp_path / "Task999" / "imagesTr"
    sources.mkdir()
    images.mkdir(parents=True)
    for index in range(10):
        source = sources / f"Patient_{index:02d}.nii.gz"
        source.write_bytes(b"not-a-real-nifti")
        os.symlink(source, images / f"Patient_{index:02d}_0000.nii.gz")

    output = tmp_path / "oof"
    prepare_oof_v1(task="Task999", images_dir=images, output_dir=output)

    with (output / "fold_assignments.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 10
    assert {row["case_id"] for row in rows} == {f"Patient_{index:02d}" for index in range(10)}


def test_prepare_oof_can_raise_export_parallelism(tmp_path: Path):
    images = tmp_path / "Task999" / "imagesTr"
    images.mkdir(parents=True)
    for index in range(10):
        (images / f"case_{index:03d}_0000.nii.gz").write_bytes(b"not-a-real-nifti")

    output = tmp_path / "oof"
    prepare_oof_v1(
        task="Task999", images_dir=images, output_dir=output, nifti_save_threads=6
    )

    script = (output / "run_oof_inference.sh").read_text(encoding="utf-8")
    assert script.count("--num_threads_nifti_save 6") == 5


def test_prepare_oof_ensemble_requires_and_preserves_fold_pairing(tmp_path: Path):
    images = tmp_path / "Task999" / "imagesTr"
    images.mkdir(parents=True)
    for index in range(10):
        (images / f"case_{index:03d}_0000.nii.gz").write_bytes(b"not-a-real-nifti")
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_oof_v1(task="Task999", images_dir=images, output_dir=first, model="2d")
    prepare_oof_v1(task="Task999", images_dir=images, output_dir=second, model="3d_fullres")

    output = tmp_path / "ensemble"
    metadata = prepare_oof_ensemble_v1(
        first_oof_dir=first,
        second_oof_dir=second,
        output_dir=output,
    )
    script = (output / "run_oof_ensemble.sh").read_text(encoding="utf-8")

    assert metadata.is_file()
    assert script.count("nnUNet_ensemble") == 5
    assert script.count("--npz") == 5
    assert "Postprocessing is intentionally omitted" in script


def test_prepare_oof_ensemble_can_apply_official_postprocessing_to_native_masks(tmp_path: Path):
    images = tmp_path / "Task999" / "imagesTr"
    images.mkdir(parents=True)
    for index in range(10):
        (images / f"case_{index:03d}_0000.nii.gz").write_bytes(b"not-a-real-nifti")
    first = tmp_path / "first"
    second = tmp_path / "second"
    prepare_oof_v1(task="Task999", images_dir=images, output_dir=first, model="3d_fullres")
    prepare_oof_v1(
        task="Task999", images_dir=images, output_dir=second, model="3d_cascade_fullres"
    )
    postprocessing = tmp_path / "postprocessing.json"
    postprocessing.write_text('{"for_which_classes": []}\n', encoding="utf-8")

    output = tmp_path / "ensemble"
    metadata_path = prepare_oof_ensemble_v1(
        first_oof_dir=first,
        second_oof_dir=second,
        output_dir=output,
        postprocessing_file=postprocessing,
    )

    script = (output / "run_oof_ensemble.sh").read_text(encoding="utf-8")
    assert f"-pp {postprocessing.resolve()}" in script
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["postprocessing"] == str(postprocessing.resolve())
    assert metadata["first_oof_dir"] == str(first.resolve())
    assert metadata["second_oof_dir"] == str(second.resolve())
    assert len(metadata["fold_assignments_sha256"]) == 64

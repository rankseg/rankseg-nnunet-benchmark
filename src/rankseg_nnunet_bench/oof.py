from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import shlex
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold


CHANNEL_PATTERN = re.compile(r"^(?P<case>.+)_(?P<channel>\d{4})\.nii\.gz$")


def discover_nnunet_cases(images_dir: Path) -> dict[str, list[Path]]:
    cases: dict[str, list[Path]] = {}
    for path in sorted(images_dir.glob("*.nii.gz")):
        match = CHANNEL_PATTERN.match(path.name)
        if match is None:
            continue
        # Keep the nnU-Net entry name when it is a symlink. Resolving here can replace a valid
        # ``case_0000.nii.gz`` name with a source archive name that has no channel suffix.
        cases.setdefault(match.group("case"), []).append(path.absolute())
    if not cases:
        raise FileNotFoundError(f"No nnU-Net channel files (*_0000.nii.gz) found in {images_dir}")
    channel_sets = {case: tuple(path.name.rsplit("_", 1)[1] for path in paths) for case, paths in cases.items()}
    expected = next(iter(channel_sets.values()))
    inconsistent = {case: channels for case, channels in channel_sets.items() if channels != expected}
    if inconsistent:
        raise ValueError(f"Cases have inconsistent channel sets: {inconsistent}")
    expected_channels = tuple(f"{index:04d}.nii.gz" for index in range(len(expected)))
    if expected != expected_channels:
        raise ValueError(
            "nnU-Net channels must be consecutive and start at 0000; "
            f"found {expected}, expected {expected_channels}"
        )
    return cases


def default_nnunet_v1_splits(case_ids: list[str]) -> list[dict[str, list[str]]]:
    sorted_ids = np.asarray(sorted(case_ids))
    splitter = KFold(n_splits=5, shuffle=True, random_state=12345)
    return [
        {
            "train": sorted_ids[train_indices].tolist(),
            "val": sorted_ids[val_indices].tolist(),
        }
        for train_indices, val_indices in splitter.split(sorted_ids)
    ]


def load_splits_file(path: Path) -> list[dict[str, list[str]]]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)  # noqa: S301 - explicit, user-supplied trusted nnU-Net artifact
    splits = []
    for split in raw:
        splits.append(
            {
                "train": [str(case) for case in split["train"]],
                "val": [str(case) for case in split["val"]],
            }
        )
    return splits


def validate_splits(splits: list[dict[str, list[str]]], case_ids: set[str]) -> None:
    if len(splits) < 2:
        raise ValueError(f"OOF evaluation requires at least 2 folds, found {len(splits)}")
    validation_membership: list[str] = []
    for fold, split in enumerate(splits):
        if len(split["train"]) != len(set(split["train"])):
            raise ValueError(f"Fold {fold} contains duplicate training case IDs")
        if len(split["val"]) != len(set(split["val"])):
            raise ValueError(f"Fold {fold} contains duplicate validation case IDs")
        train = set(split["train"])
        val = set(split["val"])
        if train & val:
            raise ValueError(f"Fold {fold} has train/validation overlap")
        if train | val != case_ids:
            raise ValueError(f"Fold {fold} does not cover exactly the discovered cases")
        validation_membership.extend(split["val"])
    if len(validation_membership) != len(case_ids) or set(validation_membership) != case_ids:
        raise ValueError("Each case must occur in exactly one validation fold")


def _ensure_symlink(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f"Existing symlink points elsewhere: {destination}")
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    os.symlink(source, destination)


def prepare_oof_v1(
    *,
    task: str,
    images_dir: Path,
    output_dir: Path,
    splits_file: Path | None = None,
    model: str = "3d_fullres",
    trainer: str | None = None,
    plans: str | None = None,
    nifti_save_threads: int = 2,
) -> Path:
    if nifti_save_threads < 1:
        raise ValueError("nifti_save_threads must be positive")
    cases = discover_nnunet_cases(images_dir)
    splits = load_splits_file(splits_file) if splits_file else default_nnunet_v1_splits(list(cases))
    validate_splits(splits, set(cases))

    inputs_root = output_dir / "fold_inputs"
    predictions_root = output_dir / "fold_predictions"
    inputs_root.mkdir(parents=True, exist_ok=True)
    predictions_root.mkdir(parents=True, exist_ok=True)
    assignment_rows: list[dict[str, int | str]] = []
    commands: list[str] = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# PyTorch 2.6+ defaults to weights_only=True. The nnU-Net v1 archives contain legacy",
        "# full checkpoints. Enable this only after validating the trusted publisher checksum.",
        "export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
        "",
    ]

    for fold, split in enumerate(splits):
        fold_input = inputs_root / f"fold_{fold}"
        fold_output = predictions_root / f"fold_{fold}"
        fold_input.mkdir(exist_ok=True)
        fold_output.mkdir(exist_ok=True)
        for case_id in sorted(split["val"]):
            for source in cases[case_id]:
                _ensure_symlink(source, fold_input / source.name)
            assignment_rows.append({"case_id": case_id, "fold": fold})

        command = [
            "nnUNet_predict",
            "-i",
            str(fold_input.resolve()),
            "-o",
            str(fold_output.resolve()),
            "-t",
            task,
            "-m",
            model,
            "-f",
            str(fold),
            "--save_npz",
            "--num_threads_nifti_save",
            str(nifti_save_threads),
        ]
        if trainer:
            command.extend(["-tr", trainer])
        if plans:
            command.extend(["-p", plans])
        commands.append(" ".join(shlex.quote(token) for token in command))

    assignment_path = output_dir / "fold_assignments.csv"
    with assignment_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("case_id", "fold"))
        writer.writeheader()
        writer.writerows(sorted(assignment_rows, key=lambda row: str(row["case_id"])))
    (output_dir / "run_oof_inference.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    metadata = {
        "task": task,
        "model": model,
        "trainer": trainer,
        "plans": plans,
        "nifti_save_threads": nifti_save_threads,
        "cases": len(cases),
        "folds": len(splits),
        "split_source": str(splits_file.resolve()) if splits_file else "nnU-Net v1 default KFold(5, shuffle=True, random_state=12345)",
        "probability_glob": "fold_*/*.npz",
        "labels_dir": str((images_dir.parent / "labelsTr").resolve()),
    }
    metadata_path = output_dir / "oof_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def _read_fold_assignments(oof_dir: Path) -> dict[str, int]:
    path = oof_dir / "fold_assignments.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing OOF fold assignments: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assignments = {row["case_id"]: int(row["fold"]) for row in rows}
    if len(assignments) != len(rows):
        raise ValueError(f"Duplicate case IDs in {path}")
    observed_folds = sorted(set(assignments.values()))
    expected_folds = list(range(len(observed_folds)))
    if observed_folds != expected_folds:
        raise ValueError(
            f"Fold IDs in {path} must be consecutive and start at zero: {observed_folds}"
        )
    return assignments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_oof_ensemble_v1(
    *,
    first_oof_dir: Path,
    second_oof_dir: Path,
    output_dir: Path,
    threads: int = 2,
    postprocessing_file: Path | None = None,
) -> Path:
    if threads < 1:
        raise ValueError("threads must be positive")
    first_assignments = _read_fold_assignments(first_oof_dir)
    second_assignments = _read_fold_assignments(second_oof_dir)
    if first_assignments != second_assignments:
        raise ValueError("Refusing to ensemble OOF predictions with different fold assignments")
    first_metadata = json.loads((first_oof_dir / "oof_metadata.json").read_text(encoding="utf-8"))
    second_metadata = json.loads((second_oof_dir / "oof_metadata.json").read_text(encoding="utf-8"))
    if first_metadata["task"] != second_metadata["task"]:
        raise ValueError("Refusing to ensemble OOF predictions from different tasks")
    if postprocessing_file is not None and not postprocessing_file.is_file():
        raise FileNotFoundError(f"Missing official postprocessing file: {postprocessing_file}")

    predictions_root = output_dir / "fold_predictions"
    predictions_root.mkdir(parents=True, exist_ok=True)
    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Average the two exported softmax arrays. The saved npz remains raw so argmax and",
        "# RankSEG decode the same probabilities.",
        (
            "# Postprocessing is intentionally omitted from native masks."
            if postprocessing_file is None
            else "# Official postprocessing is applied only while exporting native NIfTI masks."
        ),
    ]
    folds = sorted(set(first_assignments.values()))
    for fold in folds:
        first = first_oof_dir / "fold_predictions" / f"fold_{fold}"
        second = second_oof_dir / "fold_predictions" / f"fold_{fold}"
        output = predictions_root / f"fold_{fold}"
        output.mkdir(exist_ok=True)
        command = [
            "nnUNet_ensemble",
            "-f",
            str(first.resolve()),
            str(second.resolve()),
            "-o",
            str(output.resolve()),
            "-t",
            str(threads),
            "--npz",
        ]
        if postprocessing_file is not None:
            command.extend(["-pp", str(postprocessing_file.resolve())])
        commands.append(" ".join(shlex.quote(token) for token in command))
    (output_dir / "run_oof_ensemble.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    metadata = {
        "task": first_metadata["task"],
        "cases": len(first_assignments),
        "folds": len(folds),
        "first_oof_dir": str(first_oof_dir.resolve()),
        "second_oof_dir": str(second_oof_dir.resolve()),
        "first_model": first_metadata["model"],
        "second_model": second_metadata["model"],
        "fold_assignments_identical": True,
        "fold_assignments_sha256": _sha256(first_oof_dir / "fold_assignments.csv"),
        "probability_operation": "arithmetic mean via nnUNet_ensemble --npz",
        "postprocessing": (
            False if postprocessing_file is None else str(postprocessing_file.resolve())
        ),
    }
    metadata_path = output_dir / "oof_ensemble_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path

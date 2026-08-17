from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation
import SimpleITK as sitk

from .config import DatasetConfig


PROBABILITY_KEYS = ("probabilities", "softmax", "probs")


class _RestrictedPropertiesUnpickler(pickle.Unpickler):
    """Load nnU-Net array metadata without allowing arbitrary pickle globals."""

    _ALLOWED_GLOBALS = {
        ("collections", "OrderedDict"),
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
    }

    def find_class(self, module: str, name: str):
        if (module, name) not in self._ALLOWED_GLOBALS:
            raise pickle.UnpicklingError(f"Disallowed global in nnU-Net properties pickle: {module}.{name}")
        return super().find_class(module, name)


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    probabilities: Path
    label: Path
    native_prediction: Path | None = None


@dataclass(frozen=True)
class ProbabilityDiagnostics:
    key: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float
    maximum: float
    max_normalization_error: float
    zero_sum_fraction: float


def case_id_from_probability(path: Path) -> str:
    return path.name[: -len(".npz")] if path.name.endswith(".npz") else path.stem


def discover_cases(config: DatasetConfig) -> list[CaseFiles]:
    if not config.probabilities_dir.is_dir():
        raise FileNotFoundError(f"Probability directory does not exist: {config.probabilities_dir}")
    if not config.labels_dir.is_dir():
        raise FileNotFoundError(f"Label directory does not exist: {config.labels_dir}")

    requested = None if config.case_ids is None else set(config.case_ids)
    cases: list[CaseFiles] = []
    found: set[str] = set()
    probability_by_case: dict[str, Path] = {}
    for probability_path in sorted(config.probabilities_dir.glob(config.probability_glob)):
        case_id = case_id_from_probability(probability_path)
        if requested is not None and case_id not in requested:
            continue
        if case_id in probability_by_case:
            raise ValueError(
                f"Duplicate probability files for {case_id}: "
                f"{probability_by_case[case_id]} and {probability_path}"
            )
        probability_by_case[case_id] = probability_path
        label_path = config.labels_dir / f"{case_id}{config.label_extension}"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for {case_id}: {label_path}")
        native = None
        if config.native_predictions_dir is not None:
            relative_probability = probability_path.relative_to(config.probabilities_dir)
            relative_native = relative_probability.with_name(f"{case_id}{config.label_extension}")
            candidate = config.native_predictions_dir / relative_native
            flat_candidate = config.native_predictions_dir / f"{case_id}{config.label_extension}"
            native = candidate if candidate.is_file() else flat_candidate if flat_candidate.is_file() else None
            if native is None:
                raise FileNotFoundError(
                    f"Missing configured native prediction for {case_id}; checked {candidate} and {flat_candidate}"
                )
        cases.append(CaseFiles(case_id, probability_path, label_path, native))
        found.add(case_id)

    if requested is not None and found != requested:
        raise FileNotFoundError(f"Requested cases without probability files: {sorted(requested - found)}")
    if not cases:
        raise FileNotFoundError(
            f"No probability files matched {config.probability_glob!r} in {config.probabilities_dir}"
        )
    return cases


def load_probabilities(path: Path, key: str = "auto") -> tuple[np.ndarray, str]:
    with np.load(path, allow_pickle=False) as archive:
        available = tuple(archive.files)
        if key == "auto":
            matches = [candidate for candidate in PROBABILITY_KEYS if candidate in archive]
            if len(matches) != 1:
                raise KeyError(
                    f"Could not choose one probability key in {path}; keys={available}, recognized={matches}"
                )
            key = matches[0]
        if key not in archive:
            raise KeyError(f"Probability key {key!r} absent from {path}; keys={available}")
        probabilities = np.asarray(archive[key])
    if probabilities.ndim < 3:
        raise ValueError(f"Probabilities must have shape (classes, *spatial), got {probabilities.shape} in {path}")
    return probabilities, key


def restore_cropped_probabilities(
    probabilities: np.ndarray,
    *,
    probability_path: Path,
    target_shape: tuple[int, ...],
    background_channel: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    stored_shape = tuple(int(value) for value in probabilities.shape)
    if tuple(probabilities.shape[1:]) == target_shape:
        return probabilities, {
            "crop_restored": False,
            "stored_probability_shape": json.dumps(stored_shape),
            "crop_bbox": None,
            "properties_file": None,
        }
    properties_path = probability_path.with_suffix(".pkl")
    if not properties_path.is_file():
        raise ValueError(
            f"Probability/label shape mismatch and no adjacent nnU-Net properties pickle exists: "
            f"{probabilities.shape} versus {target_shape}; missing {properties_path}"
        )
    with properties_path.open("rb") as handle:
        properties = _RestrictedPropertiesUnpickler(handle).load()
    required = {"original_size_of_raw_data", "crop_bbox"}
    if isinstance(properties, (list, tuple)):
        if not properties or not all(isinstance(item, dict) for item in properties):
            raise ValueError(f"Invalid ensemble properties list in {properties_path}")
        reference_original = tuple(int(value) for value in properties[0].get("original_size_of_raw_data", ()))
        reference_bbox = tuple(
            tuple(int(bound) for bound in bounds) for bounds in properties[0].get("crop_bbox", ())
        )
        for component in properties[1:]:
            component_original = tuple(int(value) for value in component.get("original_size_of_raw_data", ()))
            component_bbox = tuple(
                tuple(int(bound) for bound in bounds) for bounds in component.get("crop_bbox", ())
            )
            if component_original != reference_original or component_bbox != reference_bbox:
                raise ValueError(f"Ensemble components have inconsistent crop metadata in {properties_path}")
        properties = properties[0]
    if not isinstance(properties, dict) or not required <= set(properties):
        raise ValueError(f"Invalid nnU-Net properties in {properties_path}; required keys: {sorted(required)}")
    original_shape = tuple(int(value) for value in properties["original_size_of_raw_data"])
    if original_shape != target_shape:
        raise ValueError(
            f"nnU-Net properties original shape {original_shape} does not match label shape {target_shape}"
        )
    bbox = tuple((int(bounds[0]), int(bounds[1])) for bounds in properties["crop_bbox"])
    if len(bbox) != len(target_shape):
        raise ValueError(f"Crop bbox dimensionality mismatch in {properties_path}: {bbox}")
    crop_shape = tuple(end - start for start, end in bbox)
    if crop_shape != tuple(probabilities.shape[1:]):
        raise ValueError(
            f"Crop bbox shape {crop_shape} does not match stored probabilities {probabilities.shape[1:]}"
        )
    if any(start < 0 or end > size or start >= end for (start, end), size in zip(bbox, target_shape)):
        raise ValueError(f"Crop bbox is outside original image shape in {properties_path}: {bbox}")
    if not 0 <= background_channel < probabilities.shape[0]:
        raise ValueError(f"Invalid background probability channel: {background_channel}")

    restored = np.zeros((probabilities.shape[0], *target_shape), dtype=probabilities.dtype)
    restored[background_channel] = 1
    spatial_slices = tuple(slice(start, end) for start, end in bbox)
    restored[(slice(None), *spatial_slices)] = probabilities
    return restored, {
        "crop_restored": True,
        "stored_probability_shape": json.dumps(stored_shape),
        "crop_bbox": json.dumps(bbox),
        "properties_file": str(properties_path.resolve()),
    }


def load_segmentation_with_voxel_volume(
    path: Path,
    *,
    reader: str = "simpleitk",
) -> tuple[np.ndarray, float]:
    if path.name.endswith(".npy"):
        if reader != "simpleitk":
            raise ValueError(f"Segmentation reader {reader!r} only supports NIfTI files")
        return np.asarray(np.load(path, allow_pickle=False)), 1.0
    if path.name.endswith(".npz"):
        if reader != "simpleitk":
            raise ValueError(f"Segmentation reader {reader!r} only supports NIfTI files")
        with np.load(path, allow_pickle=False) as archive:
            keys = list(archive.files)
            if len(keys) != 1:
                raise KeyError(f"Segmentation npz must contain exactly one array: {path}, keys={keys}")
            return np.asarray(archive[keys[0]]), 1.0
    if reader == "simpleitk":
        image = sitk.ReadImage(str(path))
        return sitk.GetArrayFromImage(image), float(np.prod(image.GetSpacing(), dtype=np.float64))
    if reader == "nnunetv2_reoriented":
        # Match nnU-Net v2's NibabelIOWithReorient.read_images exactly. Exported
        # v2 probability arrays remain in this canonical RAS array coordinate
        # system, while the hard NIfTI is written back to the original affine.
        image = nib.load(str(path))
        if image.ndim != 3:
            raise ValueError(f"Only 3D NIfTI segmentations are supported: {path}")
        reoriented = image.as_reoriented(io_orientation(image.affine))
        array = np.asanyarray(reoriented.dataobj).transpose((2, 1, 0))
        return array, float(np.prod(reoriented.header.get_zooms(), dtype=np.float64))
    raise ValueError(f"Unsupported segmentation reader: {reader!r}")


def load_segmentation(path: Path, *, reader: str = "simpleitk") -> np.ndarray:
    return load_segmentation_with_voxel_volume(path, reader=reader)[0]


def validate_case(
    probabilities: np.ndarray,
    label: np.ndarray,
    *,
    channel_labels: tuple[int, ...],
    ignore_label: int | None,
    probability_key: str,
) -> ProbabilityDiagnostics:
    if tuple(probabilities.shape[1:]) != tuple(label.shape):
        raise ValueError(
            "Probability/label shape mismatch: "
            f"probabilities={probabilities.shape}, label={label.shape}. "
            "nnU-Net npz arrays are normally (C, Z, Y, X); load NIfTI labels with SimpleITK, not nibabel."
        )
    if probabilities.shape[0] != len(channel_labels):
        raise ValueError(f"Expected {len(channel_labels)} probability channels, got {probabilities.shape[0]}")
    if not np.issubdtype(probabilities.dtype, np.floating):
        raise TypeError(f"Probabilities must be floating point, got {probabilities.dtype}")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinity")
    minimum = float(probabilities.min())
    maximum = float(probabilities.max())
    tolerance = 5e-4
    if minimum < -tolerance or maximum > 1 + tolerance:
        raise ValueError(f"Probabilities outside [0, 1]: min={minimum}, max={maximum}")

    allowed = set(channel_labels)
    if ignore_label is not None:
        allowed.add(ignore_label)
    observed = set(int(value) for value in np.unique(label))
    if not observed <= allowed:
        raise ValueError(f"Label contains unknown values: {sorted(observed - allowed)}")

    sums = probabilities.astype(np.float32, copy=False).sum(axis=0)
    nonzero = sums > 1e-6
    max_error = float(np.max(np.abs(sums[nonzero] - 1.0))) if np.any(nonzero) else float("nan")
    if not np.all(nonzero):
        raise ValueError("Probabilities contain voxels whose class probabilities sum to zero")
    # nnU-Net stores exported softmax arrays as float16, so exact unit sums are not expected.
    # Errors larger than one percent are far beyond float16 roundoff and normally indicate that
    # logits, independently-normalized region scores, or a corrupted array were supplied instead.
    if max_error > 1e-2:
        raise ValueError(
            f"Class probabilities must sum to one at every voxel; maximum error is {max_error}"
        )
    return ProbabilityDiagnostics(
        key=probability_key,
        shape=tuple(int(value) for value in probabilities.shape),
        dtype=str(probabilities.dtype),
        minimum=minimum,
        maximum=maximum,
        max_normalization_error=max_error,
        zero_sum_fraction=float(1.0 - nonzero.mean()),
    )


def diagnostics_to_dict(value: ProbabilityDiagnostics) -> dict[str, Any]:
    return {
        "key": value.key,
        "shape": list(value.shape),
        "dtype": value.dtype,
        "minimum": value.minimum,
        "maximum": value.maximum,
        "max_normalization_error": value.max_normalization_error,
        "zero_sum_fraction": value.zero_sum_fraction,
    }

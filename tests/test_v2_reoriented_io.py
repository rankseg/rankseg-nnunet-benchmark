from pathlib import Path

import nibabel as nib
import numpy as np

from rankseg_nnunet_bench.io import load_segmentation_with_voxel_volume


def test_nnunetv2_reoriented_reader_matches_canonical_probability_coordinates(tmp_path: Path):
    # Original voxel axes point L, P, S. NibabelIOWithReorient flips the first
    # two axes into RAS before transposing xyz -> zyx for nnU-Net arrays.
    original = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    affine = np.diag([-2.0, -3.0, 4.0, 1.0])
    path = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(original, affine), path)

    observed, voxel_volume = load_segmentation_with_voxel_volume(
        path,
        reader="nnunetv2_reoriented",
    )
    expected = original[::-1, ::-1, :].transpose((2, 1, 0))

    assert np.array_equal(observed, expected)
    assert voxel_volume == 24.0


def test_simpleitk_reader_remains_default(tmp_path: Path):
    original = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)
    path = tmp_path / "seg.nii.gz"
    nib.save(nib.Nifti1Image(original, np.eye(4)), path)

    observed, voxel_volume = load_segmentation_with_voxel_volume(path)

    assert np.array_equal(observed, original.transpose((2, 1, 0)))
    assert voxel_volume == 1.0

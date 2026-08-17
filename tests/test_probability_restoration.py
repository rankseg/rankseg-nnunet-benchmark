from collections import OrderedDict
import pickle

import numpy as np
import pytest

from rankseg_nnunet_bench.io import restore_cropped_probabilities, validate_case


def test_restore_v1_cropped_probabilities_uses_background_outside_bbox(tmp_path):
    probability_path = tmp_path / "case.npz"
    probabilities = np.zeros((2, 2, 2), dtype=np.float32)
    probabilities[0] = 0.25
    probabilities[1] = 0.75
    properties = OrderedDict(
        original_size_of_raw_data=np.asarray([2, 4]),
        crop_bbox=[[0, 2], [1, 3]],
        size_after_cropping=(2, 2),
    )
    with probability_path.with_suffix(".pkl").open("wb") as handle:
        pickle.dump(properties, handle)

    restored, diagnostics = restore_cropped_probabilities(
        probabilities,
        probability_path=probability_path,
        target_shape=(2, 4),
        background_channel=0,
    )

    assert restored.shape == (2, 2, 4)
    np.testing.assert_array_equal(restored[:, :, 0], np.asarray([[1, 1], [0, 0]]))
    np.testing.assert_array_equal(restored[:, :, 3], np.asarray([[1, 1], [0, 0]]))
    np.testing.assert_allclose(restored[:, :, 1:3], probabilities)
    assert diagnostics["crop_restored"] is True


def test_restore_v1_ensemble_accepts_identical_component_properties(tmp_path):
    probability_path = tmp_path / "ensemble.npz"
    probabilities = np.asarray([[[0.2]], [[0.8]]], dtype=np.float32)
    component = OrderedDict(
        original_size_of_raw_data=np.asarray([1, 3]),
        crop_bbox=[[0, 1], [1, 2]],
        spacing_after_resampling=np.float64(0.625),
    )
    with probability_path.with_suffix(".pkl").open("wb") as handle:
        pickle.dump([component, component.copy()], handle)

    restored, diagnostics = restore_cropped_probabilities(
        probabilities,
        probability_path=probability_path,
        target_shape=(1, 3),
        background_channel=0,
    )

    np.testing.assert_allclose(restored[1], [[0.0, 0.8, 0.0]])
    assert diagnostics["crop_restored"] is True


def test_validate_case_rejects_non_softmax_scores():
    probabilities = np.full((2, 2, 2), 0.8, dtype=np.float32)
    target = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="must sum to one"):
        validate_case(
            probabilities,
            target,
            channel_labels=(0, 1),
            ignore_label=None,
            probability_key="softmax",
        )


def test_validate_case_accepts_float16_roundoff():
    probabilities = np.asarray([0.3333, 0.3333, 0.3333], dtype=np.float16)[:, None, None]
    diagnostics = validate_case(
        probabilities,
        np.zeros((1, 1), dtype=np.uint8),
        channel_labels=(0, 1, 2),
        ignore_label=None,
        probability_key="softmax",
    )

    assert diagnostics.max_normalization_error < 1e-2

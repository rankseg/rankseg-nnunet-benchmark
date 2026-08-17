from pathlib import Path

import numpy as np
import pytest

from rankseg_nnunet_bench.config import DatasetConfig
from rankseg_nnunet_bench.io import discover_cases


def test_discover_cases_rejects_duplicate_case_across_folds(tmp_path: Path):
    probabilities = tmp_path / "probabilities"
    labels = tmp_path / "labels"
    for fold in (0, 1):
        folder = probabilities / f"fold_{fold}"
        folder.mkdir(parents=True)
        np.savez_compressed(folder / "case_001.npz", softmax=np.ones((1, 1, 1)))
    labels.mkdir()
    np.save(labels / "case_001.npy", np.zeros((1, 1), dtype=np.uint8))
    config = DatasetConfig(
        dataset_id="Task999",
        display_name="duplicate-test",
        labels={0: "background"},
        channel_labels=(0,),
        probabilities_dir=probabilities,
        labels_dir=labels,
        output_dir=tmp_path / "output",
        foreground_labels=(),
        probability_glob="fold_*/*.npz",
        label_extension=".npy",
    )

    with pytest.raises(ValueError, match="Duplicate probability files"):
        discover_cases(config)

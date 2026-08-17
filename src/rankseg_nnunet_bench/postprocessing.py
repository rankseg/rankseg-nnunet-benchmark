from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np


class OfficialNnUNetPostprocessor:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"nnU-Net postprocessing file does not exist: {path}")
        config = json.loads(path.read_text(encoding="utf-8"))
        classes = config.get("for_which_classes")
        if not isinstance(classes, list):
            raise ValueError(f"Invalid for_which_classes in {path}: {classes!r}")
        self.classes = [tuple(value) if isinstance(value, list) else int(value) for value in classes]
        raw_minimum = config.get("min_valid_object_sizes", "None")
        self.minimum_valid_object_sizes = (
            ast.literal_eval(raw_minimum) if isinstance(raw_minimum, str) else raw_minimum
        )
        self.path = path

    def __call__(self, prediction: np.ndarray, *, volume_per_voxel: float) -> np.ndarray:
        from nnunet.postprocessing.connected_components import remove_all_but_the_largest_connected_component

        postprocessed, _, _ = remove_all_but_the_largest_connected_component(
            prediction.copy(),
            self.classes,
            volume_per_voxel,
            self.minimum_valid_object_sizes,
        )
        return postprocessed

    def summary(self) -> dict:
        return {
            "file": str(self.path.resolve()),
            "for_which_classes": self.classes,
            "minimum_valid_object_sizes": self.minimum_valid_object_sizes,
            "applied_identically_to": ["argmax", "rankseg"],
        }

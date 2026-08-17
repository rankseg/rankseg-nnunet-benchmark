import json

import numpy as np

from rankseg_nnunet_bench.postprocessing import OfficialNnUNetPostprocessor


def test_official_postprocessing_removes_smaller_component_only_for_selected_class(tmp_path):
    path = tmp_path / "postprocessing.json"
    path.write_text(
        json.dumps({"for_which_classes": [2], "min_valid_object_sizes": "None"}),
        encoding="utf-8",
    )
    prediction = np.zeros((5, 5), dtype=np.uint8)
    prediction[0, 0] = 2
    prediction[2:4, 2:4] = 2
    prediction[4, 0] = 1

    result = OfficialNnUNetPostprocessor(path)(prediction, volume_per_voxel=1.0)

    assert result[0, 0] == 0
    assert np.count_nonzero(result == 2) == 4
    assert result[4, 0] == 1

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/env/bin:$PATH"
export nnUNet_raw_data_base="$ROOT/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$ROOT/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$ROOT/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task001_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

archive="$ROOT/artifacts/datasets/Task01_BrainTumour.tar"
model="$ROOT/artifacts/models/Task001_BrainTumour.zip"
source="$ROOT/artifacts/datasets/Task01_BrainTumour"
raw="$ROOT/work/v1/nnUNet_raw_data_base/nnUNet_raw_data/Task001_BrainTumour"
oof="$ROOT/work/v1/oof/Task001_BrainTumour_3d_fullres"
model_dir="$ROOT/work/v1/RESULTS_FOLDER/nnUNet/3d_fullres/Task001_BrainTumour/nnUNetTrainerV2__nnUNetPlansv2.1"
pp="$model_dir/postprocessing.json"

[[ -f "$archive" ]] || { echo "Missing Task001 dataset archive" >&2; exit 1; }
[[ "$(stat -c %s "$archive")" == 7608266240 ]] || { echo "Task001 dataset size mismatch" >&2; exit 1; }
[[ "$(md5sum "$archive" | cut -d' ' -f1)" == 240a19d752f0d9e9101544901065d872 ]] || {
    echo "Task001 dataset MD5 mismatch" >&2
    exit 1
}
[[ -f "$model" ]] || { echo "Missing Task001 model archive" >&2; exit 1; }
[[ "$(stat -c %s "$model")" == 1869973352 ]] || { echo "Task001 model size mismatch" >&2; exit 1; }
[[ "$(md5sum "$model" | cut -d' ' -f1)" == 08f648c1247ab5deab2465ddc84f2feb ]] || {
    echo "Task001 model MD5 mismatch" >&2
    exit 1
}

if [[ ! -f "$source/dataset.json" ]]; then
    tar -xf "$archive" -C "$ROOT/artifacts/datasets"
fi
if [[ ! -f "$raw/dataset.json" ]]; then
    nnUNet_convert_decathlon_task -i "$source"
fi
nnUNet_install_pretrained_model_from_zip "$model"

[[ -f "$pp" ]] || { echo "Missing official Task001 postprocessing file" >&2; exit 1; }

"$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-v1 \
    --task Task001_BrainTumour --images-dir "$raw/imagesTr" --output-dir "$oof" \
    --model 3d_fullres --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1 \
    --nifti-save-threads 6
bash "$oof/run_oof_inference.sh"

"$ROOT/env/bin/rankseg-nnunet-bench" evaluate \
    "$ROOT/configs/Task001_BrainTumour_3d_fullres_oof.yaml" --device cuda

echo "Task001 evaluation complete; rebuild the main result with scripts/build_full_aggregate.sh"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/env/bin:$PATH"
export nnUNet_raw_data_base="$ROOT/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$ROOT/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$ROOT/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task003_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

archive="$ROOT/artifacts/datasets/Task03_Liver.tar"
model="$ROOT/artifacts/models/Task003_Liver.zip"
source="$ROOT/artifacts/datasets/Task03_Liver"
raw="$ROOT/work/v1/nnUNet_raw_data_base/nnUNet_raw_data/Task003_Liver"
lowres="$ROOT/work/v1/oof/Task003_Liver_3d_lowres"
fullres="$ROOT/work/v1/oof/Task003_Liver_3d_fullres"
ensemble="$ROOT/work/v1/oof/Task003_Liver_ensemble"
pp="$ROOT/work/v1/RESULTS_FOLDER/nnUNet/ensembles/Task003_Liver/ensemble_3d_lowres__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json"

[[ -f "$archive" ]] || { echo "Missing Task003 dataset archive" >&2; exit 1; }
[[ "$(stat -c %s "$archive")" == 28925891584 ]] || { echo "Task003 dataset size mismatch" >&2; exit 1; }
[[ "$(md5sum "$archive" | cut -d' ' -f1)" == a90ec6c4aa7f6a3d087205e23d4e6397 ]] || {
    echo "Task003 dataset MD5 mismatch" >&2
    exit 1
}
[[ -f "$model" ]] || { echo "Missing Task003 model archive" >&2; exit 1; }
[[ "$(stat -c %s "$model")" == 5049539577 ]] || { echo "Task003 model size mismatch" >&2; exit 1; }
[[ "$(md5sum "$model" | cut -d' ' -f1)" == 552e1ee740ef47fa8d8025246c84b5da ]] || {
    echo "Task003 model MD5 mismatch" >&2
    exit 1
}

if [[ ! -f "$source/dataset.json" ]]; then
    tar -xf "$archive" -C "$ROOT/artifacts/datasets"
fi
if [[ ! -f "$raw/dataset.json" ]]; then
    nnUNet_convert_decathlon_task -i "$source"
fi
nnUNet_install_pretrained_model_from_zip "$model"

[[ -f "$pp" ]] || { echo "Missing official Task003 postprocessing file" >&2; exit 1; }

"$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-v1 \
    --task Task003_Liver --images-dir "$raw/imagesTr" --output-dir "$lowres" \
    --model 3d_lowres --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1
"$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-v1 \
    --task Task003_Liver --images-dir "$raw/imagesTr" --output-dir "$fullres" \
    --model 3d_fullres --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1
bash "$lowres/run_oof_inference.sh"
bash "$fullres/run_oof_inference.sh"
"$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-ensemble-v1 \
    --first-oof-dir "$lowres" --second-oof-dir "$fullres" --output-dir "$ensemble" \
    --postprocessing-file "$pp"
bash "$ensemble/run_oof_ensemble.sh"

"$ROOT/env/bin/rankseg-nnunet-bench" evaluate \
    "$ROOT/configs/Task003_Liver_ensemble_oof.yaml" --device cuda

echo "Task003 evaluation complete; rebuild the main result with scripts/build_full_aggregate.sh"

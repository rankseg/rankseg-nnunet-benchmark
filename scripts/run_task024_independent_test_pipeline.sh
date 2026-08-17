#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/env/bin:$PATH"
export nnUNet_raw_data_base="$ROOT/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$ROOT/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$ROOT/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task024_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

dataset_archive="$ROOT/work/v1/source_data/PROMISE12/PROMISE12_test_data.zip"
model_archive="$ROOT/work/v1/model_archives/Task024_Promise.zip"
source="$ROOT/work/v1/source_data/PROMISE12/extracted_test"
test_root="$ROOT/work/v1/independent_test/Task024_Promise"
converted="$test_root/converted"
two_d="$test_root/predictions_2d"
fullres="$test_root/predictions_3d_fullres"
ensemble="$test_root/predictions_ensemble"
model_root="$ROOT/work/v1/RESULTS_FOLDER/nnUNet"
two_d_model="$model_root/2d/Task024_Promise/nnUNetTrainerV2__nnUNetPlansv2.1"
fullres_model="$model_root/3d_fullres/Task024_Promise/nnUNetTrainerV2__nnUNetPlansv2.1"
pp="$model_root/ensembles/Task024_Promise/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json"

[[ -f "$dataset_archive" ]] || { echo "Missing PROMISE12 test archive" >&2; exit 1; }
[[ "$(stat -c %s "$dataset_archive")" == 194659961 ]] || { echo "PROMISE12 test archive size mismatch" >&2; exit 1; }
[[ "$(md5sum "$dataset_archive" | cut -d' ' -f1)" == 823ef560f4a54083348e58d67403e4bb ]] || {
    echo "PROMISE12 test archive MD5 mismatch" >&2
    exit 1
}
unzip -tqq "$dataset_archive"

[[ -f "$model_archive" ]] || { echo "Missing Task024 model archive" >&2; exit 1; }
[[ "$(stat -c %s "$model_archive")" == 2258526938 ]] || { echo "Task024 model size mismatch" >&2; exit 1; }
[[ "$(md5sum "$model_archive" | cut -d' ' -f1)" == c8dfacef77d4390786572194aa506595 ]] || {
    echo "Task024 model MD5 mismatch" >&2
    exit 1
}

mkdir -p "$source"
if [[ ! -f "$source/Case00.mhd" ]]; then
    unzip -q "$dataset_archive" -d "$source"
fi
if [[ ! -f "$converted/conversion_manifest.json" ]]; then
    "$ROOT/env/bin/python" "$ROOT/scripts/prepare_task024_promise_test.py" \
        --source-dir "$source" --output-dir "$converted" --source-archive "$dataset_archive"
fi

if [[ ! -f "$two_d_model/fold_0/model_final_checkpoint.model" || ! -f "$fullres_model/fold_0/model_final_checkpoint.model" ]]; then
    nnUNet_install_pretrained_model_from_zip "$model_archive"
fi
for fold in 0 1 2 3 4; do
    [[ -f "$two_d_model/fold_$fold/model_final_checkpoint.model" ]] || { echo "Missing Task024 2d fold $fold" >&2; exit 1; }
    [[ -f "$fullres_model/fold_$fold/model_final_checkpoint.model" ]] || { echo "Missing Task024 3d_fullres fold $fold" >&2; exit 1; }
done
[[ -f "$pp" ]] || { echo "Missing official Task024 ensemble postprocessing file" >&2; exit 1; }

mkdir -p "$two_d" "$fullres" "$ensemble"
nnUNet_predict -i "$converted/images" -o "$two_d" -t Task024_Promise -m 2d \
    -f 0 1 2 3 4 --save_npz --num_threads_nifti_save 6 \
    -tr nnUNetTrainerV2 -p nnUNetPlansv2.1
nnUNet_predict -i "$converted/images" -o "$fullres" -t Task024_Promise -m 3d_fullres \
    -f 0 1 2 3 4 --save_npz --num_threads_nifti_save 6 \
    -tr nnUNetTrainerV2 -p nnUNetPlansv2.1
nnUNet_ensemble -f "$two_d" "$fullres" -o "$ensemble" -t 6 --npz -pp "$pp"

"$ROOT/env/bin/rankseg-nnunet-bench" evaluate \
    "$ROOT/configs/Task024_PROMISE12_independent_test.yaml" --device cuda

echo "Task024 evaluation complete; rebuild the main result with scripts/build_full_aggregate.sh"

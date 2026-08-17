#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/env/bin:$PATH"
export nnUNet_raw_data_base="$ROOT/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$ROOT/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$ROOT/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task038_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

dataset_archive="$ROOT/work/v1/source_data/CHAOS/CHAOS_Train_Sets.zip"
model_archive="$ROOT/work/v1/model_archives/Task038_CHAOS_Task_3_5_Variant2.zip"
extracted="$ROOT/work/v1/source_data/CHAOS/extracted"
source="$extracted/Train_Sets"
raw="$ROOT/work/v1/nnUNet_raw_data_base/nnUNet_raw_data/Task038_CHAOS_Task_3_5_Variant2"
splits="$ROOT/work/v1/nnUNet_preprocessed/Task038_CHAOS_Task_3_5_Variant2/splits_final.pkl"
two_d="$ROOT/work/v1/oof/Task038_CHAOS_2d"
fullres="$ROOT/work/v1/oof/Task038_CHAOS_3d_fullres"
ensemble="$ROOT/work/v1/oof/Task038_CHAOS_ensemble"
model_root="$ROOT/work/v1/RESULTS_FOLDER/nnUNet"
two_d_model="$model_root/2d/Task038_CHAOS_Task_3_5_Variant2/nnUNetTrainerV2__nnUNetPlansv2.1"
fullres_model="$model_root/3d_fullres/Task038_CHAOS_Task_3_5_Variant2/nnUNetTrainerV2__nnUNetPlansv2.1"
pp="$model_root/ensembles/Task038_CHAOS_Task_3_5_Variant2/ensemble_2d__nnUNetTrainerV2__nnUNetPlansv2.1--3d_fullres__nnUNetTrainerV2__nnUNetPlansv2.1/postprocessing.json"

[[ -f "$dataset_archive" ]] || { echo "Missing official CHAOS training archive" >&2; exit 1; }
[[ "$(stat -c %s "$dataset_archive")" == 890771694 ]] || { echo "CHAOS archive size mismatch" >&2; exit 1; }
[[ "$(md5sum "$dataset_archive" | cut -d' ' -f1)" == df21053002a1cc86df918a87da3b2c19 ]] || {
    echo "CHAOS archive MD5 mismatch" >&2
    exit 1
}
unzip -tqq "$dataset_archive"

[[ -f "$model_archive" ]] || { echo "Missing official Task038 model archive" >&2; exit 1; }
[[ "$(stat -c %s "$model_archive")" == 2774011680 ]] || { echo "Task038 model size mismatch" >&2; exit 1; }
[[ "$(md5sum "$model_archive" | cut -d' ' -f1)" == 57bc8bc3c80a5dc08917aafd517407b4 ]] || {
    echo "Task038 model MD5 mismatch" >&2
    exit 1
}
unzip -tqq "$model_archive"

mkdir -p "$extracted"
if [[ ! -d "$source/MR" ]]; then
    unzip -q "$dataset_archive" 'Train_Sets/MR/*' -d "$extracted"
fi

if [[ ! -f "$raw/conversion_manifest.json" ]]; then
    "$ROOT/env/bin/python" "$ROOT/scripts/prepare_task038_chaos.py" \
        --source-dir "$source" --output-dir "$raw" --splits-file "$splits" \
        --source-archive "$dataset_archive"
fi

if [[ ! -f "$two_d_model/fold_0/model_final_checkpoint.model" || ! -f "$fullres_model/fold_0/model_final_checkpoint.model" ]]; then
    nnUNet_install_pretrained_model_from_zip "$model_archive"
fi
for fold in 0 1 2 3 4; do
    [[ -f "$two_d_model/fold_$fold/model_final_checkpoint.model" ]] || { echo "Missing Task038 2D fold $fold" >&2; exit 1; }
    [[ -f "$fullres_model/fold_$fold/model_final_checkpoint.model" ]] || { echo "Missing Task038 3D fold $fold" >&2; exit 1; }
done
[[ -f "$pp" ]] || { echo "Missing official Task038 ensemble postprocessing" >&2; exit 1; }

"$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-v1 \
    --task Task038_CHAOS_Task_3_5_Variant2 --images-dir "$raw/imagesTr" --output-dir "$two_d" \
    --splits-file "$splits" --model 2d --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1 \
    --nifti-save-threads 6
"$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-v1 \
    --task Task038_CHAOS_Task_3_5_Variant2 --images-dir "$raw/imagesTr" --output-dir "$fullres" \
    --splits-file "$splits" --model 3d_fullres --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1 \
    --nifti-save-threads 6
bash "$two_d/run_oof_inference.sh"
bash "$fullres/run_oof_inference.sh"

"$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-ensemble-v1 \
    --first-oof-dir "$two_d" --second-oof-dir "$fullres" --output-dir "$ensemble" \
    --threads 6 --postprocessing-file "$pp"
bash "$ensemble/run_oof_ensemble.sh"

"$ROOT/env/bin/rankseg-nnunet-bench" evaluate \
    "$ROOT/configs/Task038_CHAOS_ensemble_oof.yaml" --device cuda

echo "Task038 evaluation complete; rebuild the main result with scripts/build_full_aggregate.sh"

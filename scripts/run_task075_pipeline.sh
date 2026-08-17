#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/env/bin:$PATH"
export nnUNet_raw_data_base="$ROOT/work/v1/nnUNet_raw_data_base"
export nnUNet_preprocessed="$ROOT/work/v1/nnUNet_preprocessed"
export RESULTS_FOLDER="$ROOT/work/v1/RESULTS_FOLDER"
export MPLCONFIGDIR=/tmp/matplotlib_task075_pipeline
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

manual_archive="$ROOT/work/v1/source_data/CTC/Fluo-C3DH-A549.zip"
simulated_archive="$ROOT/work/v1/source_data/CTC/Fluo-C3DH-A549-SIM.zip"
source_root="$ROOT/work/v1/source_data/CTC/extracted"
manual_source="$source_root/Fluo-C3DH-A549"
simulated_source="$source_root/Fluo-C3DH-A549-SIM"
raw="$ROOT/work/v1/nnUNet_raw_data_base/nnUNet_raw_data/Task075_Fluo_C3DH_A549_ManAndSim"
splits="$ROOT/work/v1/nnUNet_preprocessed/Task075_Fluo_C3DH_A549_ManAndSim/splits_final.pkl"
oof="$ROOT/work/v1/oof/Task075_Fluo_C3DH_A549_3d_fullres"
model_archive="$ROOT/work/v1/model_archives/Task075_Fluo_C3DH_A549_ManAndSim.zip"
model_dir="$ROOT/work/v1/RESULTS_FOLDER/nnUNet/3d_fullres/Task075_Fluo_C3DH_A549_ManAndSim/nnUNetTrainerV2__nnUNetPlansv2.1"

[[ -f "$manual_archive" ]] || { echo "Missing Task075 manual dataset archive" >&2; exit 1; }
[[ "$(stat -c %s "$manual_archive")" == 256705648 ]] || { echo "Task075 manual dataset size mismatch" >&2; exit 1; }
[[ -f "$simulated_archive" ]] || { echo "Missing Task075 simulated dataset archive" >&2; exit 1; }
[[ "$(stat -c %s "$simulated_archive")" == 329974837 ]] || { echo "Task075 simulated dataset size mismatch" >&2; exit 1; }
unzip -tqq "$manual_archive"
unzip -tqq "$simulated_archive"

[[ -f "$model_archive" ]] || { echo "Missing Task075 model archive" >&2; exit 1; }
[[ "$(stat -c %s "$model_archive")" == 1336092839 ]] || { echo "Task075 model size mismatch" >&2; exit 1; }
[[ "$(md5sum "$model_archive" | cut -d' ' -f1)" == c097ad567d0a8d75a6d183bda19b5f6b ]] || {
    echo "Task075 model MD5 mismatch" >&2
    exit 1
}
for fold in 0 1 2 3; do
    [[ -f "$model_dir/fold_$fold/model_final_checkpoint.model" ]] || {
        echo "Missing Task075 fold $fold final checkpoint" >&2
        exit 1
    }
done

mkdir -p "$source_root"
if [[ ! -d "$manual_source/01" ]]; then
    unzip -q "$manual_archive" -d "$source_root"
fi
if [[ ! -d "$simulated_source/01" ]]; then
    unzip -q "$simulated_archive" -d "$source_root"
fi

if [[ ! -f "$raw/dataset.json" ]]; then
    "$ROOT/env/bin/python" "$ROOT/scripts/prepare_task075_celltracking.py" \
        --manual-source "$manual_source" --simulated-source "$simulated_source" \
        --output-root "$(dirname "$raw")" --splits-file "$splits"
fi

if [[ ! -f "$oof/oof_metadata.json" ]]; then
    "$ROOT/env/bin/rankseg-nnunet-bench" prepare-oof-v1 \
        --task Task075_Fluo_C3DH_A549_ManAndSim --images-dir "$raw/imagesTr" --output-dir "$oof" \
        --splits-file "$splits" --model 3d_fullres --trainer nnUNetTrainerV2 --plans nnUNetPlansv2.1 \
        --nifti-save-threads 6
fi
bash "$oof/run_oof_inference.sh"

"$ROOT/env/bin/rankseg-nnunet-bench" evaluate \
    "$ROOT/configs/Task075_Fluo_C3DH_A549_3d_fullres_oof.yaml" --device cuda

echo "Task075 evaluation complete; rebuild the main result with scripts/build_full_aggregate.sh"

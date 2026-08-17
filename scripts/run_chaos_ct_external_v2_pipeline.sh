#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$ROOT/env/bin:$PATH"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export nnUNet_raw="$ROOT/work/v2/nnUNet_raw"
export nnUNet_preprocessed="$ROOT/work/v2/nnUNet_preprocessed"
export nnUNet_results="$ROOT/work/v2/nnUNet_results"
export MPLCONFIGDIR=/tmp/matplotlib_chaos_ct_v2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

dataset_archive="$ROOT/work/v1/source_data/CHAOS/CHAOS_Train_Sets.zip"
model_archive="$ROOT/work/v2/model_archives/Dataset291_TotalSegmentator_part1_organs_1559subj.zip"
source="$ROOT/work/v2/source_data/CHAOS/extracted/Train_Sets/CT"
dataset="$ROOT/work/v2/datasets/CHAOS_CT_external"
model="$nnUNet_results/Dataset291_TotalSegmentator_part1_organs_1559subj/nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"
predictions="$ROOT/work/v2/predictions/CHAOS_CT_Dataset291"
config="$ROOT/configs/CHAOS_CT_TotalSegmentator_v2_external.yaml"

[[ -f "$dataset_archive" ]] || { echo "Missing official CHAOS archive" >&2; exit 1; }
[[ "$(stat -c %s "$dataset_archive")" == 890771694 ]] || { echo "CHAOS archive size mismatch" >&2; exit 1; }
[[ "$(md5sum "$dataset_archive" | cut -d' ' -f1)" == df21053002a1cc86df918a87da3b2c19 ]] || {
    echo "CHAOS archive MD5 mismatch" >&2
    exit 1
}
[[ -f "$model_archive" ]] || { echo "Missing official Dataset291 archive" >&2; exit 1; }
[[ "$(stat -c %s "$model_archive")" == 233742255 ]] || { echo "Dataset291 archive size mismatch" >&2; exit 1; }
[[ "$(sha256sum "$model_archive" | cut -d' ' -f1)" == 2cde8e6bcfb5b6a02183648a7036c3d6b28bc854576f8e1c34fbd6b4dd5c6c5b ]] || {
    echo "Dataset291 archive SHA256 mismatch" >&2
    exit 1
}
unzip -tqq "$dataset_archive"
unzip -tqq "$model_archive"

mkdir -p "$(dirname "$source")" "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$predictions"
if [[ ! -d "$source" ]]; then
    unzip -q "$dataset_archive" 'Train_Sets/CT/*' -d "$ROOT/work/v2/source_data/CHAOS/extracted"
fi
if [[ ! -f "$dataset/conversion_manifest.json" ]]; then
    "$ROOT/env/bin/python" "$ROOT/scripts/prepare_chaos_ct_external_v2.py" \
        --source-dir "$source" --source-archive "$dataset_archive" --output-dir "$dataset"
fi
if [[ ! -f "$model" ]]; then
    nnUNetv2_install_pretrained_model_from_zip "$model_archive"
fi

nnUNetv2_predict \
    -i "$dataset/imagesTs" \
    -o "$predictions" \
    -d 291 \
    -c 3d_fullres \
    -p nnUNetPlans \
    -tr nnUNetTrainerNoMirroring \
    -f 0 \
    -chk checkpoint_final.pth \
    --disable_tta \
    --save_probabilities \
    --continue_prediction \
    -npp 3 \
    -nps 2 \
    -device cuda

rankseg-nnunet-bench evaluate "$config" --device cuda
"$ROOT/env/bin/python" "$ROOT/scripts/audit_chaos_ct_external_v2.py"

bash "$ROOT/scripts/build_full_aggregate.sh"

echo "CHAOS CT external v2 benchmark complete"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$ROOT/env/bin/python" \
    -m rankseg_nnunet_bench.cli aggregate \
    "$ROOT/outputs/Task001_BrainTumour_3d_fullres_oof/summary.json" \
    "$ROOT/outputs/Task002_Heart_3d_fullres_oof/summary.json" \
    "$ROOT/outputs/Task003_Liver_ensemble_oof/summary.json" \
    "$ROOT/outputs/Task004_Hippocampus_oof/summary.json" \
    "$ROOT/outputs/Task005_Prostate_ensemble_postprocessed_oof/summary.json" \
    "$ROOT/outputs/Task006_Lung_ensemble_oof/summary.json" \
    "$ROOT/outputs/Task007_Pancreas_ensemble_oof/summary.json" \
    "$ROOT/outputs/Task008_HepaticVessel_ensemble_oof/summary.json" \
    "$ROOT/outputs/Task009_Spleen_ensemble_postprocessed_oof/summary.json" \
    "$ROOT/outputs/Task010_Colon_3d_cascade_fullres_oof/summary.json" \
    "$ROOT/outputs/Task027_ACDC_ensemble_oof/summary.json" \
    "$ROOT/outputs/Task055_SegTHOR_ensemble_oof/summary.json" \
    "$ROOT/outputs/Task075_Fluo_C3DH_A549_3d_fullres_oof/summary.json" \
    "$ROOT/outputs/Task024_PROMISE12_independent_test/summary.json" \
    "$ROOT/outputs/Task038_CHAOS_ensemble_oof/summary.json" \
    "$ROOT/outputs/CHAOS_CT_TotalSegmentator_v2_external/summary.json" \
    --output-dir "$ROOT/outputs/full_aggregate" \
    --overall-tests

echo "Full 16-dataset benchmark aggregate complete"

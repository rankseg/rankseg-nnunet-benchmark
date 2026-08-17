| Task | Dataset | Model | Cases | Argmax mean Dice (%) | RankSEG mean Dice (%) | Dice Δ (pp) | Improved (>+1 pp) | Stable (within ±1 pp) | Worsened (<-1 pp) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| CHAOS | CHAOS CT | nnU-Net v2 · TotalSegmentator Dataset291 · 3d_fullres (fold 0) | 20 | 97.08608 | 97.09150 | +0.00542 | 0 | 20 | 0 |
| Task001 | MSD BrainTumour | nnU-Net v1 · 3d_fullres | 484 | 84.57217 | 84.57988 | +0.00771 | 7 | 476 | 1 |
| Task002 | MSD Heart | nnU-Net v1 · 3d_fullres | 20 | 93.29433 | 93.29623 | +0.00191 | 0 | 20 | 0 |
| Task003 | MSD Liver | nnU-Net v1 · 3d_lowres + 3d_fullres ensemble | 131 | 80.99903 | 81.82314 | +0.82412 | 32 | 90 | 9 |
| Task004 | MSD Hippocampus | nnU-Net v1 · 3d_fullres | 260 | 88.90847 | 88.91252 | +0.00405 | 0 | 260 | 0 |
| Task005 | MSD Prostate | nnU-Net v1 · 2d + 3d_fullres ensemble | 32 | 75.95709 | 75.98004 | +0.02295 | 8 | 20 | 4 |
| Task006 | MSD Lung | nnU-Net v1 · 3d_lowres + 3d_fullres ensemble | 63 | 72.28822 | 72.74899 | +0.46077 | 20 | 33 | 10 |
| Task007 | MSD Pancreas | nnU-Net v1 · 3d_fullres + 3d_cascade_fullres ensemble | 281 | 68.39597 | 69.56287 | +1.16690 | 66 | 191 | 24 |
| Task008 | MSD HepaticVessel | nnU-Net v1 · 3d_lowres + 3d_fullres ensemble | 303 | 68.63054 | 69.16535 | +0.53481 | 85 | 169 | 49 |
| Task009 | MSD Spleen | nnU-Net v1 · 3d_fullres + 3d_cascade_fullres ensemble | 41 | 97.23659 | 97.24320 | +0.00661 | 0 | 41 | 0 |
| Task010 | MSD Colon | nnU-Net v1 · 3d_cascade_fullres (3d_lowres input) | 126 | 48.91759 | 49.15961 | +0.24203 | 14 | 90 | 22 |
| Task024 | PROMISE12 former hidden test | nnU-Net v1 · 2d + 3d_fullres ensemble | 30 | 91.93567 | 91.85350 | -0.08217 | 0 | 29 | 1 |
| Task027 | ACDC | nnU-Net v1 · 2d + 3d_fullres ensemble | 200 | 92.45381 | 92.38008 | -0.07373 | 7 | 181 | 12 |
| Task038 | CHAOS MRI | nnU-Net v1 · 2d + 3d_fullres ensemble | 60 | 92.04389 | 92.21990 | +0.17601 | 7 | 52 | 1 |
| Task055 | SegTHOR | nnU-Net v1 · 3d_fullres + 3d_cascade_fullres ensemble | 40 | 91.51999 | 91.51811 | -0.00188 | 0 | 40 | 0 |
| Task075 | CTC Fluo-C3DH-A549 manual + simulated | nnU-Net v1 · 3d_fullres | 90 | 93.94401 | 93.95587 | +0.01186 | 0 | 90 | 0 |

Macro-average Dice across datasets (each dataset weighted equally): argmax 83.63646%, RankSEG 83.84318%. Dataset-level Dice deltas: 13 positive, 0 tied, 3 negative; mean +0.20671 pp, median +0.00979 pp.

Overall benchmark p-values (dataset as the unit; two-sided): Dice exact sign test p=0.0212708, Dice Wilcoxon p=0.0109863.

Overall mean Dice is the main endpoint. Case counts and performance/treatment-effect quantiles are descriptive diagnostics. Each dataset contributes one unit to the cross-dataset summary.

Case-level Dice changes at the ±1 pp practical threshold (2181 cases): 246 improved, 1802 stable, 133 worsened.
More extreme case-level Dice changes: >5 pp: 26 losses, 64 gains; >10 pp: 9 losses, 31 gains.
Worst: Task010_Colon/colon_107 (-44.63 pp); best: Task010_Colon/colon_108 (+51.42 pp).

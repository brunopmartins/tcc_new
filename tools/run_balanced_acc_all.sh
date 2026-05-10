#!/bin/bash
# Run per-relation balanced accuracy (FaCoRNet methodology) on top checkpoints.
set -eo pipefail

PROJECT=/home/bruno/Desktop/tcc_new
PYTHON=$PROJECT/models/03_convnext_vit_hybrid/.venv/bin/python

export HSA_OVERRIDE_GFX_VERSION=10.3.0
export MIOPEN_FIND_MODE=FAST
export HSA_FORCE_FINE_GRAIN_PCIE=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export PYTHONPATH=$PROJECT/models:$PROJECT/models/shared:$PYTHONPATH

echo "============================================================"
echo "[1/4] M02 R031 (final.pt — closest to peak ep 3)"
echo "============================================================"
$PYTHON $PROJECT/tools/per_relation_balanced_accuracy.py \
  --model 02 \
  --checkpoint $PROJECT/models/02_vit_facor_crossattn/output/031/checkpoints/final.pt \
  --output_dir $PROJECT/models/02_vit_facor_crossattn/output/031/results \
  --batch_size 8 \
  --num_workers 4

echo ""
echo "============================================================"
echo "[2/4] M05 R001 (frozen DINOv2 + LoRA)"
echo "============================================================"
$PYTHON $PROJECT/tools/per_relation_balanced_accuracy.py \
  --model 05 \
  --checkpoint $PROJECT/models/05_dinov2_lora_diffattn/output/001/checkpoints/best.pt \
  --output_dir $PROJECT/models/05_dinov2_lora_diffattn/output/001/results \
  --batch_size 4 \
  --num_workers 4

echo ""
echo "============================================================"
echo "[3/4] M05 R005 (full unfreeze, M02-style LR)"
echo "============================================================"
$PYTHON $PROJECT/tools/per_relation_balanced_accuracy.py \
  --model 05 \
  --checkpoint $PROJECT/models/05_dinov2_lora_diffattn/output/005/checkpoints/best.pt \
  --output_dir $PROJECT/models/05_dinov2_lora_diffattn/output/005/results \
  --batch_size 4 \
  --num_workers 4

echo ""
echo "============================================================"
echo "[4/4] M05 R007 (DINOv2 + M02-trained-ViT hybrid)"
echo "============================================================"
$PYTHON $PROJECT/tools/per_relation_balanced_accuracy.py \
  --model 05 \
  --checkpoint $PROJECT/models/05_dinov2_lora_diffattn/output/007/checkpoints/best.pt \
  --output_dir $PROJECT/models/05_dinov2_lora_diffattn/output/007/results \
  --batch_size 4 \
  --num_workers 4

echo ""
echo "============================================================"
echo "ALL 4 COMPLETE — comparison table:"
echo "============================================================"
$PYTHON -c "
import json
from pathlib import Path
project = Path('$PROJECT')
runs = [
    ('M02 R031', project / 'models/02_vit_facor_crossattn/output/031/results/per_relation_balanced_accuracy.json'),
    ('M05 R001', project / 'models/05_dinov2_lora_diffattn/output/001/results/per_relation_balanced_accuracy.json'),
    ('M05 R005', project / 'models/05_dinov2_lora_diffattn/output/005/results/per_relation_balanced_accuracy.json'),
    ('M05 R007', project / 'models/05_dinov2_lora_diffattn/output/007/results/per_relation_balanced_accuracy.json'),
]
print(f'{\"Run\":<10s} {\"Overall Acc\":>12s} {\"Mean Pos Acc\":>14s} {\"Mean Balanced\":>14s} {\"Non-kin Acc\":>14s}')
for name, p in runs:
    try:
        d = json.loads(p.read_text())
        print(f'{name:<10s} {d[\"overall\"][\"accuracy\"]:>12.4f} {d[\"headline\"][\"mean_pos_acc\"]:>14.4f} {d[\"headline\"][\"mean_balanced_acc\"]:>14.4f} {d[\"nonkin\"][\"accuracy\"]:>14.4f}')
    except FileNotFoundError:
        print(f'{name:<10s}  (no output)')
print()
print('Literature SOTA reference: FaCoRNet-AdaFace = 0.8200 (mean balanced)')
"

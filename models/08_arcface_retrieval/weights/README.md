# ArcFace IResNet-100 weights

This directory holds the pretrained ArcFace weights. **Not committed to git**
because the file is ~250-500 MB.

## How to obtain

The standard checkpoint comes from InsightFace's model zoo. Several mirrors
exist; pick whichever works in your environment.

### Option 1 — InsightFace GitHub releases

```bash
cd models/08_arcface_retrieval/weights/

# MS1MV3-trained R100 (most common variant)
wget -O arcface_r100.pth \
    https://github.com/deepinsight/insightface/releases/download/v0.7/ms1mv3_arcface_r100_fp16.pth
```

### Option 2 — HuggingFace mirror

```bash
pip install huggingface_hub
huggingface-cli download \
    deepinsight/insightface arcface_r100_glint360k.pth \
    --local-dir models/08_arcface_retrieval/weights/

# Rename to the expected file name
mv models/08_arcface_retrieval/weights/arcface_r100_glint360k.pth \
   models/08_arcface_retrieval/weights/arcface_r100.pth
```

### Option 3 — Manual download

If neither of the above works:
1. Visit https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
2. Look for the "Performance" table linking to OneDrive / Baidu
3. Download `glint360k_r100.pth` or `ms1mv3_r100.pth` (any IResNet-100 ArcFace)
4. Save as `models/08_arcface_retrieval/weights/arcface_r100.pth`

## Variants comparison

| File | Pretrain set | Loss | LFW | IJB-C 1e-4 | Size |
|---|---|---|---:|---:|---:|
| ms1mv2_arcface_r100.pth | MS1MV2 (5.8M faces) | ArcFace | 99.83% | 96.0% | ~250 MB |
| ms1mv3_arcface_r100.pth | MS1MV3 (5.2M, cleaner) | ArcFace | 99.83% | 96.3% | ~250 MB |
| glint360k_cosface_r100.pth | Glint360K (17M faces) | CosFace | 99.85% | 96.8% | ~250 MB |

Any of these works as the M08 backbone. The differences are small; Glint360K
gives slightly better generalization due to the larger and more diverse
training set, but all are state-of-the-art for face verification.

## Verifying

After download, run:

```bash
python -c "
import sys; sys.path.insert(0, 'models/08_arcface_retrieval')
from model import ArcFaceEncoder
enc = ArcFaceEncoder(weights_path='models/08_arcface_retrieval/weights/arcface_r100.pth')
print('OK')
"
```

If the load succeeds without complaint, you're ready to run preprocessing
and training.

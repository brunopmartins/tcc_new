# `align_faces_224.py` — project-wide ArcFace-landmark alignment at 224×224

Aligns FIW (or KinFaceW) face crops to the canonical ArcFace 5-point template,
scaled to a 224×224 canvas (2× the standard 112×112 ArcFace size). Output is
the native input size of ViT-B/16, DINOv2, and ImageNet ViT-L/16, so no
quality is lost from down/upsampling at training time.

## Why two `FIW_aligned*` directories exist

| Directory | Producer | Detector | Transform | Notes |
|---|---|---|---|---|
| `datasets/FIW_aligned/` | `tools/align_fiw_dataset.py` | insightface buffalo_l (RetinaFace) | `cv2.estimateAffinePartial2D` (LMEDS) onto the 2× ArcFace template | Pre-existing 224×224 dataset, produced by an earlier pipeline. Do **not** modify or delete — current M02/M03/M05 runs reference it. |
| `datasets/FIW_aligned_224/` | `tools/align_faces_224.py` (this script) | facenet-pytorch MTCNN | Umeyama similarity transform onto the same 2× ArcFace template | New canonical ArcFace-landmark-aligned 224 dataset. Identical methodology to model 08's `align_faces.py` (which writes 112), differing only in the target canvas size. |

Both datasets target the same template; differences in output come from the
detector and from the affine solver. The model-08 lineage uses
Umeyama / facenet-pytorch MTCNN, so this script preserves that lineage at 2×
resolution.

## Usage

Default (224×224, FIW, ROCm GPU 0):
```bash
./tools/run_align_faces_224.sh
```

Override the target size, source, destination, or device via env vars:
```bash
OUTPUT_SIZE=224 \
SRC_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW \
DST_ROOT=/home/bruno/Desktop/tcc_new/datasets/FIW_aligned_224 \
GPU_ID=0 \
./tools/run_align_faces_224.sh
```

Direct Python invocation (bypasses the wrapper):
```bash
/home/bruno/Desktop/tcc_new/models/08_arcface_retrieval/.venv/bin/python \
    /home/bruno/Desktop/tcc_new/tools/align_faces_224.py \
    --dataset fiw \
    --src-root /home/bruno/Desktop/tcc_new/datasets/FIW \
    --dst-root /home/bruno/Desktop/tcc_new/datasets/FIW_aligned_224 \
    --output-size 224
```

Smoke-test against 100 images on CPU:
```bash
USE_CPU=1 LIMIT=100 DST_ROOT=/tmp/fiw_224_smoke \
    ./tools/run_align_faces_224.sh
```

## Properties

- **Resumable**: re-running skips images whose destination already exists.
- **Idempotent symlinks**: `track-I/` and `FIW_PIDs_v2.csv` are symlinked once
  at the destination root so `models/shared/dataset.py` finds them.
- **Fallback**: when MTCNN fails to detect a face (rare on FIW, ~<0.5%), the
  script center-crops the largest inscribed square and resizes — never skips,
  so the pair lists remain valid.
- **Output**: `.jpg` only (matches the dataset loader's `glob("*.jpg")`).

## Dependencies

Inherited from `models/08_arcface_retrieval/.venv`:
- `torch`, `torchvision` (ROCm/CUDA build)
- `facenet-pytorch`
- `opencv-python-headless`
- `pillow`, `tqdm`, `numpy`

If that venv does not yet exist, build it once via
`models/08_arcface_retrieval/AMD/run_preprocessing.sh` (which installs all
dependencies and then runs the 112 version). After that, this 224 script
piggy-backs on the same environment.

## Estimated processing time

On an AMD GPU (matching the 112 version's reference setup) the FIW dataset
(~600k face crops, depending on track expansion) takes roughly **2-3 h**.
Warp + JPEG-encode at 224 is slightly heavier than at 112 but still
detection-bound, so total runtime is comparable. CPU-only mode is ~5-10×
slower.

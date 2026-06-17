#!/usr/bin/env python3
"""Offline face-parsing → per-image region boxes for Model 17 (PG-RGCK).

Walks an aligned-face root (224×224 ArcFace-template crops), runs a face-parsing
segmentation network on each face, groups the semantic classes into the four
anatomical regions M12/M15 use (eyes, nose, mouth, jaw) and writes the **tight
per-face bounding box** of each region. The result is cached to an ``.npz`` that
the dataloader reads (``--region_box_cache``); training/eval then use per-image
adaptive boxes instead of the fixed ``DEFAULT_REGIONS_224`` rectangles.

This is a PURE PREPROCESSING step — it runs once, offline, on CPU or GPU, and
never touches the training loop (no extra VRAM, no parser in the hot path).

Cache format (read by models/shared/dataset.py):
    np.savez_compressed(out,
        keys=<object array of flattened rel-paths: rel.replace('/','__')>,
        boxes=<float32 (N, 4, 4) xyxy in the 224 frame, order eyes,nose,mouth,jaw>)

Backends
--------
--backend dummy   : writes the FIXED DEFAULT boxes for every face. No parser
                    needed. Use it to validate the whole M17 data path end to
                    end (cache → dataset → model → eval); an M17 run on a dummy
                    cache should reproduce M15 (sanity check) before you invest
                    in the real parser.
--backend bisenet : BiSeNet trained on CelebAMask-HQ (19 classes), the de-facto
                    standard face parser. Needs the model code + weights:
                      git clone https://github.com/zllrunning/face-parsing.PyTorch
                      # weights: 79999_iter.pth (in that repo's README / release)
                    then:
                      python tools/parse_faces_boxes.py --backend bisenet \
                        --bisenet-repo /path/to/face-parsing.PyTorch \
                        --weights /path/to/79999_iter.pth \
                        --aligned-root .../FIW_aligned --out .../fiw_region_boxes.npz

The valuable, backend-independent logic (class→region grouping, tight-bbox
extraction, min-area fallback, jaw = lower-face skin, key scheme) is implemented
here and is what makes the boxes *adaptive*. Swapping parsers only changes
``parse_labelmap``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

SRC = 224  # all boxes live in the 224 aligned frame

# Fixed-box fallback (MUST match models/shared/dataset.py:DEFAULT_ANAT_BOXES_XYXY
# and models/12_rgck_net/model.py:DEFAULT_REGIONS_224). Order: eyes,nose,mouth,jaw.
DEFAULT_ANAT_BOXES_XYXY = np.array([
    [20.0,  40.0, 204.0, 100.0],  # eyes
    [70.0,  80.0, 154.0, 150.0],  # nose
    [50.0, 140.0, 174.0, 185.0],  # mouth
    [20.0, 170.0, 204.0, 220.0],  # jaw
], dtype=np.float32)
ANAT_ORDER = ["eyes", "nose", "mouth", "jaw"]

# CelebAMask-HQ 19-class label ids (zllrunning BiSeNet order).
CMASK = {
    "background": 0, "skin": 1, "l_brow": 2, "r_brow": 3, "l_eye": 4, "r_eye": 5,
    "eye_g": 6, "l_ear": 7, "r_ear": 8, "ear_r": 9, "nose": 10, "mouth": 11,
    "u_lip": 12, "l_lip": 13, "neck": 14, "neck_l": 15, "cloth": 16, "hair": 17,
    "hat": 18,
}
# Which parsing classes compose each anatomical region.
REGION_CLASSES = {
    "eyes":  [CMASK["l_brow"], CMASK["r_brow"], CMASK["l_eye"], CMASK["r_eye"], CMASK["eye_g"]],
    "nose":  [CMASK["nose"]],
    "mouth": [CMASK["mouth"], CMASK["u_lip"], CMASK["l_lip"]],
    # jaw is derived specially (lower-face skin) — see boxes_from_labelmap.
    "jaw":   [CMASK["skin"]],
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _mask_bbox(mask: np.ndarray, pad: int = 4) -> "tuple[float,float,float,float] | None":
    """Tight xyxy bbox of a boolean mask (224 frame), padded and clamped. None
    if the mask is empty."""
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    x0 = max(0.0, float(xs.min()) - pad)
    y0 = max(0.0, float(ys.min()) - pad)
    x1 = min(float(SRC), float(xs.max()) + 1.0 + pad)
    y1 = min(float(SRC), float(ys.max()) + 1.0 + pad)
    return x0, y0, x1, y1


def boxes_from_labelmap(lab: np.ndarray, min_area: int = 20) -> "tuple[np.ndarray, list[bool]]":
    """labelmap (224,224) of class ids → (4,4) xyxy boxes in [eyes,nose,mouth,jaw]
    order, plus a per-region 'used_fallback' list. Empty/tiny regions fall back to
    the fixed DEFAULT box (so every image always yields valid boxes)."""
    boxes = DEFAULT_ANAT_BOXES_XYXY.copy()
    fell_back = [False, False, False, False]

    # Nose centroid → split skin into the lower (jaw) part.
    nose_mask = np.isin(lab, REGION_CLASSES["nose"])
    nose_cy = float(np.where(nose_mask)[0].mean()) if nose_mask.any() else SRC * 0.45

    for j, name in enumerate(ANAT_ORDER):
        if name == "jaw":
            # jaw/chin = skin pixels below the nose centroid (lower-face contour).
            mask = np.isin(lab, REGION_CLASSES["jaw"]).copy()
            mask[: int(nose_cy), :] = False
        else:
            mask = np.isin(lab, REGION_CLASSES[name])
        if int(mask.sum()) < min_area:
            fell_back[j] = True
            continue
        bb = _mask_bbox(mask)
        if bb is None or (bb[2] - bb[0]) < 2 or (bb[3] - bb[1]) < 2:
            fell_back[j] = True
            continue
        boxes[j] = np.asarray(bb, dtype=np.float32)
    return boxes.astype(np.float32), fell_back


# ---------------------------------------------------------------------------
# Parser backends → a 224×224 int labelmap
# ---------------------------------------------------------------------------

def make_parser(args):
    """Returns parse_labelmap(pil_img_rgb_224) -> np.ndarray[224,224] int."""
    if args.backend == "dummy":
        # No parser: every region falls back to the fixed DEFAULT box. The cache
        # then reproduces M15 — a plumbing/sanity check, not a real M17 run.
        def parse_labelmap(_img):
            return np.zeros((SRC, SRC), dtype=np.int64)  # all background → all fallback
        return parse_labelmap

    if args.backend == "bisenet":
        import torch
        import torchvision.transforms as T
        if not args.bisenet_repo or not args.weights:
            sys.exit("--backend bisenet requires --bisenet-repo and --weights")
        sys.path.insert(0, str(Path(args.bisenet_repo).resolve()))
        try:
            from model import BiSeNet  # zllrunning/face-parsing.PyTorch:model.py
        except Exception as e:  # noqa: BLE001
            sys.exit(f"Could not import BiSeNet from {args.bisenet_repo}: {e}")
        device = torch.device(args.device)
        net = BiSeNet(n_classes=19).to(device)
        net.load_state_dict(torch.load(args.weights, map_location=device))
        net.eval()
        # BiSeNet was trained at 512 with ImageNet normalisation.
        to_net = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        @torch.no_grad()
        def parse_labelmap(img):
            x = to_net(img).unsqueeze(0).to(device)
            out = net(x)[0]  # (1,19,512,512)
            lab512 = out.squeeze(0).argmax(0).byte().cpu().numpy()  # (512,512)
            # nearest-resize the labelmap back to the 224 frame
            lab = np.asarray(
                Image.fromarray(lab512).resize((SRC, SRC), Image.NEAREST),
                dtype=np.int64,
            )
            return lab
        return parse_labelmap

    sys.exit(f"Unknown backend: {args.backend}")


def main():
    p = argparse.ArgumentParser(description="Face-parsing → per-image region boxes (M17)")
    p.add_argument("--aligned-root", required=True, help="Root of aligned 224 faces (FIW_aligned).")
    p.add_argument("--out", required=True, help="Output .npz path.")
    p.add_argument("--backend", default="bisenet", choices=["bisenet", "dummy"])
    p.add_argument("--bisenet-repo", default=None, help="Path to zllrunning/face-parsing.PyTorch.")
    p.add_argument("--weights", default=None, help="BiSeNet weights (79999_iter.pth).")
    p.add_argument("--device", default="cpu", help="cpu | cuda:0 (parser only; offline).")
    p.add_argument("--min-area", type=int, default=20, help="Min mask pixels before fixed fallback.")
    p.add_argument("--limit", type=int, default=0, help="Process only the first N faces (debug).")
    args = p.parse_args()

    root = Path(args.aligned_root).resolve()
    if not root.is_dir():
        sys.exit(f"aligned-root not found: {root}")

    faces = sorted(f for f in root.rglob("*") if f.suffix.lower() in IMG_EXTS)
    if args.limit:
        faces = faces[: args.limit]
    if not faces:
        sys.exit(f"No images under {root}")
    print(f"Found {len(faces)} aligned faces under {root}")
    print(f"Backend: {args.backend}" + (f" (device {args.device})" if args.backend == "bisenet" else ""))

    parse_labelmap = make_parser(args)

    keys, boxes_all = [], []
    fallback_counts = np.zeros(4, dtype=np.int64)
    n_all_fallback = 0
    for f in tqdm(faces, desc="parsing"):
        rel = f.relative_to(root).as_posix()
        key = rel.replace("/", "__")  # MUST match dataset._box_key
        try:
            img = Image.open(f).convert("RGB")
            if img.size != (SRC, SRC):
                img = img.resize((SRC, SRC), Image.BILINEAR)
            lab = parse_labelmap(img)
            boxes, fell = boxes_from_labelmap(lab, min_area=args.min_area)
        except Exception as e:  # noqa: BLE001
            tqdm.write(f"  [warn] {rel}: {e} → fixed fallback")
            boxes, fell = DEFAULT_ANAT_BOXES_XYXY.copy(), [True, True, True, True]
        keys.append(key)
        boxes_all.append(boxes)
        fallback_counts += np.asarray(fell, dtype=np.int64)
        if all(fell):
            n_all_fallback += 1

    boxes_arr = np.stack(boxes_all).astype(np.float32)  # (N,4,4)
    keys_arr = np.array(keys, dtype=object)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=keys_arr, boxes=boxes_arr)

    n = len(faces)
    print(f"\nWrote {n} entries → {out}")
    print("Per-region fixed-box fallback rate (high ⇒ parser weak on that part):")
    for name, c in zip(ANAT_ORDER, fallback_counts):
        print(f"  {name:5s}: {c}/{n} ({100*c/n:.1f}%)")
    print(f"  ALL-4 fallback (parser found nothing): {n_all_fallback}/{n} ({100*n_all_fallback/n:.1f}%)")
    if args.backend == "dummy":
        print("\n[dummy] all boxes are the fixed DEFAULT — this cache reproduces M15.")


if __name__ == "__main__":
    main()

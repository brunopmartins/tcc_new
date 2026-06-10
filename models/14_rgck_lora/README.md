# Model 14 — RGCK-Net + LoRA backbone

M14 is **M12 (RGCK-Net) with the AdaFace IR-101 backbone adapted by LoRA**
instead of the stage-4 unfreeze used by M12 R002–R012.

## Motivation

The M12 R&D cycle converged on a CV-AUC ceiling of ~0.876–0.879. A diagnostic
on the R012 checkpoint (2026-06-09) showed:

- **Val AUC peaks at ~epoch 3 then declines** while train loss keeps falling
  (0.50 → 0.15): the model memorises training families almost immediately. The
  binding constraint is **generalisation**, not head capacity.
- The full stage-4 unfreeze (~31 M trainable params) gives the model a large
  surface to memorise on.

LoRA adds a **small, low-rank, regularised** adaptation surface (~0.9 M backbone
params, ~6 M total trainable vs ~34 M for R012) to the same layers (stage 4 +
`output_layer`). The hypothesis: adapt the kinship representation with far less
memorisation, raising the generalisable ceiling.

## What is shared with M12

- Architecture: `RGCKNet` is imported verbatim from `../12_rgck_net/model.py`
  (region tokens, bidirectional cross-region attention, sigmoid gate, fusion
  classifier, symmetric forward, comparison-only fusion, relation-aux head,
  and the optional ROI-Align tokenizer).
- The AMD harness (`AMD/train.py`, `AMD/test.py`, `AMD/run_pipeline.sh`) is a
  copy of M12's with the model builder swapped to `build_rgck_lora_net` and
  LoRA controls added (`--lora_rank`, `--lora_alpha`, `--lora_stage3_tail`).

## What is new

- [`lora.py`](lora.py): `LoRALinear`, `LoRAConv2d`, and
  `inject_lora_into_adaface(...)`. LoRA delta is zero at init (B / up-conv
  zero-initialised) so training starts at the frozen-AdaFace solution.
- [`model.py`](model.py): `build_rgck_lora_net(...)` builds the M12 model with a
  frozen backbone, then injects LoRA into stage 4 (+ optionally stage-3 tail)
  and `output_layer`.

## Runs

- **R001** ([`AMD/run_m14_r001.sh`](AMD/run_m14_r001.sh)): proven M12 data+head
  recipe (symmetric forward + comparison-only fusion + relation aux 0.05 +
  role-matched hard negatives 30 %) + LoRA rank 16 / alpha 16 on stage 4 +
  `output_layer`. Single LR 1e-4, no differential LR, no consistency loss.
  Outputs to `output/001/`.

Compare M14 R001 (LoRA) against M12 R012 (unfreeze) on the same fixed RFIW
Track-I test set; both share the data + head recipe, so the delta isolates
**LoRA adaptation vs stage-4 unfreeze**.

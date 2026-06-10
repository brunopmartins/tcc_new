# Model 16 — FA-RGCK (Family-Adversarial Region-Guided Cross Kinship)

M16 attacks the **one constraint that caps every model in this project**:
generalisation to unseen families (the val→test gap). It does so *directly*,
with domain-adversarial training — the first model in the project to do so.

## Why this, and why now

The whole project converges on one diagnosis:

- **The binding constraint is family memorisation.** Models with a tight
  val→test gap win (M02 −0.031, M12 R006 −0.026); models that memorise training
  families collapse on held-out families (M10 −0.140, M11 −0.128).
- **R012's diagnostic**: the M12 stack peaks on Val AUC by ~epoch 3 then declines
  while train loss keeps falling — it memorises families almost immediately.
- **Every generalisation gain so far was *indirect*:** shortcut removal (R006
  symmetric forward, R009 comparison-only fusion), LoRA regularisation (M14),
  in-distribution region tokens (M15). The only thing that cleanly broke the
  0.876 ceiling was *averaging* CV-fold models (R011 ensemble 0.8839) — variance
  reduction over models that saw different family subsets.

No model removes family-identifiable features at the source. M16 does.

## Mechanism — DANN, each FIW family is a domain

```
img_a, img_b ─► M12 R011 head (verbatim) ─► kinship logit ─► BCE
                     │
            gA, gB  (per-face contextualised global tokens, L2-norm)
                     │
            [Gradient Reversal Layer]   (λ: DANN schedule 0 → 1)
                     │
            family discriminator (num_families-way) ─► CE_family

   L = 0.5·(BCE_AB+BCE_BA) + 0.05·CE_rel  +  family_adv_weight · CE_family
                                              └─ GRL reverses this term's
                                                 gradient into the backbone
```

- Forward, the GRL is the identity; backward, it multiplies the gradient by
  **−λ**. So the discriminator learns to predict family (normal CE), while the
  backbone + cross-region adapter are pushed to make family **unrecoverable**.
- Crucially the adversary acts on the **per-face** global tokens (removing
  *which family/identity* a face is), not on the kinship logit — so the
  **pairwise** kinship signal (|diff|, products, per-region cosines) is kept.
- **λ follows the DANN schedule** `λ = 2/(1+exp(−γ·p)) − 1`, `p = epoch/epochs`.
  At λ=0 the kinship path is **identical to M12 R011** (high floor); the
  invariance pressure ramps up as training proceeds.

The val→test gap is literally what the adversary minimises — a tight theoretical
link between the objective and the project's binding metric.

## What is shared / what is new

- **Head + recipe**: M12 `RGCKNet` imported verbatim from `../12_rgck_net`
  (region tokens, cross-region attention, gate, fusion classifier, symmetric
  forward, comparison-only fusion, relation aux). R001 = the R011 stack.
- **New** ([`model.py`](model.py)): `GradReverse` (GRL), `FamilyAdversary` (MLP
  discriminator), `FARGCKNet` (wraps the M12 model; forward returns the
  unchanged M12 tuple + a family-logits dict appended last, so eval/metrics/base
  loss are untouched), `build_fa_rgck_net`.
- **Harness** ([`AMD/`](AMD/)): copy of M12's with the builder swapped and a
  `FARGCKROCmTrainer` that builds the family vocabulary, sets the DANN λ each
  epoch, and adds the masked family CE. One additive field (`family1`/`family2`)
  was added to `shared/dataset.py` (FIW family id per face; ignored by other
  models).

## Runs

- **R001** ([`AMD/run_m16_r001.sh`](AMD/run_m16_r001.sh)): R011 recipe + family
  adversary, `family_adv_weight=0.1`, DANN γ=10, single LR 1e-5. Outputs to
  `output/001/`.

M16 R001 vs M12 R011 isolates **explicit family-invariance** (the family
adversary) against everything else held fixed.

## VRAM (12 GB — AMD RX 6750 XT)

Trainable 31.3 M (= R011 31.0 M + 0.28 M adversary); backbone identical to M12.
Fits 12 GB at `BATCH_SIZE=8 GRAD_ACCUM=4`. If OOM: `BATCH_SIZE=4 GRAD_ACCUM=8`.

## Quick start

```bash
bash models/16_fa_rgck/AMD/run_m16_r001.sh
# sweep the pressure:
FAMILY_ADV_WEIGHT=0.3 bash models/16_fa_rgck/AMD/run_m16_r001.sh
```

## Honest caveats

- Adversarial training is higher-variance than the proven recipes. The DANN
  schedule + the λ=0 floor (≡ R011) de-risk it: the worst case is "no movement".
- A 571-way family discriminator is hard (long-tailed family sizes). If it
  doesn't help, the informative next probes are `family_adv_weight ∈ {0.05, 0.3}`
  and a coarser family bucketing.
- The architecture is smoke-tested (CPU); the training-loop integration (family
  vocab, DANN λ, masked CE) mirrors the proven relation-aux plumbing but should
  be sanity-checked on the **first real run** (watch: train BCE descends cleanly,
  family CE stays > 0, Val AUC trajectory).

## Reference

- DANN: Ganin & Lempitsky, "Unsupervised Domain Adaptation by Backpropagation",
  ICML 2015 (gradient reversal).
- M12 R011 recipe: `../12_rgck_net/run-review/run-011.md`.

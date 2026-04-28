# Run Log — Model 06: Retrieval-Augmented Kinship

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 002 — 2026-04-27 — Completed (early stop, regression vs run 001)

**Status:** Stopped (early stop, patience 20)
**Outcome:** Test ROC AUC = **0.7310**, best Val AUC = **0.8228** at epoch 9 — pior que run 001 em test (0.7763) apesar de val maior. Hipótese de melhoria com DINOv2 + K=64 + relation_loss_weight=0.3 **não** se confirmou.

### Launch command

```bash
SKIP_INSTALL=1 BACKBONE=vit_base_patch14_dinov2.lvd142m \
  RETRIEVAL_K=64 RELATION_LOSS_WEIGHT=0.3 STORE_GALLERY_ON_CPU=1 \
  bash models/06_retrieval_augmented_kinship/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (66.4k train / 11.4k val / 13.4k test pairs) |
| Backbone | **vit_base_patch14_dinov2.lvd142m** (frozen) |
| Img size | 224 |
| Batch size | 8 |
| Grad accum | 4 (effective batch = 32) |
| LR | 1e-4 |
| Scheduler | cosine, warmup=3 ep, min_lr=1e-6 |
| Weight decay | 1e-4 |
| Epochs | 60 (early-stopped at 29) |
| Patience | 20 |
| Loss | combined (BCE + 0.3 × contrastive + **0.3 × relation-CE**) |
| Temperature | 0.1 |
| **Retrieval K** | **64** |
| Cross-attention | 2 layers, 4 heads |
| Embedding dim | 512 |
| Dropout | 0.1 |
| Gallery cap | 200,000 |
| **Gallery on CPU** | **True** |
| Gallery refresh every | 0 (built once) |
| Max grad norm | 1.0 |
| AMP | on |
| Seed | 42 |
| Workers | 4 |

### Training trajectory

- Best Val AUC: **0.8228** at epoch 9 (best.pt saved here)
- Stopped at epoch 29 (patience 20 hit, peak na ep 9 + 20 sem melhora)
- Tempo por epoch: ~23 min (vs 21 min de run 001 — DINOv2 patch14 traz 256 tokens vs 196 do patch16, e K=64 dobra retrieval)
- Tempo total treino: ~11 h
- Galeria: 33.207 pares positivos
- Trainable params: ~12 M (cross-attn + heads + sig_to_token + relation_embed; encoder frozen)

| Epoch | Val AUC | Train Loss | LR | Note |
|-------|---------|------------|-----|------|
| 1 | 0.7921 | 1.488 | 3.33e-5 | Warmup |
| **2** | **0.8213** | 1.300 | 6.67e-5 | First peak |
| 3 | 0.8076 | 1.204 | 1.00e-4 | LR peak, dip |
| 4 | 0.7993 | 1.109 | 9.99e-5 | |
| 5 | 0.8179 | 1.047 | 9.97e-5 | |
| 6 | 0.8180 | 0.996 | 9.93e-5 | |
| 7 | 0.8064 | 0.964 | 9.88e-5 | |
| 8 | 0.8186 | 0.935 | 9.81e-5 | |
| **9** | **0.8228** | 0.914 | 9.73e-5 | **New best** |
| 10 | 0.8095 | 0.898 | 9.64e-5 | Patience 1 |
| 11 | 0.8188 | 0.885 | 9.53e-5 | Patience 2 |
| 12 | 0.8181 | 0.875 | 9.40e-5 | Patience 3 |
| 13 | 0.8105 | 0.862 | 9.27e-5 | Patience 4 |
| 14 | 0.8027 | 0.852 | 9.12e-5 | Patience 5 |
| 15 | 0.8130 | 0.844 | 8.96e-5 | Patience 6 |
| 16 | 0.8113 | 0.834 | 8.78e-5 | Patience 7 |
| 17 | 0.8212 | 0.827 | 8.60e-5 | Patience 8 |
| 18 | 0.8142 | 0.817 | 8.40e-5 | Patience 9 |
| 19 | 0.8128 | 0.811 | 8.20e-5 | Patience 10 |
| 20 | 0.8211 | 0.805 | 7.98e-5 | Patience 11 |
| 21 | 0.8081 | 0.798 | 7.76e-5 | Patience 12 |
| 22 | 0.8175 | 0.788 | 7.53e-5 | Patience 13 |
| 23 | 0.8017 | 0.781 | 7.29e-5 | Patience 14 |
| 24 | 0.8142 | 0.775 | 7.04e-5 | Patience 15 |
| 25 | 0.8125 | 0.765 | 6.79e-5 | Patience 16 |
| 26 | 0.8094 | 0.759 | 6.53e-5 | Patience 17 |
| 27 | 0.8090 | 0.751 | 6.27e-5 | Patience 18 |
| 28 | 0.8176 | 0.745 | 6.00e-5 | Patience 19 |
| 29 | 0.8051 | 0.739 | 5.73e-5 | Patience 20 → stop |

### Test metrics

Threshold = **0.55** (selected on validation, applied as-is to test).

| Metric | Run 002 (DINOv2 K=64) | Run 001 (ViT K=32) | Δ |
|--------|------------:|------------:|------------:|
| Test ROC AUC | **0.7310** | 0.7763 | **−0.0453** |
| Test Accuracy | 66.18% | 69.78% | −3.6 pp |
| Balanced Accuracy | 65.82% | 70.27% | −4.5 pp |
| Test F1 | 0.6190 | 0.7223 | −0.103 |
| Test Precision | 67.23% | 64.52% | +2.7 pp |
| Test Recall | 57.35% | 82.04% | −24.7 pp |
| Average Precision | 0.6808 | 0.7345 | −0.054 |
| TAR@FAR=0.1 | 0.2974 | 0.3881 | −0.091 |
| TAR@FAR=0.01 | 0.0423 | 0.0616 | −0.019 |
| TAR@FAR=0.001 | 0.0070 | 0.0058 | +0.001 |

**Gap val → test:** 0.8228 → 0.7310 = **−0.092** (run 001: 0.836 → 0.776 = −0.060). Aumento substancial do gap — overfitting da validação.

### Per-relation accuracy (FIW, kin recall ao threshold val)

| Relation | Run 002 | Run 001 | Δ | N |
|----------|--------:|--------:|--------:|----:|
| bb | 66.86% | 86.51% | −19.7 pp | 860 |
| ss | 64.57% | 86.32% | −21.8 pp | 731 |
| sibs | 62.39% | 87.18% | −24.8 pp | 234 |
| md | 60.69% | 85.93% | −25.2 pp | 1.038 |
| fs | 55.33% | 78.77% | −23.4 pp | 1.135 |
| ms | 54.92% | 83.88% | −29.0 pp | 1.036 |
| gfgs | 51.02% | 82.65% | −31.6 pp | 98 |
| gfgd | 50.72% | 75.36% | −24.6 pp | 138 |
| gmgd | 47.97% | 63.41% | −15.4 pp | 123 |
| fd | 47.93% | 76.91% | −29.0 pp | 918 |
| gmgs | 41.32% | 61.16% | −19.8 pp | 121 |

Todas as classes pioraram em recall. Boa parte da queda vem do threshold mais agressivo (0.55 vs 0.40), que reduz positives em todo lado.

### Notes — análise do resultado negativo

1. **Backbone DINOv2 melhorou train, piorou test.** Val AUC subiu (run 001 0.836 → run 002 0.823 — *menor* peak na verdade, mas mais sustentado nas primeiras epochs) e train loss foi mais baixo, mas test piorou. DINOv2 fornece representações mais ricas de face genérica, e o head treina rápido a explorar features que se sobre-ajustam ao split de validação.
2. **K=64 amplificou o problema, não o resolveu.** A hipótese era "frozen encoder + retrieval = retrieval é o gargalo, mais K = mais contexto". Na prática, mais K significa mais correlação galeria↔val que não generaliza para test.
3. **relation_loss_weight=0.3** — não está claro se ajudou ou prejudicou isoladamente; sem ablação cruzada não dá pra atribuir.
4. **Threshold val 0.55 foi muito alto para o test.** Run 001 escolheu 0.40 e generaliza melhor. Como o protocolo proíbe reotimizar threshold no test, o número final é o que se obtém. Ainda assim, o ROC-AUC threshold-independent caiu (0.7310 vs 0.7763), então não é só calibração — o ranking piorou.
5. **Gap val→test de 0.092 é o sinal central.** Run 001 tinha 0.060. O modelo aprendeu padrões que valem na val mas não no test, sugerindo:
   - galeria fixa + cross-attention forte = memorização da galeria
   - DINOv2 features finas amplificam essa memorização
6. **Conclusão para o TCC:** este run **fortalece a tese comparativa** com um achado negativo concreto — *"trocar o backbone para DINOv2 e aumentar K não ajuda M06; o gargalo não é representação visual, é a regularização do mecanismo de retrieval"*. Inverte a hipótese inicial.

### Follow-up ideas (revistos pós run 002)

1. **Hard-negative supports** (sem mexer no backbone) — agora prioridade absoluta. Mix retrieved-but-non-kin pairs no contexto força o cross-attention a discriminar, não memorizar.
2. **Voltar para ViT-base ImageNet** com gallery cap reduzido (50k) para reduzir over-correlation gallery-val.
3. **Reduzir K** para 16 e ver se confirma que K alto piora generalização.
4. **Gallery refresh com sampling** — refresh com subset aleatório por epoch para quebrar overfit gallery-val.
5. **NÃO** tentar LoRA agora — deixaria M05 e M06 indistinguíveis.

### Artifacts

- Checkpoints: `output/002/checkpoints/{best.pt, final.pt}`
- Logs: `output/002/logs/{train.log, test.log}` (`evaluate.log` integrated nos demais)
- Results: `output/002/results/{test_metrics_rocm.json, metrics_rocm.json, per_relation.json, confusion_matrix_rocm.png, roc_curve_rocm.png, per_relation_rocm.png}`

---

## Run 001 — 2026-04-26 — Completed (early stop)

**Status:** Stopped (early stop, patience 10)
**Outcome:** Test ROC AUC = **0.776**, best Val AUC = 0.8361 at epoch 8

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  bash models/06_retrieval_augmented_kinship/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw (66.4k train / 11.4k val / 13.4k test pairs) |
| Backbone | vit_base_patch16_224 (frozen) |
| Img size | 224 |
| Batch size | 8 |
| Grad accum | 4 (effective batch = 32) |
| LR | 1e-4 |
| Scheduler | cosine, warmup=3 ep, min_lr=1e-6 |
| Weight decay | 1e-4 |
| Epochs | 20 (early-stopped at 18) |
| Patience | 10 |
| Loss | combined (BCE + 0.3 × contrastive + 0.15 × relation-CE) |
| Temperature | 0.1 |
| Retrieval K | 32 |
| Cross-attention | 2 layers, 4 heads |
| Embedding dim | 512 |
| Dropout | 0.1 |
| Gallery cap | 200,000 |
| Gallery on CPU | False |
| Gallery refresh every | 0 (built once) |
| Max grad norm | 1.0 |
| AMP | on |
| Seed | 42 |
| Workers | 4 |

### Training trajectory

- Best Val AUC: **0.8361** at epoch 8 (best.pt saved here)
- Stopped at epoch 18 (patience 10 hit)
- Trainable params: **8,155,148 / 93,953,804 (8.68%)**
- Time per epoch: ~21 min (~6 hours total + ~7 min gallery build)
- Gallery: 33,207 positive train pairs

| Epoch | Val AUC | Train Loss | LR | Note |
|-------|---------|------------|-----|------|
| 1 | 0.7820 | 1.342 | 3.33e-5 | Warmup |
| 2 | 0.8026 | 1.226 | 6.67e-5 | |
| 3 | 0.8230 | 1.187 | 1.00e-4 | LR peak |
| 4 | 0.8261 | 1.146 | 9.92e-5 | |
| 5 | 0.8336 | 1.123 | 9.67e-5 | |
| 6 | 0.8295 | 1.106 | 9.26e-5 | Patience 1 |
| 7 | 0.8287 | 1.083 | 8.71e-5 | Patience 2 |
| **8** | **0.8361** | 1.061 | 8.03e-5 | **New best** |
| 9 | 0.8258 | 1.037 | 7.26e-5 | Patience 1 |
| 10 | 0.8202 | 1.018 | 6.40e-5 | Patience 2 |
| 11 | 0.8239 | 0.999 | 5.51e-5 | Patience 3 |
| 12 | 0.8173 | 0.977 | 4.59e-5 | Patience 4 |
| 13 | 0.8256 | 0.958 | 3.70e-5 | Patience 5 |
| 14 | 0.8248 | 0.941 | 2.84e-5 | Patience 6 |
| 15 | 0.8237 | 0.921 | 2.07e-5 | Patience 7 |
| 16 | 0.8191 | 0.906 | 1.39e-5 | Patience 8 |
| 17 | 0.8243 | 0.894 | 8.41e-6 | Patience 9 |
| 18 | 0.8258 | 0.888 | 4.34e-6 | Patience 10 → stop |

### Test metrics

Threshold = 0.40 (selected on validation, applied as-is to test).

| Metric | Value |
|--------|-------|
| Test ROC AUC | **0.776** |
| Test Accuracy | 69.8% |
| Balanced Accuracy | 70.3% |
| Test F1 | 0.722 |
| Test Precision | 64.5% |
| Test Recall | 82.0% |
| Average Precision | 0.735 |
| TAR@FAR=0.1 | 0.388 |
| TAR@FAR=0.01 | 0.062 |
| TAR@FAR=0.001 | 0.006 |

Protocol-internal Test AUC was 0.7637 (uses different evaluator) — the standalone `test.py` evaluator at AUC=0.776 is the canonical figure.

### Per-relation accuracy (FIW)

| Relation | Acc | N |
|----------|-----|---|
| sibs (mixed siblings) | 87.2% | 234 |
| bb (brothers) | 86.5% | 860 |
| ss (sisters) | 86.3% | 731 |
| md (mother-daughter) | 85.9% | 1,038 |
| ms (mother-son) | 83.9% | 1,036 |
| gfgs (grandfather-grandson) | 82.7% | 98 |
| fs (father-son) | 78.8% | 1,135 |
| fd (father-daughter) | 76.9% | 918 |
| gfgd (grandfather-granddaughter) | 75.4% | 138 |
| gmgd (grandmother-granddaughter) | 63.4% | 123 |
| gmgs (grandmother-grandson) | 61.2% | 121 |

### Notes

- **Big val→test gap:** Val AUC 0.836 → Test AUC 0.776 (Δ -0.06). Larger gap than Models 02/03 — retrieval is picking up patterns specific to the train gallery that don't generalize.
- **Per-relation is much more uniform than VLMs:** all 11 classes are at 61-87%. VLMs zero-shot got 0% on grandparent classes. Retrieval-augmentation does fix the rare-class problem.
- **AUC ceiling vs parametric models:** 0.776 vs 0.850. Most likely cause: encoder is frozen — the 85.8M backbone params never see kinship.
- **Mild overfitting after ep 8:** train loss kept falling 1.061 → 0.888 while val AUC plateaued. The 8M head capacity is enough to overfit eventually.

### Follow-up ideas (ranked)

1. **LoRA on q/v projections** of the encoder (~1-3M extra params, fits 12GB) — biggest expected gain (+0.04 to +0.07 AUC).
2. **Hard-negative supports:** mix in top-K positives for non-kin queries so cross-attention sees counterfactuals. Should close val→test gap.
3. **Multi-vector retrieval (ColBERT-style):** patch-level retrieval with MaxSim, would specifically help gmgs/gmgd (61-63%).
4. **Try DINOv2 encoder** (`vit_base_patch14_dinov2.lvd142m`) — drop-in swap, often +0.01-0.02 AUC for free.
5. **Larger K** (64 or 128) — frozen encoder makes retrieval the bottleneck. Cheap to test.
6. **Per-relation thresholds** — TAR@FAR=0.01=0.062 is very low; per-class calibration would help strict thresholds.

### Artifacts

- Checkpoints: `output/001/checkpoints/{best.pt, final.pt, epoch_{5,10,15}.pt}`
- Logs: `output/001/logs/{train,test,evaluate}.log`
- Results: `output/001/results/{test_metrics_rocm.json, metrics_rocm.json, per_relation.json, confusion_matrix_rocm.png, roc_curve_rocm.png, per_relation_rocm.png}`

---

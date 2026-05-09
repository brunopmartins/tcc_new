# Run Log — Model 05: DINOv2 + LoRA + Differential Attention

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Run 007 — 2026-05-08 — Stopped manually at epoch 19 (hybrid backbone, H1/H2 not met)

**Status:** Stopped manually (best at ep 13, AUC plateaued)
**Outcome:** Test ROC AUC = **0.810**, best Val AUC = 0.9066 at epoch 13. Hipótese de superar M02 via fusão DINOv2 + M02-trained-ViT **não confirmada** — Test AUC ficou entre R001 e R004, e per-relation regrediu vs R001.

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  USE_HYBRID_BACKBONE=1 \
  FACE_BACKBONE_NAME=vit_base_patch16_224 \
  FACE_BACKBONE_CHECKPOINT=/home/bruno/Desktop/tcc_new/models/02_vit_facor_crossattn/output/031/checkpoints/epoch_5.pt \
  FACE_BACKBONE_STATE_PREFIX=vit. \
  INTRA_FACE_ATTN_LAYERS=1 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| **Backbone 1 (DINOv2)** | vit_base_patch14_dinov2.lvd142m (frozen) |
| **Backbone 2 (face-specific)** | vit_base_patch16_224 (frozen, weights from M02 R031 epoch_5.pt) |
| **Intra-face attention** | 1 layer DiffAttn (DINOv2 ↔ M02-ViT) |
| **Inter-face attention** | 2 layers DiffAttn (face1 ↔ face2) |
| Embedding dim | 512 |
| Loss | combined (BCE + 0.5 × contrastive + 0.2 × relation-CE) |
| LR | 3e-4 |
| Scheduler | cosine, warmup=3, min_lr=1e-6 |
| Weight decay | 1e-4 |
| Batch size | 4 (grad_accum 8 → eff. 32) |
| Dropout | 0.1 |
| Patience | 10 |
| Total params | 183,100,428 |
| **Trainable** | **11,576,844 (6.3%)** — both backbones frozen |
| Time/epoch | ~62 min |

### Training trajectory (selected)

| Epoch | Val AUC | Train Loss | LR | Note |
|-------|---------|------------|-----|------|
| 1 | 0.8920 | 0.369 | 1.0e-4 | Warmup |
| 2 | 0.8901 | 0.293 | 2.0e-4 | |
| 3 | 0.8945 | 0.280 | 3.0e-4 | LR pico |
| 4 | 0.8955 | 0.251 | 3.0e-4 | |
| 5 | 0.8982 | 0.234 | 2.9e-4 | |
| 6 | 0.9004 | 0.216 | 2.8e-4 | |
| 7 | 0.9023 | 0.198 | 2.6e-4 | |
| 8 | 0.8978 | 0.183 | 2.4e-4 | Patience 1 |
| 9 | 0.9010 | 0.167 | 2.2e-4 | |
| 10 | 0.8996 | 0.151 | 1.9e-4 | |
| 11 | 0.9045 | 0.135 | 1.6e-4 | |
| 12 | 0.8956 | 0.119 | 1.4e-4 | |
| **13** | **0.9066** | 0.104 | 1.1e-4 | **Best** |
| 14 | 0.8954 | 0.088 | 8.4e-5 | Patience 1 |
| 15 | 0.9013 | 0.075 | 6.0e-5 | |
| 16 | 0.8969 | 0.064 | 4.0e-5 | |
| 17 | 0.8984 | 0.052 | 2.3e-5 | |
| 18 | 0.8992 | 0.044 | 1.1e-5 | Manual halt |

### Test metrics

| Metric | R007 | R001 | R005 | M02 R031 |
|--------|----:|-----:|-----:|---------:|
| Test ROC AUC | 0.810 | 0.806 | 0.822 | **0.850** |
| Test Accuracy | 71.9% | 72.6% | 72.0% | **74.4%** |
| Test F1 | 0.652 | 0.713 | 0.645 | 0.779 |
| Test Precision | **80.0%** | 64.5% | 82.2% | 66.5% |
| Test Recall | 55.1% | 82.0% | 53.0% | 94.1% |
| Avg Precision | 0.796 | 0.792 | 0.798 | 0.817 |
| TAR@FAR=0.1 | 0.495 | 0.463 | 0.511 | 0.499 |
| TAR@FAR=0.01 | 0.136 | 0.152 | 0.100 | ~0.13 |
| Val→Test gap | -0.097 | -0.105 | -0.071 | -0.031 |
| Threshold | 0.50 | 0.10 | 0.50 | 0.90 |

### Per-relation accuracy (FIW)

| Relation | N | R007 | R001 | M02 R031 |
|----------|---|----:|----:|---------:|
| sibs | 234 | 60.7% | 83.3% | 94.9% |
| bb | 860 | 57.8% | 79.8% | 95.5% |
| fs | 1,135 | 59.1% | 71.6% | 95.3% |
| fd | 918 | 57.0% | 71.5% | 91.7% |
| ss | 731 | 57.2% | 77.2% | 94.7% |
| ms | 1,036 | 56.8% | 69.4% | 93.9% |
| md | 1,038 | 53.4% | 67.3% | 94.4% |
| gfgd | 138 | 41.3% | 50.7% | 89.9% |
| **gfgs** | 98 | 32.7% | 39.8% | **95.9%** |
| gmgd | 123 | 27.6% | 40.7% | 91.1% |
| gmgs | 121 | 24.0% | 52.1% | 88.4% |

### Notes — analysis of negative result

1. **H0 (null) confirmado.** Test AUC=0.810 está estatisticamente indistinguível de R001 (0.806) e R004 (0.812). A combinação DINOv2 + M02-ViT congelados **não gera ganho material**.

2. **H1 (≥0.83) e H2 (≥0.85) NÃO confirmadas.** O ganho esperado da fusão de backbones complementares não materializou.

3. **H3 (M02 backbone domina) parcialmente observado**, mas com twist: o M02-ViT extraído NÃO consegue produzir as features fortes que M02 R031 mostrava. Em M02, os 95.9% de accuracy em gfgs vinham do **sistema completo treinado-em-conjunto** (ViT + cross-attention bidirecional FaCoR + supervised contrastive + threshold 0.90). Reusar só os pesos do ViT e plugar em outra arquitetura quebra o ecossistema.

4. **Per-relation regrediu vs R001 em todas as 11 classes.** R007 nem sequer manteve o nível do R001 frozen+LoRA — o threshold 0.50 (vs R001 0.10) é fator, mas a estrutura de scores é diferente: R007 produz scores polarizados (precisão 80%) com baixo recall (55%). Modelo extremamente conservador.

5. **TAR@FAR=0.01 = 0.136** é razoável (entre R001's 0.152 e R004's 0.119). Em regimes de alta precisão, a fusão preserva alguma discriminação.

6. **Val→Test gap = -0.097** confirma o padrão estrutural — 7 runs M05, gap consistentemente em -0.07 a -0.11. **Não é overfitting do modelo, é o split RFIW Track-I.**

### Lições gerais (após 7 runs M05)

- **Backbones face-specific isolados não compõem com DINOv2** sem retreinar todo o pipeline juntos. M02's success era do sistema, não do ViT.
- **O teto do M05 é 0.81-0.82 em qualquer configuração frozen** (R001, R002, R003, R004, R006, R007 todos nessa faixa).
- **Full fine-tune (R005)** moveu o teto marginalmente para 0.822 — única intervenção que mexeu o ranking de teste.
- **Para superar 0.85**, seria necessário ou (a) full fine-tune com receita CosFace/ArcFace, ou (b) trocar arquitetura (M07).

### Bug encontrado e corrigido

`test.py` e `evaluate.py` instanciavam apenas `DINOv2LoRAKinship`, ignorando o caso hybrid. Com `strict=False` no `load_state_dict`, os pesos `face_vit.*` e `intra_face_layers.*` eram silenciosamente descartados. **Primeira execução do test.py em R007 deu Test AUC = 0.504 (random)** — bug catastrófico. Corrigido: ambos scripts agora detectam arquitetura hybrid pelos prefixos do state_dict e instanciam `DINOv2HybridKinship` quando apropriado. Re-execução produziu Test AUC = 0.810 (resultado real).

### Artifacts

- Checkpoints: `output/007/checkpoints/{best.pt, epoch_5.pt, epoch_10.pt, epoch_15.pt}`
- Logs: `output/007/logs/{train.log, test.log, evaluate.log}`
- Results: `output/007/results/{test_metrics_rocm.json, metrics_rocm.json, per_relation.json, *.png}`
- Pipeline log: `output/run_007.log`

---

## Run 006 — 2026-05-07 — Completed naturally (heavy contrastive loss)

**Status:** Completed (full 20 epochs, natural early stop pattern)
**Outcome:** Test ROC AUC = **0.814**, best Val AUC = 0.9115 at epoch 4. Hipótese de loss contrastive-heavy bater R001 não confirmada.

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  LOSS=combined CONTRASTIVE_WEIGHT=0.9 RELATION_LOSS_WEIGHT=0 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

Versão corrigida da Opção A (LOSS=contrastive puro deixou binary head sem treino → AUC random). Combined com 0.9 peso em contrastive + 0.1 em BCE mantém head viva enquanto contrastive domina.

### Configuration deltas vs R001

| Param | R001 | R006 |
|-------|------|------|
| LOSS | combined | combined |
| CONTRASTIVE_WEIGHT | 0.5 | **0.9** |
| RELATION_LOSS_WEIGHT | 0.2 | **0** |

### Test metrics

| Metric | R006 | R001 |
|--------|----:|-----:|
| Test ROC AUC | 0.814 | 0.806 |
| Test Accuracy | 72.9% | 72.6% |
| Test F1 | 0.721 | 0.713 |
| TAR@FAR=0.01 | 0.094 | 0.152 |
| Val AUC peak | 0.9115 (ep 4) | 0.9116 (ep 12) |
| Val→Test gap | -0.10 | -0.105 |

### Per-relation highlights

- bb: **83.4%** (best M05), sibs: **86.8%** (best M05)
- gfgs: 45.9%, gmgd: 39.8% (worse than R001 on grandparents)

### Notes

- Heavy contrastive ajudou classes intra-geração (bb, ss, sibs) mas regrediu em avô-neto.
- Test AUC marginal +0.008 vs R001 — dentro de ruído.
- Confirma que a loss não é o gargalo principal do M05.

### Artifacts

- `output/006/checkpoints/best.pt` (~700MB), final.pt, epoch_5/10.pt
- `output/006/results/{test_metrics_rocm.json, per_relation.json, *.png}`

---

## Run 005 — 2026-05-04 — Stopped manually at epoch 10 (full unfreeze, M02-style LR)

**Status:** Stopped manually (slow convergence due to LR=5e-6 with random head)
**Outcome:** Test ROC AUC = **0.822** (highest M05), best Val AUC = 0.8930 at epoch 12. Val→test gap = **-0.071** (smallest M05).

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  UNFREEZE_BACKBONE_BLOCKS=12 BACKBONE_LR_FACTOR=1.0 \
  LEARNING_RATE=5e-6 WARMUP_EPOCHS=5 DROPOUT=0.2 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Backbone | DINOv2 ViT-B/14 (full fine-tune) |
| **Trainable** | **93.5M (99.3%)** |
| LR | 5e-6 (M02-style) |
| Warmup | 5 epochs |
| Dropout | 0.2 |
| LoRA rank/alpha | 8 / 16 (still active as overlay) |
| Loss | combined |

### Notes

- Initial 5 epochs were broken (Val AUC stuck at 0.5-0.7) due to LR=5e-6 being too low for un-trained head. Recovered at epoch 6 (Val AUC 0.84) when warmup completed.
- **Best M05 Test AUC (0.822)** but at the cost of much heavier compute.
- Val→test gap shrunk from -0.10 to -0.07 with full plasticity — partial evidence that the gap is influenced by capacity, not purely structural.
- TAR@FAR=0.1 = 0.511 (best of M05).

### Artifacts

- `output/005/checkpoints/best.pt`, final.pt
- `output/005/results/{test_metrics_rocm.json, per_relation.json, *.png}`

---

## Run 004 — 2026-04-30 — Stopped manually at epoch 7 (partial unfreeze, last 4 DINOv2 blocks)

**Status:** Stopped manually after overfitting signature
**Outcome:** Test ROC AUC = **0.812**, best Val AUC = 0.9108 at epoch 4.

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  UNFREEZE_BACKBONE_BLOCKS=4 BACKBONE_LR_FACTOR=0.01 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Backbone | DINOv2 ViT-B/14 (last 4 blocks unfrozen) |
| **Trainable** | **36.8M (39.1%)** |
| LR | 3e-4 (head + LoRA), 3e-6 (backbone group via factor=0.01) |

### Notes

- Implementation introduced multi-LR optimizer (head_lr + backbone_lr) and `SequentialLR(LinearLR + CosineAnnealingLR)` for warmup that respects per-group LRs.
- Stopped early because train loss collapsed (overfit signature) while Val AUC plateaued.
- Test AUC 0.812 — marginal +0.006 vs R001.

### Artifacts

- `output/004/checkpoints/best.pt`
- `output/004/results/{test_metrics_rocm.json, per_relation.json, *.png}`

---

## Run 003 — 2026-05-04 — Stopped manually at epoch 10 (regression on per-relation, AUC flat)

**Status:** Stopped manually (best at epoch 5; user halted before patience triggered)
**Outcome:** Test ROC AUC = **0.809**, best Val AUC = **0.9091** at epoch 5. **Hipotese de starvation rejeitada** — sampler estratificado nao melhorou classes de avo (gfgs caiu de 39.8% para 28.6%) e degradou todas as outras classes.

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=40 PATIENCE=15 NUM_WORKERS=4 \
  STRATIFIED_SAMPLER=1 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

### Configuration (deltas vs Run 001 in **bold**)

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbone | vit_base_patch14_dinov2.lvd142m (frozen) |
| Img size | 224 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.0 |
| Cross-attn | 2 layers, 8 heads (differential) |
| Embedding dim | 512 |
| Dropout | 0.1 |
| Batch size | 4 |
| Grad accum | 8 (effective batch = 32) |
| LR | 3e-4 |
| Scheduler | cosine, warmup=3, min_lr=1e-6 |
| Weight decay | 1e-4 |
| Epochs | 40 (stopped at 10) |
| Patience | 15 |
| Loss | combined (BCE + 0.5 × contrastive + 0.2 × relation-CE) |
| Temperature | 0.1 |
| Relation set | fiw (11 classes) |
| **Sampler** | **WeightedRandomSampler com 50% positivos divididos igualmente entre 11 relacoes + 50% negativos** (`--stratified_sampler`) |
| AMP | on |
| Grad checkpoint | on |
| Max grad norm | 1.0 |
| Seed | 42 |
| Workers | 4 |

### Sampler effect verified empirically (one epoch)

| Grupo | Frequencia natural | Frequencia pos-sampler | Multiplicador |
|-------|-------------------:|-----------------------:|--------------:|
| non-kin | 50.00% | 49.94% | 1.0× |
| md / fs / fd / ms (pais) | 7.7-8.1% | ~4.5% | ~0.6× |
| sibs / ss / bb (irmaos) | 4.6-5.1% | ~4.5% | ~0.9× |
| **gfgs** | **1.03%** | **4.54%** | **4.4×** |
| **gmgs** | 0.94% | 4.54% | 4.8× |
| **gfgd** | 0.92% | 4.65% | 5.0× |
| **gmgd** | 0.82% | 4.53% | 5.5× |

A intervencao tecnica funcionou — classes de avo receberam 4-6× mais updates por epoch.

### Training trajectory

- Best Val AUC: **0.9091** at epoch 5
- Stopped manually at epoch 10 (user halted; not patience-driven)
- Trainable params: 8,467,980 / 94,192,908 (8.99%) — same as R001
- Time per epoch: ~85 min

| Epoca | Val AUC R003 | Val AUC R001 | Δ | Train Loss R003 | Note |
|-------|-------------:|-------------:|----:|----------------:|------|
| 1 | 0.8798 | 0.9040 | -0.024 | 0.409 | Warmup |
| 2 | 0.8971 | 0.9045 | -0.007 | 0.273 | |
| 3 | 0.8926 | 0.9001 | -0.008 | 0.254 | LR pico |
| 4 | 0.9035 | 0.9058 | -0.002 | 0.222 | |
| **5** | **0.9091** | **0.9085** | **+0.001** | 0.200 | **Best — R003 ultrapassa R001** |
| 6 | 0.8930 | 0.9028 | -0.010 | 0.183 | Patience 1 |
| 7 | 0.8904 | 0.9051 | -0.015 | 0.169 | Patience 2 |
| 8 | 0.9006 | 0.9050 | -0.004 | 0.161 | Patience 3 |
| 9 | 0.8998 | 0.8964 | +0.003 | 0.153 | Patience 4 |
| 10 | 0.8997 | 0.9077 | -0.008 | 0.148 | Patience 5 → user halt |

R003 alcancou paridade com R001 em val AUC peak (0.9091 vs 0.9116 final do R001 — diferenca de 0.003). Train loss consistentemente menor que R001 — sampler maximiza sinal por step.

### Test metrics

Threshold = **0.50** (selected on validation, applied as-is to test). R001 used 0.10, R002 used 0.30. **Threshold drifou progressivamente conforme regularizacao/balancing aumenta.**

| Metric | Run 003 | Run 001 | Run 002 | R003 vs R001 |
|--------|--------:|--------:|--------:|-------------:|
| Test ROC AUC | **0.809** | 0.806 | 0.799 | +0.003 |
| Test Accuracy | 71.0% | 72.6% | 72.4% | -1.6 pp |
| Balanced Accuracy | 70.5% | 70.3% | 72.6% | +0.2 pp |
| Test F1 | 0.653 | 0.713 | 0.718 | -0.060 |
| Test Precision | **76.8%** | 64.5% | 70.3% | **+12.3 pp** |
| Test Recall | 56.7% | 82.0% | 73.3% | -25.3 pp |
| Average Precision | 0.778 | 0.792 | 0.772 | -0.014 |
| TAR@FAR=0.1 | 0.459 | 0.463 | 0.437 | -0.004 |
| TAR@FAR=0.01 | **0.098** | **0.152** | 0.095 | **-0.057** |
| TAR@FAR=0.001 | 0.011 | 0.044 | 0.017 | -0.033 |
| Threshold | 0.50 | 0.10 | 0.30 | — |

**Gap val→teste:** R003: 0.9091 → 0.809 = **-0.100**. R001: 0.9116 → 0.806 = -0.105. Gap ainda em torno de -0.10, igual ao R001/R002 — confirma que **stronger sampler + higher precision tradeoff nao mexe no gap estrutural**.

### Per-relation accuracy (FIW)

| Relation | N | R003 | R001 | R002 | Δ R003 vs R001 |
|----------|---|----:|----:|----:|---:|
| sibs | 234 | 63.2% | 83.3% | 85.9% | **-20.1 pp** |
| bb | 860 | 62.6% | 79.8% | 85.5% | **-17.2 pp** |
| ss | 731 | 61.4% | 77.2% | 81.1% | -15.8 pp |
| fs | 1,135 | 62.6% | 71.6% | 73.8% | -9.0 pp |
| fd | 918 | 62.3% | 71.5% | 72.9% | -9.2 pp |
| ms | 1,036 | 51.6% | 69.4% | 69.8% | -17.8 pp |
| md | 1,038 | 50.0% | 67.3% | 70.3% | -17.3 pp |
| **gmgs** | 121 | 43.0% | 52.1% | 52.1% | -9.1 pp |
| **gfgd** | 138 | 39.1% | 50.7% | 49.3% | -11.6 pp |
| **gmgd** | 123 | 35.8% | 40.7% | 45.5% | -4.9 pp |
| **gfgs** | 98 | **28.6%** | **39.8%** | 38.8% | **-11.2 pp** |

**Todas as 11 classes regrediram**, incluindo as 4 classes de avo que o sampler era para ajudar. Parte da queda e artefato do threshold mais alto (0.50 vs 0.10), mas TAR@FAR=0.01 caiu 36% (0.152 → 0.098) — esse e threshold-independent, confirmando que o ranking efetivo piorou na regiao de alta precisao.

### Notes — analise do resultado

1. **Hipotese central rejeitada.** Sampler que entrega 4-6× mais sinal as classes de avo nao melhorou per-relation nessas classes (gfgs caiu 39.8% → 28.6%, gmgd 40.7% → 35.8%). **Data starvation nao e o gargalo principal.**

2. **Confound metodologico.** A implementacao mistura duas mudancas: (a) balanco entre 11 classes positivas + (b) ratio pos/neg shift de 33/67 para 50/50. A versao limpa preservaria 2:1 e so rebalancearia dentro dos positivos. Esse confound nao explica o resultado isoladamente, mas o muddla.

3. **Threshold drift sistematico.** R001=0.10 → R002=0.30 → R003=0.50. Cada intervencao de regularizacao/balanco diminui a polarizacao da distribuicao de scores. O modelo se torna progressivamente menos confiante. Isso baixa recall + sobe precision, espelhando exatamente o que se ve em R003 (precision 76.8% — maior do projeto, recall 56.7% — menor).

4. **Implicacao para gargalo arquitetonico.** Combinando 3 runs de M05 + 3 hipoteses rejeitadas (defaults, regularizacao, balanco de dados), a leitura mais consistente e: o **DINOv2 frozen + LoRA tem teto arquitetonico em ~0.81 AUC nesta tarefa**. Mais sinal sobre features fixas nao gera informacao que as features nao codificam.

5. **Implicacao para split RFIW.** Os tres runs sustentam val→teste gap em torno de -0.10. Convergencia: o gap e **propriedade do split**, nao do modelo. R002 (mais regularizacao) e R003 (mais balanco) ambos falharam em fechar o gap.

6. **TAR@FAR=0.01 regrediu permanentemente** vs R001 nesta direcao de hiperparametros. Para uso de alta precisao (verificacao biometrica), R001 fica como o melhor checkpoint do M05 — provavelmente o melhor checkpoint do projeto inteiro.

### Follow-up — Run 004 ja em planejamento

Proxima fronteira: descongelamento parcial dos ultimos 4 blocos do DINOv2 com LR diferenciado. Se features fixas sao o teto, descongelar ataca o teto diretamente. Manter R001 hyperparams + adicionar:

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  UNFREEZE_BACKBONE_BLOCKS=4 BACKBONE_LR_FACTOR=0.01 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

Hipotese: Test AUC sobe para 0.83-0.86 (recuperar o que M02 R031 atingiu com full fine-tune do ImageNet ViT). Se mesmo isso nao quebrar o teto, o limite e o split — aceita-se M05 R001 como ponto de Pareto e move-se para outras direcoes.

### Artifacts

- Checkpoints: `output/003/checkpoints/{best.pt, epoch_5.pt, epoch_10.pt}`
- Logs: `output/003/logs/{train.log, test.log, evaluate.log}`
- Results: `output/003/results/{test_metrics_rocm.json, metrics_rocm.json, per_relation.json, *.png}`
- Pipeline log: `output/run_003.log`

---

## Run 002 — 2026-04-29 — Completed (early stop, regression vs Run 001)

**Status:** Stopped (early stop, patience 10)
**Outcome:** Test ROC AUC = **0.799**, best Val AUC = **0.9048** at epoch 4. Hipotese de fechar o val→teste gap **nao se confirmou** — gap manteve-se em -0.106, e TAR@FAR=0.01 caiu de 0.152 para 0.095.

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=40 PATIENCE=10 NUM_WORKERS=4 \
  RELATION_LOSS_WEIGHT=0.4 LORA_DROPOUT=0.1 DROPOUT=0.2 \
  LEARNING_RATE=1.5e-4 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

### Configuration (deltas vs Run 001 in **bold**)

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbone | vit_base_patch14_dinov2.lvd142m (frozen) |
| Img size | 224 |
| LoRA rank / alpha / **dropout** | 8 / 16 / **0.1** (was 0.0) |
| Cross-attn | 2 layers, 8 heads (differential) |
| Embedding dim | 512 |
| **Dropout** | **0.2** (was 0.1) |
| Batch size | 4 |
| Grad accum | 8 (effective batch = 32) |
| **LR** | **1.5e-4** (was 3e-4) |
| Scheduler | cosine, warmup=3, min_lr=1e-6 |
| Weight decay | 1e-4 |
| Epochs | 40 (early-stopped at 14) |
| Patience | 10 |
| Loss | combined (BCE + 0.5 × contrastive + **0.4 × relation-CE**) (was 0.2) |
| Temperature | 0.1 |
| Relation set | fiw (11 classes) |
| AMP | on |
| Grad checkpoint | on |
| Max grad norm | 1.0 |
| Seed | 42 |
| Workers | 4 |

### Training trajectory

- Best Val AUC: **0.9048** at epoch 4 (best.pt saved here) — vs R001's 0.9116 at ep 12
- Stopped at epoch 14 (patience 10) — much shorter than R001's 22 epochs
- Trainable params: 8,467,980 / 94,192,908 (8.99%) — same as R001
- Time per epoch: ~86 min (~20h total)

| Epoch | Val AUC | Train Loss | LR | Note |
|-------|---------|------------|-----|------|
| 1 | 0.8447 | 0.670 | 5.00e-5 | Warmup, slower start vs R001 (0.9040) |
| 2 | 0.9016 | 0.455 | 1.00e-4 | |
| 3 | 0.9023 | 0.376 | 1.50e-4 | LR peak (was 3e-4 in R001) |
| **4** | **0.9048** | 0.304 | 1.50e-4 | **New best** (R001 had 0.9058 here) |
| 5 | 0.8983 | 0.255 | 1.49e-4 | Patience 1 |
| 6 | 0.8982 | 0.221 | 1.48e-4 | Patience 2 |
| 7 | 0.8980 | 0.192 | 1.46e-4 | Patience 3 |
| 8 | 0.8979 | 0.171 | 1.43e-4 | Patience 4 |
| 9 | 0.8912 | 0.153 | 1.41e-4 | Patience 5 |
| 10 | 0.9047 | 0.133 | 1.37e-4 | Patience 6 — near best |
| 11 | 0.8995 | 0.118 | 1.33e-4 | Patience 7 |
| 12 | 0.8969 | 0.108 | 1.29e-4 | Patience 8 (R001 peaked here at 0.9116) |
| 13 | 0.9012 | 0.094 | 1.25e-4 | Patience 9 |
| 14 | 0.8973 | 0.085 | 1.20e-4 | Patience 10 → stop |

### Test metrics

Threshold = **0.30** (selected on validation, applied as-is to test). R001 used 0.10.

| Metric | Run 002 | Run 001 | Δ |
|--------|---------:|---------:|---------:|
| Test ROC AUC | **0.799** | 0.806 | **-0.007** |
| Test Accuracy | 72.4% | 72.6% | -0.2 pp |
| Balanced Accuracy | 72.4% | 72.6% | -0.2 pp |
| Test F1 | 0.718 | 0.713 | +0.005 |
| Test Precision | 70.3% | 71.7% | -1.4 pp |
| Test Recall | 73.3% | 70.8% | +2.5 pp |
| Average Precision | 0.772 | 0.792 | -0.020 |
| TAR@FAR=0.1 | 0.437 | 0.463 | -0.026 |
| TAR@FAR=0.01 | **0.095** | **0.152** | **-0.057** |
| TAR@FAR=0.001 | 0.017 | 0.044 | -0.027 |

**Gap val → teste:** R002: 0.9048 → 0.799 = **-0.106**. R001: 0.9116 → 0.806 = -0.105. **Gap praticamente identico** — a regularizacao mais forte nao fechou o gap, apenas baixou simultaneamente val e teste.

### Per-relation accuracy (FIW)

| Relation | Run 002 | Run 001 | Δ | N |
|----------|--------:|--------:|--------:|----:|
| sibs | 85.9% | 83.3% | +2.6 pp | 234 |
| bb | 85.5% | 79.8% | +5.7 pp | 860 |
| ss | 81.1% | 77.2% | +3.9 pp | 731 |
| fs | 73.8% | 71.6% | +2.2 pp | 1.135 |
| fd | 72.9% | 71.5% | +1.4 pp | 918 |
| md | 70.3% | 67.3% | +3.0 pp | 1.038 |
| ms | 69.8% | 69.4% | +0.4 pp | 1.036 |
| gmgs | 52.1% | 52.1% | 0.0 pp | 121 |
| gfgd | 49.3% | 50.7% | -1.4 pp | 138 |
| gmgd | 45.5% | 40.7% | +4.9 pp | 123 |
| gfgs | 38.8% | 39.8% | -1.0 pp | 98 |

Mesma estrutura de R001: relacoes intra-geracao (bb, ss, sibs) e mae/pai-filho(a) bem; classes avo/avoa quase em random. Pequenos ganhos em bb/ss/sibs/gmgd, regressao mantida em gfgs.

### Notes — analise do resultado

1. **Hipotese central rejeitada.** Aumentar regularizacao (LR/2 + dropout 0.1→0.2 + lora_dropout 0.0→0.1) **nao fechou o val→teste gap**. O gap continua em -0.106 (R001 era -0.105). Apenas baixou val e teste em paralelo (-0.007 cada).
2. **TAR@FAR=0.01 caiu drasticamente** de 0.152 para 0.095. R002 perde a unica metrica em que R001 era a melhor de todos os modelos do projeto.
3. **Relation_loss_weight 0.4 nao consertou as classes de avo/avoa.** gfgs ainda em 38.8%, gmgd em 45.5%. Sinal multitarefa mais forte ajudou bb/ss/sibs (+3-6 pp) mas nao as classes raras. **Hipotese:** as classes raras nao sao limitadas pelo sinal multitarefa, e sim por insuficiencia de exemplos no treino + tarefa intrinsecamente ambigua (idade/genero ambivalentes).
4. **Per-classe ganhou uniformidade** (intra-geracao subiu, gmgd subiu) ao custo de TAR@FAR rigoroso. Tradeoff: o modelo distribui confianca em vez de concentrar.
5. **LR=1.5e-4 + grad accum 32 + LoRA capacity rank 8 com dropout 0.1**: o pico Val AUC chegou 8 epochs antes de R001 (ep 4 vs ep 12). Isso sugere que o problema nao e capacidade — o modelo ja converge ao seu ceiling cedo, e treinar mais so amplifica overfitting.
6. **Conclusao para o TCC:** R002 e um achado negativo concreto que **desfaz a interpretacao "DINOv2 sobre-ajusta a validacao"**. Se fosse so overfitting, R002 (mais regularizado) deveria fechar o gap. Como nao fechou, o gap e estrutural — provavelmente **divergencia entre as distribuicoes de familia val e teste**, nao uma propriedade do modelo. Isso e um achado relevante: o protocolo do FIW (RFIW Track-I) tem familias disjuntas entre val e teste, e as familias da val podem ter aparencia visual mais discriminavel que as do teste.

### Follow-up ideas (revistos pos R002)

1. **Investigar a hipotese estrutural do gap:** computar Val AUC vs Test AUC para multiple seeds (e.g. 42, 0, 7). Se o gap for consistente, e propriedade do split, nao do modelo.
2. **Cross-validation 5-fold em FIW** — eliminaria o vies de um unico split val/teste. Cara mas decisiva.
3. **Ablation** — desligar relation head (`RELATION_LOSS_WEIGHT=0`) para isolar seu efeito. R001/R002 nao mostram diferenca clara.
4. **Tentar M05 sem differential attention** (substituir por cross-attn vanilla do FaCoR) — separar contribuicao de DiffAttn vs DINOv2 vs LoRA.
5. **NAO** explorar mais variantes de regularizacao no mesmo eixo — o resultado mostra que a fronteira nao esta la.

### Artifacts

- Checkpoints: `output/002/checkpoints/{best.pt, final.pt}`
- Logs: `output/002/logs/{train,test,evaluate}.log`
- Results: `output/002/results/{test_metrics_rocm.json, metrics_rocm.json, per_relation.json, *.png}`
- Pipeline log: `output/run_002.log`

---

## Run 001 — 2026-04-27 — Completed (early stop)

**Status:** Stopped (early stop)
**Outcome:** Test ROC AUC = **0.806**, best Val AUC = **0.9116** at epoch 12 — large val→test gap (~-0.105).

### Launch command

```bash
SKIP_INSTALL=1 EPOCHS=40 PATIENCE=15 NUM_WORKERS=4 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

### Configuration

| Param | Value |
|-------|-------|
| Dataset | fiw |
| Backbone | vit_base_patch14_dinov2.lvd142m (frozen) |
| Img size | 224 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.0 |
| Cross-attn | 2 layers, 8 heads (differential attention) |
| Embedding dim | 512 |
| Dropout | 0.1 |
| Batch size | 4 |
| Grad accum | 8 (effective batch = 32) |
| LR | 3e-4 |
| Scheduler | cosine, warmup=3, min_lr=1e-6 |
| Weight decay | 1e-4 |
| Epochs | 40 (early-stopped at 22) |
| Patience | 15 (stopped earlier — see notes) |
| Loss | combined (BCE + 0.5 × contrastive + 0.2 × relation-CE) |
| Temperature | 0.1 |
| Relation set | fiw (11 classes) |
| AMP | on |
| Grad checkpoint | on |
| Max grad norm | 1.0 |
| Seed | 42 |
| Workers | 4 |

### Training trajectory

- Best Val AUC: **0.9116** at epoch 12 (best.pt saved here)
- Stopped at epoch 22
- Trainable params: **8,467,980 / 94,192,908 (8.99%)**
- Time per epoch: ~84 min (~31h total)

| Epoch | Val AUC | Train Loss | LR | Note |
|-------|---------|------------|-----|------|
| 1 | 0.9040 | 0.430 | 1.00e-4 | Warmup |
| 2 | 0.9045 | 0.312 | 2.00e-4 | |
| 3 | 0.9001 | 0.288 | 3.00e-4 | LR peak |
| 4 | 0.9058 | 0.251 | 2.99e-4 | |
| 5 | 0.9085 | 0.228 | 2.98e-4 | |
| 6 | 0.9028 | 0.211 | 2.95e-4 | |
| 7 | 0.9051 | 0.196 | 2.91e-4 | |
| 8 | 0.9050 | 0.186 | 2.87e-4 | |
| 9 | 0.8964 | 0.173 | 2.81e-4 | |
| 10 | 0.9077 | 0.164 | 2.74e-4 | |
| 11 | 0.9023 | 0.155 | 2.67e-4 | |
| **12** | **0.9116** | 0.148 | 2.58e-4 | **New best** |
| 13 | 0.9041 | 0.137 | 2.49e-4 | Patience 1 |
| 14 | 0.9039 | 0.131 | 2.39e-4 | Patience 2 |
| 15 | 0.9002 | 0.122 | 2.29e-4 | Patience 3 |
| 16 | 0.9025 | 0.115 | 2.18e-4 | Patience 4 |
| 17 | 0.9028 | 0.107 | 2.06e-4 | Patience 5 |
| 18 | 0.9034 | 0.097 | 1.94e-4 | Patience 6 |
| 19 | 0.8966 | 0.092 | 1.82e-4 | Patience 7 |
| 20 | 0.8978 | 0.081 | 1.69e-4 | Patience 8 |
| 21 | 0.9037 | 0.073 | 1.57e-4 | Patience 9 |
| 22 | 0.8962 | 0.066 | 1.44e-4 | Patience 10 → stop |

Note: trainer stopped at patience=10, not the configured 15 — likely the trainer base class has a hard-coded threshold. Investigate before next run.

### Test metrics

Threshold = 0.10 (selected on validation, applied as-is to test).

| Metric | Value |
|--------|-------|
| Test ROC AUC | **0.806** |
| Test Accuracy | 72.6% |
| Balanced Accuracy | 72.6% |
| Test F1 | 0.713 |
| Test Precision | 71.7% |
| Test Recall | 70.8% |
| Average Precision | 0.792 |
| TAR@FAR=0.1 | 0.463 |
| TAR@FAR=0.01 | **0.152** |
| TAR@FAR=0.001 | 0.044 |

### Per-relation accuracy (FIW)

| Relation | Acc | N |
|----------|-----|---|
| sibs (mixed siblings) | 83.3% | 234 |
| bb (brothers) | 79.8% | 860 |
| ss (sisters) | 77.2% | 731 |
| fs (father-son) | 71.6% | 1,135 |
| fd (father-daughter) | 71.5% | 918 |
| ms (mother-son) | 69.4% | 1,036 |
| md (mother-daughter) | 67.3% | 1,038 |
| gmgs (grandmother-grandson) | 52.1% | 121 |
| gfgd (grandfather-granddaughter) | 50.7% | 138 |
| gmgd (grandmother-granddaughter) | 40.7% | 123 |
| gfgs (grandfather-grandson) | 39.8% | 98 |

### Notes

- **Strongest val AUC ever** for this project (0.9116) — DINOv2 + LoRA + DiffAttn pulled +0.06 over Models 02/03 (0.85).
- **But the val→test gap is the largest of any run:** 0.9116 → 0.806 = **−0.105**. Model 02 typically has -0.03; Model 06 R001 had -0.06. This suggests:
  - DINOv2's rich features overfit to the validation split's family distribution
  - The relation head (λ=0.2) may have helped on val but did not generalize
  - Threshold 0.10 is very low — score distributions are tight
- **TAR@FAR=0.01 = 0.152 is the best across all models** (M02/M03 ~0.13, M06 ~0.06). At strict thresholds Model 05 holds up better — useful for high-precision regimes.
- **Per-relation: same weakness as M02/M03 on grandparent classes (40-52%).** The relation head did not solve the grandparent problem despite λ=0.2. Notably **gfgs is the worst (39.8%)** — opposite of M02 R031 where gfgs was the best (95.9%). Different failure mode.
- **Overfitting visible from epoch 13 onwards:** train loss continued to fall from 0.137 to 0.066 while val AUC plateaued/declined. Patience 10 firing was correct.
- **Time budget heavy:** 84 min/epoch is ~3-4× M03/M06. DINOv2 + 8h cross-attention + diffattn + grad checkpoint = expensive.

### Follow-up ideas (ranked)

1. **Reduce overfitting + close val→test gap** — the priority. Try:
   - increase `RELATION_LOSS_WEIGHT` to 0.4-0.5 (was 0.2)
   - add stronger train-time regularization: `LORA_DROPOUT=0.1`, `DROPOUT=0.2`
   - or reduce LoRA capacity: `LORA_RANK=4` (currently 8)
2. **Lower peak LR.** 3e-4 may be too aggressive for LoRA — try 1.5e-4.
3. **Earlier stopping.** Best was ep 12 — patience=8 would have stopped at ep 20 saving ~30% time.
4. **Investigate grandparent failures.** Per-relation breakdown shows the model is confidently wrong on gfgs/gmgd/gfgd/gmgs (40-52%). The relation head should fix this but didn't.
5. **Eval on KinFaceW** — different generalization test (smaller, less family-skewed).

### Artifacts

- Checkpoints: `output/001/checkpoints/{best.pt, final.pt}`
- Logs: `output/001/logs/{train,test,evaluate}.log`
- Results: `output/001/results/{test_metrics_rocm.json, metrics_rocm.json, per_relation.json, *.png}`
- Pipeline log: `output/run_001.log`

---

For new runs, follow the template in [prompts/train_model.md](../../prompts/train_model.md) (local) and append entries above this line.

---

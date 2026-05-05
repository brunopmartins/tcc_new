# Overview — DINOv2 + LoRA + Differential Cross-Attention

**Model:** 05 — DINOv2 ViT-B/14 (frozen) + LoRA rank=8 + differential cross-attention + auxiliary relation head
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (RFIW Track-I)

Individual run details: [run-001.md](run-001.md) · [run-002.md](run-002.md) · [run-003.md](run-003.md)

---

## Run Comparison

| | Run 001 | Run 002 | Run 003 |
|---|---|---|---|
| **Date** | 2026-04-27 | 2026-04-29 | 2026-05-04 |
| **Purpose** | Defaults, design hypothesis | Stronger regularization closes val→test gap? | Stratified sampler closes grandparent gap? |
| **Backbone** | DINOv2 ViT-B/14 (frozen) | same | same |
| **LoRA rank / alpha** | 8 / 16 | same | same |
| **LoRA dropout** | 0.0 | **0.1** | 0.0 |
| **Head dropout** | 0.1 | **0.2** | 0.1 |
| **LR (peak)** | 3e-4 | **1.5e-4** | 3e-4 |
| **Warmup** | 3 ep | same | same |
| **Loss λ_rel** | 0.2 | **0.4** | 0.2 |
| **Sampler** | random | random | **WeightedRandomSampler** (50/50 pos-neg + 11-rel balance) |
| **Patience** | 15 (effective 10) | 10 | 15 |
| **Epochs trained** | 22/40 | 14/40 | 10/40 (manual halt) |
| **Best epoch** | 12 | 4 | 5 |
| **Trainable params** | 8.47M / 94.2M (8.99%) | same | same |
| **Time / epoch** | ~84 min | ~86 min | ~85 min |

### Validation peak

| | Run 001 | Run 002 | Run 003 |
|---|---|---|---|
| **Val AUC peak** | **0.9116** | 0.9048 | 0.9091 |
| **Val Acc** | 83.7% | 82.6% | 82.2% |
| **Threshold (val)** | 0.10 | 0.30 | **0.50** |

Threshold drift sistematico (0.10 → 0.30 → 0.50) — cada intervencao de regularizacao/balanceamento empurra a distribuicao de scores para o centro.

### Test metrics (validation-selected threshold)

| Metric | Run 001 | Run 002 | Run 003 |
|---|---|---|---|
| **Test ROC-AUC** | 0.806 | 0.799 | **0.809** |
| **Test Accuracy** | **72.6%** | 72.4% | 71.0% |
| **Test F1** | 0.713 | **0.718** | 0.653 |
| **Test Precision** | 64.5% | 70.3% | **76.8%** |
| **Test Recall** | **82.0%** | 73.3% | 56.7% |
| **Avg Precision** | **0.792** | 0.772 | 0.778 |
| **TAR @ FAR=0.1** | **0.463** | 0.437 | 0.459 |
| **TAR @ FAR=0.01** | **0.152** | 0.095 | 0.098 |
| **TAR @ FAR=0.001** | **0.044** | 0.017 | 0.011 |
| **Val→Test gap** | -0.105 | -0.106 | -0.100 |

**Tres runs, mesmo gap val→teste (~-0.10)**, forte evidencia de propriedade do split RFIW Track-I. **R001 permanece como o melhor checkpoint do projeto para uso de alta precisao** (TAR@FAR=0.01 = 0.152, melhor de todos os modelos).

### Per-relation test accuracy

| Relation | N | Run 001 | Run 002 | Run 003 |
|---|---|---:|---:|---:|
| sibs | 234 | 83.3% | **85.9%** | 63.2% |
| bb | 860 | 79.8% | **85.5%** | 62.6% |
| ss | 731 | 77.2% | **81.1%** | 61.4% |
| fs | 1,135 | 71.6% | **73.8%** | 62.6% |
| fd | 918 | 71.5% | **72.9%** | 62.3% |
| md | 1,038 | 67.3% | **70.3%** | 50.0% |
| ms | 1,036 | 69.4% | **69.8%** | 51.6% |
| gmgs | 121 | **52.1%** | **52.1%** | 43.0% |
| gfgd | 138 | **50.7%** | 49.3% | 39.1% |
| gmgd | 123 | 40.7% | **45.5%** | 35.8% |
| gfgs | 98 | **39.8%** | 38.8% | 28.6% |

R002 distribuiu confianca um pouco mais uniformemente; **R003 regrediu em todas as 11 classes**, incluindo as classes de avo que o sampler era para ajudar. Parte da queda do R003 e artefato do threshold mais alto (0.50 vs 0.10), mas o ranking efetivo (TAR@FAR=0.01) tambem caiu — confirma que **starvation de dados nao era o gargalo**.

---

## Train Data Distribution (FIW positives)

The grandparent relations are **systematically underrepresented**:

| Relation | Train pairs | % of pos | Test acc R001 |
|---|---:|---:|---:|
| md | 5,349 | 16.1% | 67.3% |
| fs | 5,299 | 16.0% | 71.6% |
| fd | 5,268 | 15.9% | 71.5% |
| ms | 5,109 | 15.4% | 69.4% |
| sibs | 3,399 | 10.2% | 83.3% |
| ss | 3,298 | 9.9% | 77.2% |
| bb | 3,030 | 9.1% | 79.8% |
| **gfgs** | **682** | **2.1%** | **39.8%** |
| **gmgs** | **623** | **1.9%** | **52.1%** |
| **gfgd** | **608** | **1.8%** | **50.7%** |
| **gmgd** | **542** | **1.6%** | **40.7%** |

Grandparent classes have **~7× fewer training pairs** than parent-child classes. The poor accuracy on gfgs/gmgs/gfgd/gmgd is not an architecture failure — it's data starvation. The model receives 6-9× fewer gradient updates teaching grandparent kinship than parent-child kinship.

---

## Issues Log

| # | Issue | Severity | Status | First seen | Notes |
|---|-------|----------|--------|------------|-------|
| 1 | Patience config not honored — trainer stopped at patience 10 although `--patience 15` was passed | Medium | ❌ Open | Run 001 | Investigate `ROCmTrainer` base class — likely a hard-coded threshold somewhere |
| 2 | Large val→test gap (-0.10 in R001, R002 e R003) | High | ❌ Confirmed structural | Run 001 | **Tres runs, mesmo gap.** Nao fechado por regularizacao (R002) nem por class balancing (R003). Forte evidencia de divergencia entre distribuicoes de familia val/teste no RFIW Track-I — propriedade do split, nao do modelo. Confirmacao definitiva exigiria multi-seed ou 5-fold CV |
| 3 | Grandparent classes ~40-52% accuracy | High | ❌ Open — re-diagnosticado | Run 001 | Original hypothesis "data starvation (1.6-2.1% of train)" rejected by R003 (stratified sampler com 4-6× mais updates piorou em vez de melhorar). Nova hipotese: **gargalo arquitetonico** — DINOv2 frozen + LoRA nao consegue separar features ja indistintas no espaco extraido, por mais updates que recebam. Test direto requer descongelar o backbone (R004 planejado) |
| 4 | Time per epoch ~84 min (~3-4× M03/M06) | Low | Accepted | Run 001 | DINOv2 patch14 + grad checkpointing + diffattn. Not a blocker |
| 5 | Threshold drift sistematico (R001=0.10, R002=0.30, R003=0.50) | Medium | ❌ Open | Run 002 | Cada intervencao de regularizacao/balanceamento empurra a distribuicao de scores para 0.5. Modelo se torna progressivamente menos confiante mas nao mais discriminativo. Sintoma, nao causa raiz |

---

## Comparison with Other Models (FIW)

| Model | Test AUC | Test Acc | TAR@FAR=0.01 | Trainable | Min per-rel |
|---|---:|---:|---:|---:|---:|
| Modelo 02 R031 | **0.850** | 74.4% | ~0.13 | ~86M | 88.4% |
| Modelo 03 R002 | **0.850** | 47.9%* | 0.130 | ~176M | — |
| Modelo 03 R006 | 0.848 | 50.5%* | 0.132 | ~176M | — |
| **Modelo 05 R001** | 0.806 | **72.6%** | **0.152** | **8.47M** | 39.8% |
| **Modelo 05 R002** | 0.799 | 72.4% | 0.095 | 8.47M | 38.8% |
| **Modelo 05 R003** | 0.809 | 71.0% | 0.098 | 8.47M | 28.6% |
| Modelo 06 R001 | 0.776 | 69.8% | 0.062 | 8.16M | 61.2% |
| Modelo 06 R002 | 0.731 | 66.2% | 0.042 | ~12M | 41.3% |

*threshold=0.5 default — accuracy not directly comparable.

M05 R001 has the **best TAR@FAR=0.01 of any model in the project** (0.152), making it the strongest choice for high-precision regimes. Test AUC sits at 0.806 — below the parametric models (0.850) but with **10-20× fewer trainable parameters**.

---

## Conclusion

Tres runs, tres hipoteses testadas, tres rejeitadas:

1. **R001 (defaults).** Estabeleceu o teto: Test AUC 0.806, Val AUC 0.9116 (pico do projeto), TAR@FAR=0.01 = 0.152 (tambem pico do projeto). gfgs/gmgd em ~40%, val→teste gap -0.105.

2. **R002 (regularizacao mais forte).** Testou "gap = overfitting". **Rejeitada** — gap unchanged em -0.106. Test AUC -0.007. TAR@FAR=0.01 caiu para 0.095. Distribuiu confianca um pouco mais entre classes intra-geracao mas nao tocou as classes de avo.

3. **R003 (stratified sampler).** Testou "avos = data starvation". **Rejeitada** — todas as classes regrediram, incluindo as 4 classes de avo (gfgs caiu de 39.8% para 28.6%). Gap unchanged em -0.100.

### Convergencia diagnostica

- **O gap val→teste de ~-0.10 e estrutural** (3 runs, 3 hipoteses, mesmo gap). Provavel propriedade do split de familia RFIW Track-I, nao do modelo. Confirmacao requer multi-seed ou 5-fold CV.
- **O gargalo de classes raras nao e starvation de dados**, ja que mais updates nao ajudaram. **Hipotese remanescente**: limite arquitetonico — DINOv2 frozen extrai features que nao separam essas classes natively, e nenhuma quantidade de gradiente sobre features fixas pode adicionar informacao que nao esta nas features.
- **R001 permanece como o ponto de Pareto do M05** — melhor TAR@FAR=0.01 do projeto, com 10× menos params treinaveis que M02/M03.

### Direcao para Run 004

A unica hipotese arquitetonica nao testada: **descongelamento parcial do DINOv2 com LR diferenciado.** Plano:

```bash
SKIP_INSTALL=1 EPOCHS=20 PATIENCE=10 NUM_WORKERS=4 \
  UNFREEZE_BACKBONE_BLOCKS=4 BACKBONE_LR_FACTOR=0.01 \
  bash models/05_dinov2_lora_diffattn/AMD/run_pipeline.sh
```

- Descongela os ultimos 4 blocos do DINOv2 (~28M params)
- LR backbone = LR head × 0.01 ≈ 3e-6 (bem dentro do regime "preserve pretrained features")
- Mantem todos os outros hyperparams de R001

**Hipotese:** Test AUC 0.83-0.86, validando que o teto era arquitetonico. Se nao subir, o limite e o split, e a investigacao move-se para validacao cruzada.

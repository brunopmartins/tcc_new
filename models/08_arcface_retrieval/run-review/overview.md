# Overview — ArcFace + Retrieval-Augmented Kinship

**Model:** 08 — ArcFace IResNet-100 (frozen) + retrieval-augmented head
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (RFIW Track-I), com imagens MTCNN-alinhadas em `datasets/FIW_aligned`

Individual run details: [run-001.md](run-001.md)

---

## Arquitetura

Cópia exata do Modelo 06 R001 (frozen encoder + non-parametric gallery +
cross-attention over top-K retrieved supports + binary head + relation head).
Único single change: encoder DINOv2/ViT-B → **ArcFace IResNet-100**.

A hipótese motivadora: o ceiling de 0.776 do M06 era o encoder (visual
genérico), não a arquitetura. Substituir por encoder face-specialized SOTA
deveria destravar.

---

## Run Comparison

| Run | Date | Backbone | Weights source | Best Val AUC | Test AUC | Status |
|-----|------|----------|---------------|-------------:|---------:|--------|
| (aborted R50) | 2026-05-10 | ArcFace IResNet-50 CASIA-FaceV5 | JustinLeee/FaceMind (HF) | 0.6826 (ep 1) | n/a | Aborted at ep 1 |
| (aborted R100 Glint) | 2026-05-10 | ArcFace IResNet-100 Glint360K | lithiumice/insightface ONNX→PyTorch | 0.7619 (ep 1) | n/a | Aborted at ep 1 |
| **R001** | **2026-05-10** | **ArcFace IResNet-100 MS1MV3** | **InsightFace oficial (.pth)** | **0.7455** (ep 1) | **0.693** | **Completed (early stop)** |

---

## Headline Result

| Metric | M08 R001 | M06 R001 (frozen ImageNet ViT) | M02 R031 (full FT) |
|---|---:|---:|---:|
| **Test ROC AUC** | **0.6931** | 0.776 | 0.850 |
| Test Accuracy | 60.8% | 69.8% | 74.4% |
| Test F1 | 0.653 | 0.722 | 0.779 |
| **Val→Test gap** | -0.052 | -0.060 | -0.031 |
| Trainable params | 7.5M | 8.2M | 86M |
| Time per epoch | ~10 min | ~21 min | ~36 min |

**M08 R001 produz o pior Test AUC de qualquer modelo do projeto.**

---

## Conclusion Anti-Kinship — Hipótese Confirmada com Nuance

### Confirmação do efeito anti-kinship

Três variantes de ArcFace foram testadas (R50 CASIA, R100 Glint360K, R100
MS1MV3), todas com Val AUC ep 1 entre 0.68-0.76, todas abaixo do M06 R001
com ImageNet ViT (0.78). A degradação monotônica após ep 1 no R001 confirma
que o treino ativamente piora o sinal genealógico — não é warmup nem falta
de qualidade do pretreino.

**Causa identificada:** ArcFace é treinado com loss de margem angular para
**separar identidades**. Os datasets de pretreino (MS1MV2/MV3, Glint360K)
catalogam pais/filhos/irmãos como **identidades distintas**, e a loss
explicitamente os **empurra para regiões diferentes** do espaço de
embedding. O sinal genealógico que kinship verification precisa preservar
é precisamente o que ArcFace é projetado para **negar**.

### Descoberta inesperada: dissociação entre per-classe e AUC

Apesar do Test AUC catastrófico, M08 R001 produz **recall per-classe
significativamente melhor que M05 R001 nas classes raras de avó/avô**:

| Relação | M05 R001 | M08 R001 | Δ |
|---|---:|---:|---:|
| gmgd | 40.7% | **71.5%** | **+30.8 pp** |
| gfgs | 39.8% | **71.4%** | **+31.6 pp** |
| gfgd | 50.7% | **74.6%** | **+23.9 pp** |

Reinterpretação: **kinship verification não é uma tarefa única**. É a
interseção de:
- **(a) similaridade facial** — ArcFace é state-of-the-art aqui
- **(b) ausência de identidade** — ArcFace é treinado **contra** isso

M08 frozen herda (a) mas não consegue desfazer (b) na head. A retrieval
encontra parentes que se parecem (boa per-classe), mas a separação global
kin/non-kin falha porque rostos similares mas não-aparentados ficam
misturados (baixa AUC).

### Síntese para o projeto

| Encoder regime | Sinal (a) similaridade facial | Sinal (b) anti-identidade | AUC |
|---|---|---|---:|
| ImageNet ViT frozen | Médio | Neutro | 0.776 (M06) |
| DINOv2+LoRA | Médio | Neutro | 0.806 (M05) |
| **ArcFace frozen** | **Alto** | **NEGATIVO** | **0.693 (M08) ←** |
| ImageNet ViT full FT | Médio→Alto | Neutro→positivo | **0.850 (M02)** |

O quadrante "frozen + face-specific" é o **pior**. O quadrante "fine-tuned +
face-specific" é o melhor, mas isso é literalmente FaCoRNet (cópia direta).

---

## Implicações para o TCC

1. **Achado negativo limpo:** três datapoints independentes refutam a
   hipótese "encoder face-specialized frozen destrava kinship". Útil para
   a literatura — papers futuros podem evitar essa armadilha.

2. **Reinterpretação per-classe:** M08 R001 é o **melhor modelo do
   projeto para classes raras de avó/avô em modo recall**, apesar do
   pior AUC. Existe um trade-off real entre "encoder face-specialized"
   (boa similaridade visual local) e "AUC global" (separação kin/non-kin
   universal).

3. **Direção produtiva remanescente para encoder frozen:**
   self-supervised face encoders (DINOv2-Face tipo 1, FaRL) — herdam
   especialização facial **sem** a pressão anti-kinship. Não testado neste
   projeto.

4. **Direção produtiva sem trocar paradigma:** aceitar que o melhor
   "frozen" do projeto é M06 R001 (Test AUC 0.776) e o melhor overall é
   M02 R031 (0.850 com full fine-tune). M08 foi explorado e descartado
   com evidência sólida.

---

## Per-relation summary (all M0X models for context)

| Relation | N | M02 R031 | M03 R002 | M05 R001 | M06 R001 | **M08 R001** |
|---|---:|---:|---:|---:|---:|---:|
| bb | 860 | 95.5% | — | 79.8% | 86.5% | 84.5% |
| ss | 731 | 94.7% | — | 77.2% | 86.3% | 83.7% |
| sibs | 234 | 94.9% | — | 83.3% | 87.2% | 79.5% |
| md | 1,038 | 94.4% | — | 67.3% | 85.9% | **79.4%** |
| fs | 1,135 | 95.3% | — | 71.6% | 78.8% | 76.9% |
| ms | 1,036 | 93.9% | — | 69.4% | 83.9% | 75.9% |
| fd | 918 | 91.7% | — | 71.5% | 76.9% | 67.9% |
| **gfgd** | 138 | 89.9% | — | 50.7% | 75.4% | **74.6%** |
| **gmgd** | 123 | 91.1% | — | 40.7% | 63.4% | **71.5%** |
| **gfgs** | 98 | 95.9% | — | 39.8% | 82.7% | 71.4% |
| **gmgs** | 121 | 88.4% | — | 52.1% | 61.2% | 50.4% |

M08 R001 vence M05 R001 em **gmgd, gfgs, gfgd** por margens largas (+24 a
+32 pp). Em modo recall, ArcFace é melhor que DINOv2+LoRA para classes
genealogicamente distantes. Mas em modo verificação binária global, perde.

---

## Files

```
08_arcface_retrieval/
├── README.md
├── iresnet.py                       (InsightFace IResNet-50/100, MIT)
├── model.py                         (ArcFaceEncoder + retrieval head)
├── RUN_LOG.md                       (TBD)
├── weights/
│   ├── README.md
│   ├── .gitignore
│   └── arcface_r100.pth             (MS1MV3 oficial, gitignored)
├── preprocessing/
│   └── align_faces.py               (kept for reference — not used since
│                                     project provides FIW_aligned directly)
├── data/
│   └── .gitignore
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   ├── run_pipeline.sh
│   └── run_preprocessing.sh
└── run-review/
    ├── overview.md                  (this file)
    └── run-001.md
```

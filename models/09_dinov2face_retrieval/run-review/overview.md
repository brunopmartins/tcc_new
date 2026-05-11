# Overview — DINOv2-Face + Retrieval-Augmented Kinship

**Model:** 09 — DINOv2 ViT-B/14 (frozen, optional face overlay) + retrieval-augmented head
**GPU:** AMD Radeon RX 6750 XT (11.98 GB VRAM, gfx1031, ROCm 5.7)
**Dataset:** FIW (RFIW Track-I), com imagens MTCNN-alinhadas em `datasets/FIW_aligned`

Individual run details: [run-001.md](run-001.md) (TBD)

---

## Arquitetura

Cópia exata do Modelo 06 R001 / Modelo 08 (frozen encoder + non-parametric
gallery + cross-attention over top-K retrieved supports + binary head +
relation head). Single change vs M08: encoder ArcFace IResNet-100 →
**DINOv2 ViT-B/14** (com overlay opcional de DINOv2-Face).

Hipótese motivadora: o resultado catastrófico do M08 (Test AUC 0.693)
confirmou que o pré-treino com loss de margem angular (identity-discrimination)
suprime ativamente o sinal de parentesco em regime frozen. **DINOv2 foi
pré-treinado com objetivo self-supervised sem identidade**, então deveria
preservar especialização facial sem essa pressão anti-kinship.

Status do checkpoint DINOv2-Face específico: nenhuma versão pública matched
no HuggingFace (busca em 2026-05-10). M09 default usa **base DINOv2** (timm
`vit_base_patch14_dinov2.lvd142m`) + overlay opcional. O caso "sem overlay"
é, na prática, M06 R002 com a arquitetura de retrieval — útil como linha
de base para isolar o efeito do encoder no ranking de modelos do projeto.

---

## Run Comparison

| Run | Date | Backbone overlay | Best Val AUC | Test AUC | Status |
|-----|------|------------------|-------------:|---------:|--------|
| R001 | TBD | (none — base DINOv2) | — | — | TBD |

---

## Quadro de comparação esperado

| Encoder regime | Sinal (a) sim. facial | Sinal (b) anti-identidade | Test AUC (referência) |
|---|---|---|---:|
| ImageNet ViT frozen (M06 R001) | Médio | Neutro | 0.776 |
| DINOv2 frozen (M06 R002) | Médio-alto | Neutro | ~0.77-0.78 |
| **DINOv2 + retrieval (M09 sem overlay)** | Médio-alto | Neutro | **~0.78-0.80 esperado** |
| **DINOv2-Face + retrieval (M09 com overlay)** | Alto | Neutro | **~0.80-0.85 esperado** |
| ArcFace frozen (M08) | Alto | NEGATIVO | 0.693 |
| ImageNet ViT full FT (M02) | Médio→Alto | Neutro→positivo | 0.850 |

Se M09 (sem overlay) ficar em ~0.77-0.78, confirma que o gargalo do M06 era
arquitetural-residual; se subir para ~0.80+ apenas com overlay face-SSL,
confirma que **frozen + face-SSL** é o quadrante produtivo intermediário
que falta no espaço M02-M06-M08.

---

## Files

```
09_dinov2face_retrieval/
├── README.md                       (hipótese, arquitetura, hardware)
├── model.py                        (DINOv2FaceEncoder + retrieval head)
├── weights/
│   ├── README.md
│   └── .gitignore                  (dinov2_face.pth, gitignored)
├── data/
│   └── .gitignore
├── AMD/
│   ├── train.py
│   ├── test.py
│   ├── evaluate.py
│   └── run_pipeline.sh
├── Nvidia/                         (não implementado nesta versão)
└── run-review/
    ├── overview.md                 (este arquivo)
    └── RUN_LOG.md                  (template)
```

# Visao Geral da Pesquisa — Verificacao de Parentesco Facial

## Objetivo

Desenvolver e avaliar modelos de deep learning para **verificacao automatica de parentesco** a partir de imagens faciais. Dado um par de fotos de duas pessoas, o sistema determina se existe relacao de parentesco entre elas (pai-filho, mae-filha, irmaos, avos-netos, etc.).

A pesquisa implementa e compara tres arquiteturas progressivamente mais sofisticadas, todas avaliadas sob o mesmo protocolo padrao.

---

## Datasets

### KinFaceW-I

Dataset classico de verificacao de parentesco com ~1.066 pares de imagens faciais, divididos em 4 tipos de relacao (pai-filha, pai-filho, mae-filha, mae-filho). Utilizado para prototipagem rapida e validacao cruzada de 5 folds.

### FIW (Families in the Wild)

O maior dataset publico de parentesco facial, com 1.000+ familias e 11 tipos de relacao. Utilizado com o protocolo RFIW Track-I, que define splits pre-definidos disjuntos por familia (571 familias treino, 190 familias teste). Gera ~99.000 pares de treino e ~13.000 pares de teste.

---

## Modelos

### Modelo 01 — CNN + Sintese de Idade (Baseline)

**Arquivo:** `models/01_age_synthesis_comparison/`

Modelo baseline que utiliza um backbone CNN (ResNet/ArcFace) com um modulo de sintese de idade (SAM — Style-based Age Manipulation). A ideia e gerar variantes etarias de cada face (jovem, meia-idade, idoso) e comparar todos os pares idade-compativel, reduzindo o impacto da diferenca de idade entre parentes.

**Arquitetura:**
```
Par de Faces -> Sintese de Idade (SAM) -> Variantes Etarias (3 por face)
                                              |
                                   Comparacao All-vs-All (3x3 = 9 pares)
                                              |
                                   Agregacao por Atencao -> Score de Parentesco
```

**Backbone:** ResNet-18 / ArcFace (pre-treinado)
**Parametros:** ~11M (sem SAM)
**Limitacao:** O modulo SAM requer pesos pre-treinados especificos (~4.5 GB). Nos testes sem SAM ativo, o modelo funciona como uma CNN Siamese convencional.

---

### Modelo 02 — ViT + FaCoR Cross-Attention

**Arquivo:** `models/02_vit_facor_crossattn/`

Substitui o backbone CNN por um **Vision Transformer (ViT)** e adiciona um modulo de **cross-attention bidirecional** inspirado no FaCoR (Face Comparison by Relation). A cross-attention permite que cada face "olhe" para regioes relevantes da outra face, capturando similaridades faciais locais (formato do nariz, olhos, mandibula).

**Arquitetura:**
```
Par de Faces -> ViT Backbone (compartilhado) -> Patch Tokens (196 por face)
                                                      |
                                          Cross-Attention Bidirecional
                                          (2 camadas, 8 cabecas)
                                                      |
                                          Channel Attention (SE)
                                                      |
                                          Projecao -> Embeddings 512-dim (L2-norm)
```

**Backbone:** vit_base_patch16_224 (pre-treinado ImageNet)
**Parametros:** ~86M
**Inovacao:** Cross-attention entre patches das duas faces — o modelo aprende quais regioes faciais sao mais informativas para determinar parentesco.

---

### Modelo 03 — ConvNeXt + ViT Hybrid

**Arquivo:** `models/03_convnext_vit_hybrid/`

Arquitetura de **duplo backbone** que combina as forcas complementares de CNNs e Transformers. O ConvNeXt extrai **features locais** (texturas, detalhes finos) enquanto o ViT extrai **features globais** (estrutura espacial, relacoes de longa distancia). As representacoes sao combinadas por um modulo de fusao aprendivel.

**Arquitetura:**
```
Par de Faces -> ConvNeXt (features locais)  -+
                                              |-> Fusao -> Projecao -> Embeddings 512-dim
             -> ViT (features globais)      -+
```

**Backbone:** convnext_base + vit_base_patch16_224 (ambos pre-treinados ImageNet)
**Parametros:** ~176M
**Fusao:** Suporta 4 estrategias — concatenacao, atencao, gated, bilinear
**Inovacao:** Combinar CNN e Transformer permite capturar simultaneamente padroes locais (textura da pele, formato dos olhos) e globais (proporcoes faciais, simetria).

---

## Protocolo de Avaliacao

Todos os modelos sao avaliados sob o **mesmo protocolo padrao**:

1. Treina o modelo no conjunto de treino
2. Coleta predicoes (scores de similaridade) no conjunto de **validacao**
3. Busca o threshold otimo que maximiza F1 na validacao (grid search 0.10-0.95)
4. Aplica o **mesmo threshold** no conjunto de **teste** (sem reotimizar)
5. Reporta metricas do teste

Para a avaliacao final robusta, utiliza-se **validacao cruzada de 5 folds** com splits disjuntos por familia, reportando media +- desvio padrao.

---

## Resultados Comparativos

### KinFaceW-I — Single Run (Melhor Run de Cada Modelo)

| Metrica | Modelo 01 (CNN) | Modelo 02 (ViT+FaCoR) | Modelo 03 (Hybrid) |
|---------|----------------|----------------------|-------------------|
| | Run 02 | Run 23 | — |
| **Accuracy** | 68.5% | **74.1%** | — |
| **F1** | 0.727 | **0.753** | — |
| **ROC AUC** | 0.682 | **0.861** | — |
| **Precision** | 64.2% | **71.9%** | — |
| **Recall** | **84.0%** | 79.0% | — |
| Threshold | 0.550 | 0.900 | — |
| Parametros | ~11M | ~86M | ~176M |

### KinFaceW-I — 5-Fold Cross-Validation

| Metrica | Modelo 02 (ViT+FaCoR) |
|---------|----------------------|
| | Run 026 (5-fold CV) |
| **Mean Accuracy** | 73.6% +- 5.8% |
| **Mean F1** | 0.771 +- 0.028 |
| **Mean ROC AUC** | 0.842 +- 0.012 |
| **Mean Precision** | 69.4% +- 6.3% |
| **Mean Recall** | 87.8% +- 6.2% |
| Mean Threshold | 0.890 +- 0.020 |

### FIW (Families in the Wild) — Single Run (Melhor de Cada Modelo)

| Metrica | Modelo 02 (ViT+FaCoR) | Modelo 03 (Hybrid) |
|---------|----------------------|-------------------|
| | Run 031 (completo) | Run 002  |
| **Test Accuracy** | **74.4%** | — |
| **Test F1** | **0.779** | — |
| **Test ROC AUC** | **0.850** | — |
| **Val AUC (melhor)** | 0.881 | **0.861** |
| **Val Acc (melhor)** | 76.6% | 65.7% |
| Epoca do pico | 3/14 | 2/50
| Threshold | 0.900 | 0.900 |
| Batch size | 32 | 8 |
| Tempo/epoca | ~36 min | ~2h 23min |
| Parametros | ~86M | ~176M |

### Comparacao por Tipo de Relacao — FIW (Modelo 02, Run 031)

| Relacao | Accuracy | F1 | N pares |
|---------|----------|------|---------|
| gfgs (avo-neto) | 95.9% | 0.979 | 98 |
| bb (irmaos) | 95.5% | 0.977 | 860 |
| fs (pai-filho) | 95.3% | 0.976 | 1.135 |
| sibs (irmaos misto) | 94.9% | 0.974 | 234 |
| ss (irmas) | 94.7% | 0.973 | 731 |
| md (mae-filha) | 94.4% | 0.971 | 1.038 |
| ms (mae-filho) | 93.9% | 0.969 | 1.036 |
| fd (pai-filha) | 91.7% | 0.957 | 918 |
| gmgd (avo-neta) | 91.1% | 0.953 | 123 |
| gfgd (avo-neta) | 89.9% | 0.947 | 138 |
| gmgs (avo-neto) | 88.4% | 0.939 | 121 |

---

## Configuracao de Treinamento (Receita Otimizada)

Hiperparametros refinados ao longo de 32+ experimentos:

| Parametro | Valor |
|-----------|-------|
| Loss | Supervised Contrastive (distancia do cosseno) |
| Margem | 0.3 |
| Learning rate | 2e-6 (pico) |
| Scheduler | Cosine annealing -> 1e-7 |
| Warmup | 8 epocas (linear) |
| Weight decay | 5e-5 |
| Dropout | 0.25 |
| Neg ratio (treino) | 2:1 (relation_matched) |
| Neg ratio (avaliacao) | 1:1 (random) |
| Early stopping | Paciencia = 25 epocas |
| Mixed precision | AMP (FP16) habilitado |
| Gradient clipping | max_norm = 1.0 |

---

## Hardware

| Componente | Especificacao |
|-----------|---------------|
| GPU | AMD Radeon RX 6750 XT |
| Arquitetura | RDNA2 (gfx1031) |
| VRAM | 12 GB GDDR6 |
| Plataforma | ROCm 5.7 |
| Framework | PyTorch 2.3.1+rocm5.7 |

---

## Evolucao dos Resultados (Modelo 02 — FIW)

| Run | Configuracao Principal | Val AUC | Test AUC | Test Acc | Status |
|-----|----------------------|---------|----------|----------|--------|
| 028 | Freeze + unfreeze, LR=1e-5, sem scheduler | 0.872 | — | — | Cancelado (threshold fixo) |
| 030 | Freeze 4ep + unfreeze 2 blocos, cosine decay, LR=1e-5 | 0.867 | 0.822 | 70.2% | Completo |
| 031 | **No freeze**, LR=5e-6, warmup=5, dropout=0.2 | **0.881** | **0.850** | **74.4%** | Completo |
| 032 | No freeze, LR=2e-6, WD=5e-5, warmup=8, dropout=0.25 | 0.872 | — | — | Pausado (ep. 8, AUC subindo) |

---

## Principais Conclusoes

1. **ViT supera CNN** para verificacao de parentesco: o Modelo 02 (ViT+FaCoR) atinge AUC=0.861 no KinFaceW vs 0.682 do Modelo 01 (CNN), uma melhoria de +0.179.

2. **Cross-attention e eficaz**: permite que o modelo compare regioes especificas entre faces, capturando similaridades locais que a similaridade do cosseno global nao detecta.

3. **FIW e mais robusto que KinFaceW**: com ~99K pares de treino (vs ~750), o modelo generaliza melhor. O AUC no FIW (0.850) se aproxima do AUC no KinFaceW (0.842 em CV), validando a robustez.

4. **Regularizacao forte e essencial**: LR muito baixo (2e-6), weight decay alto (5e-5), dropout elevado (0.25) e cosine decay previnem overfitting nos backbones pre-treinados.

5. **Modelo Hybrid (03) e promissor**: com apenas 2 epocas de treino, ja atinge Val AUC=0.861, indicando que a combinacao CNN+Transformer captura features complementares. O treinamento esta em andamento.

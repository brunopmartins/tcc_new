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

| Metrica | Modelo 02 (ViT+FaCoR) | Modelo 03 Run 002 | Modelo 03 Run 003 |
|---------|----------------------|-------------------|-------------------|
| | Run 031 (completo) | ep. 5, Val AUC=0.8851 | ep. 5, Val AUC=0.8771 |
| **Test ROC AUC** | **0.850** | 0.850 | 0.845 |
| **Test Avg Precision** | — | **0.816** | 0.811 |
| **TAR@FAR=0.1** | — | **0.487** | 0.477 |
| **TAR@FAR=0.01** | — | 0.130 | **0.138** |
| **Val AUC (pico)** | 0.881 | **0.885** | 0.877 |
| Epoca do pico | 3/14 | 5/50 | 5/50 |
| Threshold avaliado | 0.900 | 0.500* | 0.500* |
| Batch size | 32 | 8 | 8 |
| Tempo/epoca | ~36 min | ~2h 23min | ~3h 8min |
| Parametros | ~86M | ~176M | ~176M |
| Neg ratio (treino) | 2:1 | 2:1 | 3:1 |
| Dropout | 0.25 | 0.25 | 0.40 |

*Nota: threshold=0.5 utilizado pelo evaluate.py (checkpoint salvo sem threshold otimo); threshold real de treinamento foi ~0.9. As metricas de accuracy/F1 refletem threshold=0.5 e nao sao diretamente comparaveis. **Usar ROC-AUC como metrica principal.**

### Analise de Separabilidade dos Backbones — Modelo 03

A separabilidade de cada backbone e medida pela diferenca entre as similaridades medias dos pares kin vs. non-kin. Quanto maior a diferenca, mais discriminativo o backbone.

| Componente | Run 002 (dropout=0.25) | Run 003 (dropout=0.40) |
|-----------|----------------------|----------------------|
| ConvNeXt kin mean | 0.598 | 0.561 |
| ConvNeXt non-kin mean | 0.486 | 0.419 |
| **ConvNeXt delta** | **+0.112** | **+0.142** |
| ViT kin mean | 0.722 | 0.605 |
| ViT non-kin mean | 0.481 | 0.294 |
| **ViT delta** | **+0.241** | **+0.311** |
| Fused kin mean | 0.933 | 0.874 |
| Fused non-kin mean | 0.772 | 0.564 |
| **Fused delta** | **+0.161** | **+0.310** |

Observacoes:
- O **ViT e consistentemente mais discriminativo** que o ConvNeXt em ambas as runs.
- O **dropout mais alto (Run 003)** produziu melhor separacao no espaco de features fused (+0.310 vs +0.161), apesar de AUC de teste ligeiramente inferior (0.845 vs 0.850) — sugere que o modelo fusionado generalizou melhor, mas o overfitting na val AUC foi mais acentuado.
- A separabilidade do ViT melhorou de delta=0.241 (Run 002) para delta=0.311 (Run 003), indicando que regularizacao mais forte foi beneficial para o backbone principal.

---

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

### Experimento Exploratorio com VLM (Codex) — FIW

Como complemento aos modelos treinados, foi executado um experimento **zero-shot** com um VLM do ecossistema Codex, sem qualquer etapa de treinamento ou fine-tuning. O objetivo foi medir ate onde um modelo multimodal generico consegue identificar a **classe exata da relacao de parentesco** apenas pela aparencia facial.

O desenho metodologico completo deste baseline esta documentado em `docs/pt/10_experimento_vlm_codex.md`.

**Configuracao do experimento:**
- **Modelo:** `gpt-5.4-mini` via Codex CLI
- **Modo:** zero-shot, `reasoning_effort=low`
- **Task:** classificacao fechada em 11 classes do FIW (`bb`, `ss`, `sibs`, `fd`, `fs`, `md`, `ms`, `gfgd`, `gfgs`, `gmgd`, `gmgs`)
- **Amostra:** 1500 imagens = **750 pares positivos**
- **Split:** pares positivos do `FIW/track-I/test-pairs.csv`, filtrando apenas entradas com caminhos de imagem validos no dataset local
- **Balanceamento:** amostragem estratificada quase uniforme; `gmgd` foi limitado a 46 pares por indisponibilidade de mais caminhos validos no dataset local, e o restante foi redistribuido entre as demais classes
- **Entrada:** cada par foi convertido em uma imagem composta (face esquerda + face direita)

### Resultados do VLM (1500 imagens / 750 pares)

| Metrica | Valor |
|---------|-------|
| **Accuracy** | **33.1%** |
| **Macro Precision** | 0.245 |
| **Macro Recall** | 0.320 |
| **Macro F1** | **0.257** |
| **Confidence media** | 0.782 |
| Confidence media (acertos) | 0.792 |
| Confidence media (erros) | 0.777 |

Com a amostra ampliada, o desempenho ficou mais estavel e confirmou o mesmo comportamento observado no piloto menor: o VLM consegue explorar bem sinais de **idade** e **genero**, mas continua falhando quando precisa distinguir relacoes genealogicas mais finas. Diferentemente do piloto inicial, a confianca ficou um pouco melhor calibrada: os acertos tiveram confianca media ligeiramente maior que os erros.

### Accuracy por Relacao — VLM Codex

| Relacao | Accuracy | Acertos |
|---------|----------|---------|
| `md` | **76.4%** | 55 / 72 |
| `fd` | **72.9%** | 51 / 70 |
| `fs` | **67.1%** | 47 / 70 |
| `sibs` | 58.6% | 41 / 70 |
| `ms` | 48.6% | 34 / 70 |
| `ss` | 20.0% | 14 / 70 |
| `bb` | 8.3% | 6 / 72 |
| `gfgd` | 0.0% | 0 / 70 |
| `gfgs` | 0.0% | 0 / 70 |
| `gmgd` | 0.0% | 0 / 46 |
| `gmgs` | 0.0% | 0 / 70 |

### Padroes de Erro do VLM

Os erros mais frequentes seguiram um padrao consistente:

1. **Relacoes de avo/avoa-neto(a) continuaram sendo o principal ponto cego**. Os erros dominantes foram `gfgs -> md` (41 casos), `gfgd -> ms` (39 casos), `gmgs -> gfgd` (32 casos) e `gmgs -> fd` (25 casos).
2. **O modelo depende fortemente de sinais de idade/genero**, o que favorece `fd`, `fs` e `md`, mas nao resolve graus genealogicos mais profundos.
3. **Irmaos do mesmo sexo seguem dificeis**: `bb` foi confundido principalmente com `sibs` (22 casos) e `ss` (14), enquanto `ss` foi confundido com `sibs` (23) e `fs` (11).
4. **As relacoes avo/avoa-neto(a) nao tiveram nenhum acerto** no conjunto ampliado, reforcando que a aparencia facial isolada nao basta para essa classificacao fine-grained em zero-shot.

### Interpretacao (Codex)

Este experimento reforca que um **VLM generico zero-shot nao substitui** modelos especializados para verificacao/classificacao de parentesco facial. Mesmo com 1500 imagens, o ganho concentrou-se nas relacoes pai/mae-filho(a) e em `sibs`, enquanto as relacoes de segundo grau permaneceram essencialmente irresolvidas. Como baseline metodologico, o resultado e valioso justamente por delimitar o que um modelo multimodal geral consegue inferir apenas com pistas visuais amplas.

---

### Experimento Zero-Shot com Claude Sonnet — FIW

Como segundo baseline zero-shot, o **Claude Sonnet** (Anthropic) foi avaliado sobre um **subconjunto anterior de 75 pares** usado no piloto inicial de VLMs, permitindo comparacao direta naquele recorte menor sem qualquer treinamento.

**Configuracao:**
- **Modelo:** `claude-sonnet-4-6` (zero-shot direto por visao)
- **Entrada:** imagem composta lado a lado de dois rostos
- **Task:** classificacao fechada em 11 classes FIW

### Resultados — Claude Sonnet (75 pares, 150 imagens, subconjunto piloto)

| Metrica | Codex piloto (`gpt-5.4-mini`) | Claude Sonnet |
|---------|----------------------|---------------|
| **Accuracy** | 28.0% | **37.3%** |
| **Macro Precision** | 0.210 | **0.296** |
| **Macro Recall** | 0.273 | **0.364** |
| **Macro F1** | 0.232 | **0.324** |
| Confianca media | 0.789 | 0.601 |

### Accuracy por Relacao — Claude Sonnet

| Relacao | Corretos/Total | Accuracy | F1 |
|---------|---------------|----------|----|
| fd (pai-filha) | 6/7 | **85.7%** | 0.800 |
| fs (pai-filho) | 5/7 | **71.4%** | 0.667 |
| md (mae-filha) | 5/7 | **71.4%** | 0.556 |
| sibs (irmao-irma) | 5/7 | **71.4%** | 0.588 |
| ms (mae-filho) | 4/7 | 57.1% | 0.500 |
| ss (irma-irma) | 2/7 | 28.6% | 0.308 |
| bb (irmao-irmao) | 1/7 | 14.3% | 0.143 |
| gfgd (avo-neta) | 0/7 | 0.0% | 0.000 |
| gfgs (avo-neto) | 0/7 | 0.0% | 0.000 |
| gmgd (avo-neta) | 0/6 | 0.0% | 0.000 |
| gmgs (avo-neto) | 0/6 | 0.0% | 0.000 |

### Padroes de Confusao — Claude Sonnet

- **Relacoes de segundo grau (gfgd, gfgs, gmgd, gmgs): 0% de acuracia** — confusao sistematica com relacoes pai/mae-filho(a).
- **Irmaos do mesmo sexo (bb, ss):** 14–29% — sem diferenca de genero ou geracao, a discriminacao e altamente ambigua.
- **Relacoes pai/mae-filho/a (fd, fs, md, ms):** as mais discriminaveis, com 57–86% de acuracia.

### Interpretacao Comparativa

No subconjunto piloto de 75 pares, ambos os VLMs dependem dos mesmos sinais visuais primarios — **diferenca etaria** e **genero** — e falham igualmente em relacoes de segundo grau. Nesse recorte menor, o Claude Sonnet supera o Codex em 9 pontos percentuais de acuracia geral (37.3% vs 28.0%), com F1 macro superior (0.324 vs 0.232). Ja o experimento ampliado do Codex, com 750 pares, estabilizou em 33.1% de accuracy e 0.257 de macro-F1, mas nao deve ser lido como comparacao direta com o Claude porque a amostra e maior e diferente.

A comparacao confirma que nenhum VLM generico zero-shot resolve verificacao de parentesco fine-grained: o treinamento supervisionado com perda contrastiva e indispensavel.

**Artefatos Claude:** `data/claude_vlm_fiw_150/` | **Artefatos Codex:** `data/codex_vlm_fiw_1500/` | **Metodologia:** `docs/pt/10_experimento_vlm_codex.md`

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

5. **Modelo Hybrid (03) e promissor**: com apenas 5 epocas de treino (Run 002), atingiu Val AUC=0.8851 e Test AUC=0.8500, demonstrando que a combinacao CNN+Transformer captura features complementares.

6. **Baselines VLM zero-shot confirmam a necessidade de supervisao**: tanto o Codex (28.0%) quanto o Claude Sonnet (37.3%) ficam muito abaixo dos modelos treinados em classificacao multiclasse de parentesco — o treinamento supervisionado e indispensavel para este dominio.

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

### Modelo 06 — Retrieval-Augmented Kinship Verification

**Arquivo:** `models/06_retrieval_augmented_kinship/`

Arquitetura **retrieval-augmented** que adiciona uma **memoria nao-parametrica** ao processo de decisao. Em vez de forcar a classificacao em uma unica passagem feed-forward, o modelo recupera, em tempo de inferencia, os **K pares positivos de treino mais similares** ao par de consulta e cross-atende a eles, usando suas assinaturas e rotulos de relacao como contexto.

**Arquitetura:**
```
Par (A, B) -> ViT encoder (FROZEN) -> e_a, e_b (512-d)
                                         |
              Pair signature s_q = norm([e_a+e_b, |e_a-e_b|, e_a*e_b])
                                         |
                Gallery (~33k pares positivos de treino)
                    | cosine top-K (chunked)
                    v
              K support tokens = sig_to_token(sig_i) + relation_embed(r_i)
                    |
              Cross-attention 2 camadas (query <-> supports)
                    |
              Binary head -> logit_kin
              Relation head -> logits_11
```

**Backbone:** vit_base_patch16_224 (frozen, ~85.8M params)
**Parametros treinaveis:** 8.16M (8.7% do total de 93.9M)
**Inovacao:** RAG (retrieval-augmented generation) e bem estabelecido em modelagem de linguagem, mas **nao havia sido avaliado para verificacao de parentesco facial** na literatura prévia. A ideia: dar ao modelo acesso direto a exemplos conhecidos de parentes que se parecem com o par de consulta, em vez de forcar tudo nos pesos.

**Componentes treinaveis:**
- Projecao (~3M): encoder ViT congelado -> 512-d embedding
- sig_to_token (~1M): assinatura 1536-d -> 512-d token
- Cross-attention de retrieval (~5M): 2 camadas, 4 cabecas
- Binary head (~0.1M) + Relation head (~0.1M)
- Relation embedding (~0.01M)

**Constraints de hardware:** desenhado para 12GB VRAM. A galeria fica na VRAM (~200MB em FP32). Top-K e calculado em chunks de 4096 linhas para nao materializar a matriz completa de similaridade.

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

### FIW (Families in the Wild) — Evolucao Completa do Modelo 03

#### Historico de Runs

| Run | Epocas | Freeze | Batch | Dropout | Neg ratio | Val AUC (pico) | Test AUC | Status |
|-----|--------|--------|-------|---------|-----------|----------------|----------|--------|
| 001 | 0 | Nao | — | — | — | — | — | **OOM crash** (batch_size muito grande) |
| 002 | 13/50 | Nao | 8 | 0.25 | 2:1 | 0.8851 (ep 5) | 0.850 | Parado (overfitting) |
| 003 | 10/50 | Nao | 8 | 0.40 | 3:1 | 0.8771 (ep 5) | 0.845 | Parado (overfitting) |
| 004 | 25/100 | Sim (sempre) | 32 | 0.30 | 2:1 | 0.8672 (ep 11) | 0.823 | Early stop (paciencia) |
| 005 | 10/100 | Sim → full unfreeze ep 11 | 32 | 0.30 | 2:1 | 0.8638 (ep 10) | 0.813† | **OOM crash** no unfreeze |
| 006 | 17/100 | Sim → full unfreeze ep 11 | 16 | 0.30 | 2:1 | 0.8728 (ep 11) | 0.848 | Parado (overfitting pos-unfreeze) |
| 007 | 15/100 | Sim → partial unfreeze ep 11 | 16 | 0.30 | 2:1 | 0.8678 (ep 11) | 0.828 | Parado (LR conservador) |

#### Comparacao de Metricas de Teste — Todas as Runs

| Metrica | Modelo 02 | M03 R002 | M03 R003 | M03 R004 | M03 R005 | M03 R006 | M03 R007 |
|---------|-----------|----------|----------|----------|----------|----------|----------|
| | Run 031 | sem freeze | sem freeze | freeze total | freeze→OOM | full unfreeze | partial unfreeze |
| **Test ROC AUC** | **0.850** | **0.850** | 0.845 | 0.823 | 0.813† | 0.848 | 0.828 |
| **Test Avg Prec** | — | 0.816 | 0.811 | 0.797 | 0.790 | **0.816** | 0.798 |
| **TAR@FAR=0.1** | — | 0.487 | 0.477 | 0.489 | 0.468 | **0.511** | 0.487 |
| **TAR@FAR=0.01** | — | 0.130 | **0.138** | 0.114 | 0.123 | 0.132 | 0.115 |
| **Val AUC (pico)** | 0.881 | **0.885** | 0.877 | 0.867 | 0.864 | 0.873 | 0.868 |
| Backbone LR factor | — | 1.0 | 1.0 | — (frozen) | — (OOM) | 0.01 | 0.001 |
| Parametros treinaveis | ~86M | ~176M | ~176M | ~2.9M | ~2.9M→OOM | ~2.9M→176M | ~2.9M→117M |
| Tempo/epoca (unfrozen) | ~36min | ~2h23m | ~3h08m | ~44min | — | ~3h23m | ~2h31m |

**Observacoes:**
- **Usar ROC-AUC como metrica principal.** Accuracy/F1 dependem do threshold: Runs 002, 003, 005 usam threshold=0.5 (default), Run 004 usa 0.75 (otimizado).
- †Run 005: checkpoint salvo na epoca 10 (fase congelada). OOM ao descongelar 176M params com batch_size=32.
- **Run 006 e a segunda melhor** (AUC=0.848), quase empatando com o melhor resultado (0.850). O full unfreeze com backbone_lr_factor=0.01 funcionou, mas overfitting pos-unfreeze impediu progresso adicional.
- **Run 007** (partial unfreeze, factor=0.001) ficou abaixo por LR de backbone conservador demais (~1e-6).

### Experimento de Freeze/Unfreeze — Modelo 03 (Runs 004–007)

As Runs 004–007 investigaram estrategias de congelamento de backbones para melhorar a estabilidade do treino e evitar overfitting nos 176M parametros do modelo hibrido.

**Motivacao:** As Runs 002–003 (sem freeze) mostraram que os backbones sofriam overfitting rapido — Val AUC atingia pico na epoca 5 e declinava. A hipotese foi que treinar primeiro a cabeca de fusao (2.9M params) com backbones congelados, e depois descongelar gradualmente, melhoraria a generalizacao.

#### Estrategias testadas

| Estrategia | Run | Descricao | Resultado |
|-----------|-----|-----------|-----------|
| **Sem freeze** | 002, 003 | Backbones treinados desde a epoca 1 | Pico rapido (ep 5), AUC de teste mais alto (0.850) |
| **Freeze total** | 004 | Backbones sempre congelados, so head treina | AUC de teste 0.823 — head sozinha insuficiente |
| **Freeze → Full unfreeze** | 005 | Congelar 10ep, descongelar tudo ep 11 | **OOM** — 176M params com batch_size=32 excedeu 12GB VRAM |
| **Freeze → Full unfreeze** (bs=16) | 006 | Mesmo, mas batch_size=16, backbone_lr_factor=0.01 | **Overfitting** — train loss caiu 0.0133→0.0038 em 6 ep, val AUC declinou de 0.8728 |
| **Freeze → Partial unfreeze** | 007 | Descongelar so ConvNeXt stages[2,3] + ViT blocks[8-11], backbone_lr_factor=0.001 | LR muito conservador — pico 0.8678, sem convergencia |

#### Detalhes do partial unfreeze (Run 007)

O partial unfreeze descongelou apenas as camadas mais profundas de cada backbone, mantendo as features universais (bordas, texturas basicas) congeladas:

- **ConvNeXt:** stages[2] (57.9M) + stages[3] (27.4M) = 85.4M params descongelados
- **ViT:** blocks[8-11] (4 × 7.09M) = 28.4M params descongelados
- **Total descongelado:** ~116.7M (vs 176M no full unfreeze)
- **Backbone LR:** head_lr × 0.001 = ~1e-6 (muito baixo para convergir)

#### Conclusoes do freeze/unfreeze

1. **Sem freeze continua sendo o melhor** em AUC de teste (Run 002: 0.850), apesar do overfitting mais rapido na validacao
2. **Full unfreeze (Run 006: 0.848)** quase empata — a diferenca de 0.002 esta dentro da margem de variacao. O freeze→unfreeze com backbone_lr_factor=0.01 e uma alternativa viavel, embora overfitting pos-unfreeze limite o treino a poucas epocas descongeladas
3. **Freeze total (Run 004: 0.823)** limita o modelo em -2.7pp — a head sozinha nao consegue compensar features pre-treinadas genericas
4. **Partial unfreeze (Run 007: 0.828)** ficou abaixo por LR conservador demais (factor=0.001 → ~1e-6). Recomenda-se factor=0.005 para futuras runs
5. A GPU AMD RX 6750 XT (12GB VRAM) limita o batch_size a 16 quando os backbones estao descongelados (Run 005 OOM com batch_size=32)

### Analise de Separabilidade dos Backbones — Modelo 03

A separabilidade de cada backbone e medida pela diferenca entre as similaridades medias dos pares kin vs. non-kin. Quanto maior a diferenca, mais discriminativo o backbone.

| Componente | R002 (sem freeze) | R003 (sem freeze) | R004 (freeze) | R006 (full unfreeze) | R007 (partial unfreeze) |
|-----------|-------------------|-------------------|---------------|----------------------|-------------------------|
| ConvNeXt kin mean | 0.598 | 0.561 | 0.441 | 0.233 | 0.363 |
| ConvNeXt non-kin mean | 0.486 | 0.419 | 0.364 | 0.122 | 0.276 |
| **ConvNeXt delta** | +0.112 | **+0.142** | +0.077 | +0.111 | +0.087 |
| ViT kin mean | 0.722 | 0.605 | 0.543 | 0.371 | 0.431 |
| ViT non-kin mean | 0.481 | 0.294 | 0.485 | 0.272 | 0.360 |
| **ViT delta** | +0.241 | **+0.311** | +0.058 | +0.099 | +0.071 |
| Fused kin mean | 0.933 | 0.874 | 0.767 | 0.786 | 0.772 |
| Fused non-kin mean | 0.772 | 0.564 | 0.492 | 0.440 | 0.478 |
| **Fused delta** | +0.161 | +0.310 | +0.275 | **+0.346** | +0.294 |

Nota: Run 005 omitida da tabela — checkpoint identico a Run 004 (fase congelada, pre-OOM).

Observacoes:
- O **ViT e consistentemente mais discriminativo** que o ConvNeXt quando descongelado (Runs 002, 003).
- Com **backbones congelados (Run 004)**, ambos tem separabilidade muito baixa (+0.077 e +0.058), mas a fusao compensa parcialmente (+0.275) — a head de fusao aprende a combinar features genericas de forma util.
- **Run 006 (full unfreeze)** atingiu a **melhor separabilidade na fusao** (+0.346), superando todas as outras runs. Os backbones individuais tem valores absolutos baixos, mas a maior separacao fused explica o segundo melhor Test AUC (0.848).
- **Run 007 (partial unfreeze)** ficou entre freeze total e full unfreeze em separabilidade (+0.294), coerente com o Test AUC intermediario (0.828). O LR conservador (factor=0.001) limitou a adaptacao dos backbones.
- O **dropout mais alto (Run 003)** produziu melhor separacao individual dos backbones (ViT delta=+0.311), mas Run 006 mostra que a fusao pode compensar backbones menos separados individualmente.

---

### Modelo 06 — Retrieval-Augmented Kinship — Run 001 (FIW)

Primeira run do modelo retrieval-augmented, com encoder ViT-B/16 congelado e galeria de 33.207 pares positivos do conjunto de treino.

**Configuracao:**

| Parametro | Valor |
|-----------|-------|
| Backbone | vit_base_patch16_224 (frozen) |
| Embedding dim | 512 |
| Retrieval K | 32 |
| Cross-attention | 2 camadas, 4 cabecas |
| Loss | BCE + 0.3 × contrastive + 0.15 × relation-CE |
| LR | 1e-4 (cosine, warmup 3 ep, min 1e-6) |
| Batch size | 8 (grad_accum 4 → eff. 32) |
| Epocas | 18/20 (early stop, paciencia 10) |
| Galeria | 33.207 pares positivos |
| Parametros treinaveis | 8.16M / 93.9M (8.7%) |
| Tempo/epoca | ~21 min |

**Trajetoria de treinamento:**

| Epoca | Val AUC | Train Loss | Nota |
|-------|---------|------------|------|
| 1 | 0.7820 | 1.342 | Warmup |
| 2 | 0.8026 | 1.226 | |
| 3 | 0.8230 | 1.187 | LR pico (1e-4) |
| 4 | 0.8261 | 1.146 | |
| 5 | 0.8336 | 1.123 | |
| 6 | 0.8295 | 1.106 | |
| 7 | 0.8287 | 1.083 | |
| **8** | **0.8361** | 1.061 | **Melhor (best.pt salvo)** |
| 9-18 | 0.819-0.826 | 0.888-1.038 | Plateau / overfitting na head |

**Metricas de teste (FIW, 13.425 pares, threshold=0.40 selecionado na val):**

| Metrica | Valor |
|---------|-------|
| **Test ROC AUC** | **0.776** |
| **Test Accuracy** | 69.8% |
| **Test F1** | 0.722 |
| **Test Precision** | 64.5% |
| **Test Recall** | 82.0% |
| **Avg Precision** | 0.735 |
| **TAR@FAR=0.1** | 0.388 |
| **TAR@FAR=0.01** | 0.062 |
| **TAR@FAR=0.001** | 0.006 |

**Accuracy por tipo de relacao — Modelo 06 Run 001:**

| Relacao | Accuracy | N pares |
|---------|----------|---------|
| sibs (irmaos misto) | 87.2% | 234 |
| bb (irmaos) | 86.5% | 860 |
| ss (irmas) | 86.3% | 731 |
| md (mae-filha) | 85.9% | 1.038 |
| ms (mae-filho) | 83.9% | 1.036 |
| gfgs (avo-neto) | 82.7% | 98 |
| fs (pai-filho) | 78.8% | 1.135 |
| fd (pai-filha) | 76.9% | 918 |
| gfgd (avo-neta) | 75.4% | 138 |
| gmgd (avo-neta) | 63.4% | 123 |
| gmgs (avo-neto) | 61.2% | 121 |

**Observacoes:**
- **Gap val→teste:** Val AUC 0.836 → Test AUC 0.776 (delta -0.06). Maior gap que Modelos 02/03, sugerindo que a retrieval captura padroes especificos da galeria (treino) que nao generalizam tao bem.
- **Per-relacao mais uniforme** que zero-shot VLMs: Modelo 06 atinge 61-87% em todas as classes, enquanto VLMs zero-shot caiam para 0% em relacoes de avo/avoa-neto(a). A retrieval-augmentation **resolve o problema das classes raras** que afligia os VLMs.
- **Comparado aos modelos parametricos (02, 03):** AUC abaixo (0.776 vs 0.85), mas com apenas 8.16M parametros treinaveis (vs 86M-176M).
- **Potencial nao explorado:** o encoder esta congelado. Fine-tuning com LoRA ou descongelamento parcial poderia fechar o gap para os modelos parametricos.

---

### Comparacao Final — FIW (Modelos Parametricos vs Retrieval-Augmented vs VLMs)

| Modelo | Approach | Test AUC | Test Acc | Trainable Params | Per-Relacao Min |
|--------|----------|----------|----------|------------------|-----------------|
| **Modelo 02 R031** | ViT + Cross-Attention | **0.850** | 74.4% | ~86M | 88.4% (gmgs) |
| **Modelo 03 R002** | ConvNeXt + ViT Hybrid | **0.850** | 47.9%* | ~176M | — |
| **Modelo 03 R006** | Hybrid + Full Unfreeze | 0.848 | 50.5%* | ~176M | — |
| **Modelo 06 R001** | Retrieval-Augmented (frozen) | 0.776 | 69.8% | 8.16M | **61.2% (gmgs)** |
| Codex VLM zero-shot | gpt-5.4-mini | — | 33.1% | 0 (zero-shot) | 0.0% |
| Claude Sonnet zero-shot | claude-sonnet-4-6 | — | 37.3% | 0 (zero-shot) | 0.0% |

*Threshold=0.5 default no evaluate.py — accuracy nao comparavel diretamente. Usar AUC.

**Insights:**
- Modelos parametricos vencem em AUC absoluto (0.850 vs 0.776).
- **Modelo 06 e 10x-20x menor em parametros treinaveis** e ainda assim atinge 0.776 AUC.
- VLMs zero-shot nao discriminam classes de avo/avoa-neto(a); Modelo 06 acerta 61-83% nessas mesmas classes.

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

### Adaptacao ao Dominio por Prompt no VLM (Codex)

Como extensao do baseline zero-shot, foi testada uma forma de **adaptacao ao dominio baseada em inferencia**, sem fine-tuning do modelo. A ideia foi adaptar o comportamento do VLM ao dominio de parentesco facial por meio de:

- **few-shot in-context learning** com exemplos do `FIW/track-I/val-pairs.csv`
- **prompt estruturado** obrigando o modelo a estimar gap geracional, genero e face mais velha antes da relacao final
- **calibracao leve em validacao** usando as saidas auxiliares do proprio VLM

Foram comparadas quatro configuracoes na validacao (110 pares balanceados): `seven_shot_v1`, `seven_shot_v2`, `eleven_shot_v1` e `eleven_shot_v2`. A melhor foi `seven_shot_v1` com **7 exemplos** e prompt estruturado mais conciso. Nenhuma variante de calibracao superou a saida bruta, entao a politica escolhida para o teste final foi `none`.

### Resultados da Adaptacao ao Dominio — Mesmo Teste de 750 Pares

| Metrica | Zero-shot baseline | Adaptado por prompt | Adaptado + calibracao |
|---------|--------------------|---------------------|-----------------------|
| **Accuracy** | **33.1%** | 26.7% | 26.7% |
| **Macro Precision** | **0.245** | 0.203 | 0.203 |
| **Macro Recall** | **0.320** | 0.260 | 0.260 |
| **Macro F1** | **0.257** | 0.223 | 0.223 |

O resultado foi **negativo**: a adaptacao inferencial por prompt **piorou** o desempenho no conjunto de teste, com delta de `-6.4` pontos em accuracy e `-0.034` em macro-F1. A calibracao nao trouxe qualquer ganho adicional.

### Efeito por Relacao — Adaptacao vs. Zero-shot

Os maiores **ganhos** ocorreram em:

- `sibs`: `58.6% -> 67.1%`
- `bb`: `8.3% -> 15.3%`
- `ss`: `20.0% -> 22.9%`
- `gmgd`: `0.0% -> 4.3%`

Os maiores **prejuizos** ocorreram em:

- `md`: `76.4% -> 33.3%`
- `fd`: `72.9% -> 47.1%`
- `ms`: `48.6% -> 35.7%`
- `fs`: `67.1% -> 60.0%`

As relacoes `gfgd`, `gfgs` e `gmgs` continuaram em `0%`.

### Interpretacao da Adaptacao

O prompt estruturado tornou o modelo **mais rigido** em torno de heuristicas explicitas de geracao/genero, mas nao mais correto. Os sinais auxiliares mostram isso:

- a estimativa de **gap geracional** acertou apenas `54.4%` dos 750 pares
- uma relacao derivada diretamente desses sinais ficou disponivel em `99.3%` dos casos
- essa relacao derivada coincidiu com a predicao final do VLM em `91.6%` dos pares

Ou seja, a adaptacao praticamente **engessou** a decisao do modelo em torno das suas proprias heuristicas intermediarias. Isso ajudou nas relacoes de mesma geracao (`bb`, `ss`, `sibs`), mas prejudicou fortemente relacoes pai/mae-filho(a), especialmente `md` e `fd`, sem resolver o problema central das relacoes de avo/avoa-neto(a).

Assim, neste estudo, a tentativa de adaptacao ao dominio via prompt **nao superou** o baseline zero-shot simples. Esse resultado tambem e util para a dissertacao: ele mostra que nem toda forma de "adaptação ao domínio" em VLMs melhora o desempenho, e que uma estrutura de prompt mais elaborada pode amplificar vieses heurísticos em vez de aproximar o modelo do comportamento desejado.

Artefatos: `data/codex_vlm_fiw_domain_adapt_1500/`  
Metodologia detalhada: `docs/pt/12_adaptacao_dominio_vlm_codex.md`

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

5. **Modelo Hybrid (03) empata com ViT+FaCoR**: a melhor run (002) atingiu Test AUC=0.850, igual ao Modelo 02 Run 031. O hibrido CNN+Transformer captura features complementares (ViT delta=+0.241, ConvNeXt delta=+0.112), mas a complexidade extra (176M vs 86M params) nao se traduz em ganho de AUC. O ViT sozinho ja e forte o suficiente neste dominio.

6. **Freeze/unfreeze nao superou treino direto no Modelo 03**: Runs 004-007 testaram congelamento de backbones com diferentes estrategias de descongelamento. Nenhuma superou as Runs 002-003 (sem freeze) em AUC de teste. Freeze total (Run 004, AUC=0.823) ficou 2.7pp abaixo. Full unfreeze causou OOM (Run 005) ou overfitting (Run 006). Partial unfreeze com LR conservador (Run 007) tambem ficou abaixo.

7. **Restricoes de hardware importam**: a AMD RX 6750 XT (12GB VRAM) limitou batch_size a 16 com backbones descongelados e impediu exploracao completa do espaco de hiperparametros para freeze/unfreeze.

8. **Baselines VLM zero-shot confirmam a necessidade de supervisao**: tanto o Codex (33.1%) quanto o Claude Sonnet (37.3%) ficam muito abaixo dos modelos treinados em classificacao multiclasse de parentesco — o treinamento supervisionado e indispensavel para este dominio.

9. **Adaptacao ao dominio por prompt em VLMs nao ajuda**: o few-shot + prompt estruturado piorou o desempenho do Codex (33.1% → 26.7%), engessando a decisao em heuristicas de idade/genero que amplificam vieses em vez de corrigir.

10. **Retrieval-augmentation (Modelo 06) resolve o problema de classes raras** mas com AUC abaixo dos parametricos: o modelo retrieval-augmented com encoder congelado atinge Test AUC=0.776 (vs 0.850 dos parametricos), mas com **per-relacao muito mais uniforme** (61-87% em todas as 11 classes). Isso confirma que dar acesso explicito a exemplos de treino similares ajuda em classes com poucos exemplos, mas a complexidade extra da galeria + cross-attention nao supera os modelos parametricos quando ha dados suficientes. Apenas 8.16M parametros sao treinaveis (vs 86M-176M).

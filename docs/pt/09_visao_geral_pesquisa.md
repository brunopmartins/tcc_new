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

### Modelo 05 — DINOv2 + LoRA + Differential Cross-Attention

**Arquivo:** `models/05_dinov2_lora_diffattn/`

Arquitetura que integra tres tecnicas que **nao haviam sido validadas para verificacao de parentesco facial** na literatura previa do projeto:

1. **DINOv2 ViT-B/14** (Oquab et al., 2024) como backbone — modelo self-supervised treinado em 142M imagens nao rotuladas. Produz features faciais de granularidade fina sem o vies de classes do ImageNet.
2. **LoRA adapters** (Hu et al., 2021) injetados nas projecoes qkv + proj do backbone. O backbone permanece **congelado**; apenas matrizes de baixo posto (~3-8M params) e as cabecas sao treinadas. Evita o overfitting que o Modelo 03 sofreu apos full unfreeze e reduz drasticamente o uso de VRAM.
3. **Differential cross-attention** (Ye et al., 2024 — arXiv:2410.05258). A cross-attention bidirecional do FaCoR e reformulada como uma **diferenca de dois mapas de atencao softmax**, amplificando correspondencias discriminativas entre regioes faciais e suprimindo as espurias.
4. **Cabeca auxiliar de classificacao de relacao**. Um token `[REL]` aprendivel alimenta um classificador 11-classes (FIW), treinado conjuntamente com a perda binaria de verificacao. O sinal multitarefa empurra o modelo a separar relacoes geneologicas finas (ex: `gmgs` vs `gfgd`) — exatamente onde os Modelos 02/03 e os baselines VLM falham.

**Arquitetura:**
```
Par (A, B) -> DINOv2 ViT-B/14 (FROZEN) + LoRA -> patch tokens (B, 256, 768)
                                                       |
                              Linear + LN -> (B, 256, 512); prepend [REL]
                                                       |
                              Differential bidirectional cross-attention
                                          (2 camadas, 8 cabecas)
                                                       |
                              Mean-pool -> emb1, emb2 (512-d L2-norm)
                                                       |
                                  ┌─────────┴─────────┐
                                  v                   v
                           Binary head          Relation head
                           [e1, e2, |e1-e2|,    [relA, relB] -> 11 classes
                            e1·e2] -> kin score
```

**Backbone:** vit_base_patch14_dinov2.lvd142m (frozen, ~86M params)
**Parametros treinaveis:** 8.47M (8.99% do total de 94.2M)
**Por que LoRA em vez de full/partial unfreeze:** as Runs 004-007 do Modelo 03 mostraram que full unfreeze com batch 32 causa OOM, full unfreeze com batch 16 causa overfitting pos-unfreeze, e partial unfreeze com LR factor baixo nao converge. LoRA contorna ambos os problemas — apenas ~3-8M params tem gradientes (cabe em 12 GB), os pesos pre-treinados do backbone permanecem intactos (sem catastrophic forgetting do espaco de features rico do DINOv2), e a restricao de baixo posto introduz regularizacao implicita.

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

### Modelo 05 — DINOv2 + LoRA + Differential Attention — Run 001 (FIW)

Primeira run do modelo com DINOv2 (frozen) + LoRA rank=8 + differential cross-attention + cabeca auxiliar de relacao. Defaults do `run_pipeline.sh`.

**Configuracao:**

| Parametro | Valor |
|-----------|-------|
| Backbone | vit_base_patch14_dinov2.lvd142m (frozen) |
| LoRA rank / alpha | 8 / 16 |
| Cross-attention | 2 camadas, 8 cabecas (differential) |
| Embedding dim | 512 |
| Loss | combined (BCE + 0.5 × contrastive + 0.2 × relation-CE) |
| LR | 3e-4 (cosine, warmup 3 ep, min 1e-6) |
| Batch size | 4 (grad_accum 8 → eff. 32) |
| Dropout | 0.1 |
| Epocas | 22/40 (early stop, paciencia 10 disparou) |
| Parametros treinaveis | 8.47M / 94.2M (8.99%) |
| Tempo/epoca | ~84 min |

**Trajetoria de treinamento (selecionada):**

| Epoca | Val AUC | Train Loss | Nota |
|-------|---------|------------|------|
| 1 | 0.9040 | 0.430 | Warmup |
| 3 | 0.9001 | 0.288 | LR pico (3e-4) |
| 5 | 0.9085 | 0.228 | |
| 10 | 0.9077 | 0.164 | |
| **12** | **0.9116** | 0.148 | **Melhor (best.pt salvo)** |
| 18 | 0.9034 | 0.097 | Plateau / overfitting |
| 22 | 0.8962 | 0.066 | Stop por paciencia |

**Metricas de teste (FIW, 13.425 pares, threshold=0.10 selecionado na val):**

| Metrica | Valor |
|---------|-------|
| **Test ROC AUC** | **0.806** |
| **Test Accuracy** | 72.6% |
| **Test F1** | 0.713 |
| **Test Precision** | 71.7% |
| **Test Recall** | 70.8% |
| **Avg Precision** | 0.792 |
| **TAR@FAR=0.1** | 0.463 |
| **TAR@FAR=0.01** | **0.152** |
| **TAR@FAR=0.001** | 0.044 |

**Accuracy por tipo de relacao — Modelo 05 Run 001:**

| Relacao | Accuracy | N pares |
|---------|----------|---------|
| sibs (irmaos misto) | 83.3% | 234 |
| bb (irmaos) | 79.8% | 860 |
| ss (irmas) | 77.2% | 731 |
| fs (pai-filho) | 71.6% | 1.135 |
| fd (pai-filha) | 71.5% | 918 |
| ms (mae-filho) | 69.4% | 1.036 |
| md (mae-filha) | 67.3% | 1.038 |
| gmgs (avo-neto) | 52.1% | 121 |
| gfgd (avo-neta) | 50.7% | 138 |
| gmgd (avo-neta) | 40.7% | 123 |
| gfgs (avo-neto) | 39.8% | 98 |

**Observacoes:**
- **Maior Val AUC do projeto:** 0.9116 (vs 0.885 do M02 R031 e 0.885 do M03 R002). DINOv2 + LoRA + DiffAttn captura discriminacao mais forte na validacao.
- **Maior gap val→teste do projeto:** 0.9116 → 0.806 = **−0.105**. M02 tipicamente tem -0.03; M06 R001 teve -0.06. Indica overfitting da validacao.
- **TAR@FAR=0.01 = 0.152 e o melhor entre todos os modelos** (M02/M03 ~0.13, M06 ~0.06). Em regimes de threshold rigoroso, M05 e o mais forte.
- **Mesma fraqueza em classes de avo/avoa** (40-52%), apesar do peso de relation-CE de 0.2. **gfgs e o pior (39.8%)** — oposto de M02 R031 onde gfgs era o melhor (95.9%). Modo de falha diferente.
- **Custo computacional alto:** 84 min/epoca, ~3-4× M03/M06. DINOv2 patch14 + grad checkpoint + diffattn somam.
- **Hipoteses para o gap val→teste:** (a) features ricas do DINOv2 + LoRA com pouco dropout favorecem overfitting da distribuicao familiar da validacao; (b) cabeca de relacao com lambda=0.2 ajudou val mas nao generalizou. Follow-ups: aumentar relation_loss_weight para 0.4-0.5, adicionar `LORA_DROPOUT=0.1`, reduzir LR pico para 1.5e-4.

---

### Modelo 05 — DINOv2 + LoRA + Differential Attention — Run 002 (FIW, ablation de regularizacao)

Run 002 testa a hipotese da Run 001: **se o gap val→teste de -0.105 era overfitting**, regularizacao mais forte deveria fechar o gap. Mudancas: LR pico 3e-4 → 1.5e-4, dropout das heads 0.1 → 0.2, lora_dropout 0.0 → 0.1, relation_loss_weight 0.2 → 0.4.

**Configuracao (deltas vs Run 001 em negrito):**

| Parametro | Run 001 | Run 002 |
|-----------|---------|---------|
| LR pico | 3e-4 | **1.5e-4** |
| Dropout (heads) | 0.1 | **0.2** |
| LoRA dropout | 0.0 | **0.1** |
| Relation loss weight | 0.2 | **0.4** |
| Patience | 15 | 10 |
| Outros | — | identicos |

**Trajetoria de treinamento (selecionada):**

| Epoca | Val AUC R001 | Val AUC R002 | Δ |
|-------|------------:|------------:|------:|
| 1 | 0.9040 | 0.8447 | -0.060 |
| 4 | 0.9058 | **0.9048** | -0.001 |
| 12 | **0.9116** (R001 peak) | 0.8969 | -0.015 |
| Stop | ep 22 | **ep 14** | — |

R002 atingiu seu pico (0.9048) na epoca 4, R001 atingiu 0.9116 na epoca 12. R002 estabilizou cedo — sem capacidade adicional para chegar ao ceiling de R001.

**Metricas de teste — comparacao R001 vs R002:**

| Metrica | Run 001 | Run 002 | Δ |
|---------|--------:|--------:|------:|
| **Test ROC AUC** | 0.806 | 0.799 | **-0.007** |
| Test Accuracy | 72.6% | 72.4% | -0.2 pp |
| Test F1 | 0.713 | 0.718 | +0.005 |
| Avg Precision | 0.792 | 0.772 | -0.020 |
| **TAR@FAR=0.1** | 0.463 | 0.437 | -0.026 |
| **TAR@FAR=0.01** | **0.152** | 0.095 | **-0.057** |
| **TAR@FAR=0.001** | 0.044 | 0.017 | -0.027 |
| **Val AUC peak** | 0.9116 | 0.9048 | -0.007 |
| **Val→teste gap** | -0.105 | -0.106 | ~0 |
| Threshold (val) | 0.10 | 0.30 | — |

**Per-relation comparison:**

| Relacao | R001 | R002 | Δ |
|---------|-----:|-----:|------:|
| sibs | 83.3% | **85.9%** | +2.6 |
| bb | 79.8% | **85.5%** | +5.7 |
| ss | 77.2% | **81.1%** | +3.9 |
| md | 67.3% | **70.3%** | +3.0 |
| fs | 71.6% | **73.8%** | +2.2 |
| fd | 71.5% | **72.9%** | +1.4 |
| ms | 69.4% | **69.8%** | +0.4 |
| gmgs | 52.1% | 52.1% | 0.0 |
| gmgd | 40.7% | **45.5%** | +4.9 |
| gfgd | 50.7% | 49.3% | -1.4 |
| gfgs | 39.8% | 38.8% | -1.0 |

**Observacoes — hipotese rejeitada:**
- **O val→teste gap nao fechou** (-0.106 vs -0.105). A regularizacao baixou val e teste em paralelo, sem mudar a magnitude da divergencia. Isso **descarta "overfitting da validacao"** como explicacao isolada para o gap.
- **TAR@FAR=0.01 caiu de 0.152 para 0.095** — R002 perde a unica metrica em que R001 era a melhor do projeto. O tradeoff: R002 distribui confianca mais uniformemente entre classes, ao custo de precisao em thresholds rigorosos.
- **Per-classe mais uniforme:** ganhos de +3-6 pp em bb/ss/sibs/md, +5 pp em gmgd. Mas as classes de avo/avoa ainda quase em random (38-52%). **Aumentar relation_loss_weight de 0.2 para 0.4 nao consertou as classes raras** — sugere que a limitacao nao e o sinal multitarefa, e sim **insuficiencia de exemplos** no treino dessas classes ou ambiguidade visual intrinseca.
- **Implicacao estrutural para o gap val→teste:** se nao e overfitting, e provavel que seja **divergencia entre as distribuicoes de familia val e teste** (o split RFIW Track-I tem familias disjuntas, e a val pode conter familias visualmente mais discriminaveis que o teste). Isso e um achado relevante: o gap pode ser propriedade do protocolo, nao do modelo.
- **Convergencia mais rapida** (pico na ep 4 vs ep 12 da R001) sugere que a capacidade nao e o gargalo — o modelo ja converge ao seu maximo cedo.

---

### Modelo 05 — DINOv2 + LoRA + Differential Attention — Run 003 (FIW, ablation de class-balanced sampling)

Run 003 testa a hipotese de que **starvation de dados nas classes de avo (1.6-2.1% do treino) explica a falha em gfgs/gmgd** observada em R001. Substitui o sampler aleatorio por `WeightedRandomSampler` que entrega 50% de positivos divididos igualmente entre as 11 relacoes do FIW + 50% de negativos. Em verificacao empirica, gfgs sai de 1.03% para 4.54% das amostras por epoch (4.4× mais updates).

Todos os outros hyperparams identicos a R001 — ablacao limpa do sampler.

**Configuracao (delta vs R001):**

| Parametro | R001 | R003 |
|-----------|------|------|
| Sampler | random shuffle | **WeightedRandomSampler** (50/50 pos-neg + balance entre 11 relacoes positivas) |

Outros parametros (LR, dropout, λ_rel, etc) inalterados. Treino interrompido manualmente na epoca 10 com best em epoca 5.

**Trajetoria de val (selecionada):**

| Epoca | Val AUC R003 | Val AUC R001 | Δ |
|-------|------------:|------------:|------:|
| 1 | 0.8798 | 0.9040 | -0.024 |
| 4 | 0.9035 | 0.9058 | -0.002 |
| **5** | **0.9091** | 0.9085 | **+0.001** (R003 paridade) |
| Best (R001 ep 12) | — | **0.9116** | — |
| Stop | ep 10 (manual) | ep 22 | — |

R003 atingiu paridade com R001 em pico de val mas nao progrediu alem da ep 5. Train loss consistentemente menor que R001 em cada epoch.

**Metricas de teste — comparacao R001 vs R002 vs R003:**

| Metrica | R001 | R002 | R003 |
|---------|------:|------:|------:|
| **Test ROC AUC** | 0.806 | 0.799 | **0.809** |
| Test Accuracy | **72.6%** | 72.4% | 71.0% |
| Test F1 | 0.713 | 0.718 | 0.653 |
| Test Precision | 64.5% | 70.3% | **76.8%** |
| Test Recall | **82.0%** | 73.3% | 56.7% |
| Avg Precision | **0.792** | 0.772 | 0.778 |
| **TAR@FAR=0.1** | **0.463** | 0.437 | 0.459 |
| **TAR@FAR=0.01** | **0.152** | 0.095 | 0.098 |
| TAR@FAR=0.001 | **0.044** | 0.017 | 0.011 |
| Threshold (val) | 0.10 | 0.30 | **0.50** |
| Val→teste gap | -0.105 | -0.106 | -0.100 |

**Per-relation comparison:**

| Relacao | N | R001 | R002 | R003 | Δ R003 vs R001 |
|---------|---:|----:|----:|----:|---:|
| sibs | 234 | 83.3% | **85.9%** | 63.2% | -20.1 pp |
| bb | 860 | 79.8% | **85.5%** | 62.6% | -17.2 pp |
| ss | 731 | 77.2% | **81.1%** | 61.4% | -15.8 pp |
| fs | 1,135 | 71.6% | **73.8%** | 62.6% | -9.0 pp |
| fd | 918 | 71.5% | **72.9%** | 62.3% | -9.2 pp |
| md | 1,038 | 67.3% | **70.3%** | 50.0% | -17.3 pp |
| ms | 1,036 | 69.4% | **69.8%** | 51.6% | -17.8 pp |
| **gmgs** | 121 | **52.1%** | **52.1%** | 43.0% | -9.1 pp |
| **gfgd** | 138 | **50.7%** | 49.3% | 39.1% | -11.6 pp |
| **gmgd** | 123 | 40.7% | **45.5%** | 35.8% | -4.9 pp |
| **gfgs** | 98 | **39.8%** | 38.8% | **28.6%** | -11.2 pp |

**Observacoes — hipotese rejeitada:**

- **Todas as 11 classes regrediram em R003**, incluindo as 4 classes de avo que o sampler era para ajudar. gfgs caiu para 28.6% (era 39.8% em R001).
- **Test AUC ficou estatisticamente igual** (0.809 vs 0.806). TAR@FAR=0.01 caiu para 0.098 (de 0.152 em R001).
- Parte da queda per-relation e artefato do threshold mais alto (0.50 vs 0.10). Mas TAR@FAR=0.01 e threshold-independent, e tambem caiu — entao o ranking efetivo na regiao de alta precisao piorou.
- **Convergencia das tres runs:** val→teste gap em torno de -0.10 nas tres (-0.105, -0.106, -0.100). Tres intervencoes diferentes, mesmo gap. Isso e a evidencia mais forte de que **o gap e propriedade do split RFIW Track-I**, nao do modelo.
- **Hipotese sobrevivente para classes de avo:** **gargalo arquitetonico** — DINOv2 frozen extrai features que nao distinguem essas classes, e nenhuma quantidade de updates sobre features fixas pode adicionar informacao nao-codificada. Teste direto: descongelar parcialmente o backbone (Run 004 planejado).

---

### Modelo 05 — Sumario das Runs 004-007 (FIW)

Quatro runs adicionais foram executadas testando direcoes arquitetonicas e
variantes de loss / plasticidade. **Nenhuma moveu o teto de Test AUC do M05
significativamente** alem do patamar 0.81-0.82 ja estabelecido em R001-R003.
Detalhes completos em [models/05_dinov2_lora_diffattn/RUN_LOG.md](../../models/05_dinov2_lora_diffattn/RUN_LOG.md)
e [run-review/](../../models/05_dinov2_lora_diffattn/run-review/).

| Run | Approach | Trainable | Test AUC | Val→Test gap | TAR@FAR=0.01 | Veredito |
|-----|----------|----------:|---------:|-------------:|-------------:|----------|
| R004 | Partial unfreeze (last 4 DINOv2 blocks) | 36.8M | 0.812 | -0.099 | 0.119 | Marginal +0.006 vs R001 |
| R005 | Full unfreeze + LR M02-style (5e-6) | 93.5M | **0.822** | **-0.071** | 0.100 | Best M05 AUC; menor gap |
| R006 | Heavy contrastive loss (peso 0.9 vs 0.5) | 8.5M | 0.814 | -0.10 | 0.094 | Marginal +0.008 vs R001 |
| R007 | DINOv2 + M02-trained-ViT hybrid backbone | 11.6M | 0.810 | -0.097 | 0.136 | H0 confirmado, hibrido nao bate M02 |

**Tres direcoes arquitetonicas testadas em R004-R007:**

1. **Plasticidade total (R005)** — descongelar todo o DINOv2 com LR M02-style (5e-6) **moveu o teto marginalmente** para 0.822 (de 0.806 do R001) e **reduziu o val→teste gap** para -0.071 (vs -0.10 das demais). Unica intervencao com efeito mensuravel sobre o gap, sugerindo que parte do gap e atribuivel a capacidade limitada do head com backbone frozen.

2. **Loss heavy-contrastive (R006)** — substituir o peso 0.5/0.5 entre BCE e contrastive por 0.9/0.1, com `relation_loss_weight=0`. Teste de "pode loss puramente contrastive (estilo M02) ajudar M05?". **Resultado:** ganho de +0.008 AUC vs R001, dentro de margem de ruido. Loss nao e o gargalo principal de M05.

3. **Backbone hibrido (R007)** — fundir DINOv2 (general visual) + M02-trained-ViT (kinship-tuned, extraido de M02 R031 epoch_5.pt) com cross-attention diferencial intra-face. **Hipotese**: combinar self-supervised + task-specific deveria gerar features complementares que nem M02 nem DINOv2 sozinhos atingem. **Resultado: H0 confirmado** — Test AUC = 0.810, identico a R001 sem o M02 backbone. **Per-relation regrediu em todas as 11 classes** vs R001. Conclusao: as features kinship-fortes do M02 R031 nao sao do ViT alone — vem do **sistema completo** (ViT + FaCoR cross-attention + supervised contrastive + threshold 0.90 calibrados juntos). Reusar so os pesos do ViT em outra arquitetura quebra o ecossistema.

**Bug encontrado durante R007:** `test.py` e `evaluate.py` instanciavam apenas `DINOv2LoRAKinship`, ignorando a arquitetura hybrid. Com `strict=False` em `load_state_dict`, os pesos `face_vit.*` e `intra_face_layers.*` eram silenciosamente descartados, produzindo Test AUC random (0.504) na primeira execucao. Corrigido para detectar arquitetura via prefixos do state_dict.

**Sintese pos-7-runs do M05:** o teto arquitetonico de 0.81-0.82 e robusto a 7 hipoteses diferentes. O val→teste gap de ~-0.10 e estrutural ao split RFIW Track-I (so quebra parcialmente com plasticidade total). Para superar M02 R031 (Test AUC 0.85), o caminho viavel seria (a) full fine-tune com receita CosFace/ArcFace + LoRA rank 16 + LR 1e-4 (replicacao de FRoundation), ou (b) trocar arquitetura completa para M07 com backbone face-specific pretrained + co-design de head/loss. **R001 permanece como o checkpoint Pareto-otimal do M05** para TAR@FAR=0.01 (0.152, melhor de todo o projeto).

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

### Modelo 06 — Retrieval-Augmented Kinship — Run 002 (Ablation: DINOv2 + K=64)

Segunda run, com tres mudancas em relacao a Run 001: troca do encoder para **DINOv2 ViT-B/14** (frozen), aumento de **K=32 → K=64** no retrieval, e elevacao do **relation_loss_weight de 0.15 para 0.30**. Hipotese: features self-supervised do DINOv2 + mais contexto de retrieval + sinal multitask mais forte deveriam empurrar AUC para cima.

**Configuracao (apenas o que mudou em relacao a Run 001):**

| Parametro | Run 001 | Run 002 |
|-----------|---------|---------|
| Backbone | vit_base_patch16_224 | **vit_base_patch14_dinov2.lvd142m** |
| Retrieval K | 32 | **64** |
| relation_loss_weight | 0.15 | **0.30** |
| Gallery on CPU | False | True (devido ao maior K) |
| Epocas (max / actual) | 20 / 18 | 60 / 29 |
| Patience | 10 | 20 |

**Trajetoria de treinamento (resumo):**

| Epoca | Val AUC | Train Loss |
|-------|---------|------------|
| 1 | 0.7921 | 1.488 |
| 2 | 0.8213 | 1.300 |
| 3 | 0.8076 | 1.204 |
| 5 | 0.8179 | 1.047 |
| **9** | **0.8228** | 0.914 (melhor) |
| 17 | 0.8212 | 0.827 |
| 20 | 0.8211 | 0.805 |
| 29 | 0.8051 | 0.739 (early stop) |

**Metricas de teste (FIW, 13.425 pares, threshold=0.55 selecionado na val):**

| Metrica | Run 002 | Run 001 | Delta |
|---------|--------:|--------:|------:|
| **Test ROC AUC** | **0.7310** | 0.7763 | **−0.045** |
| Test Accuracy | 66.18% | 69.78% | −3.6 pp |
| Balanced Accuracy | 65.82% | 70.27% | −4.5 pp |
| Test F1 | 0.6190 | 0.7223 | −0.103 |
| Test Precision | 67.23% | 64.52% | +2.7 pp |
| Test Recall | 57.35% | 82.04% | −24.7 pp |
| Avg Precision | 0.6808 | 0.7345 | −0.054 |
| TAR@FAR=0.1 | 0.2974 | 0.3881 | −0.091 |
| TAR@FAR=0.01 | 0.0423 | 0.0616 | −0.019 |

**Gap val → teste:** 0.8228 → 0.7310 = **−0.092** (Run 001: 0.836 → 0.776 = −0.060). O gap aumentou em 53%, sinal de overfitting da validacao.

**Accuracy por tipo de relacao — Modelo 06 Run 002:**

| Relacao | Run 002 | Run 001 | Delta |
|---------|--------:|--------:|------:|
| bb | 66.86% | 86.51% | −19.7 pp |
| ss | 64.57% | 86.32% | −21.8 pp |
| sibs | 62.39% | 87.18% | −24.8 pp |
| md | 60.69% | 85.93% | −25.2 pp |
| fs | 55.33% | 78.77% | −23.4 pp |
| ms | 54.92% | 83.88% | −29.0 pp |
| gfgs | 51.02% | 82.65% | −31.6 pp |
| gfgd | 50.72% | 75.36% | −24.6 pp |
| gmgd | 47.97% | 63.41% | −15.4 pp |
| fd | 47.93% | 76.91% | −29.0 pp |
| gmgs | 41.32% | 61.16% | −19.8 pp |

Todas as 11 classes pioraram. Boa parte da queda em recall vem do threshold mais agressivo (0.55 vs 0.40), mas o ROC AUC, threshold-independente, tambem caiu (−0.045) — o ranking generaliza pior.

**Conclusao da ablacao:**

A hipotese de que DINOv2 + K=64 + relation_weight 0.3 melhoraria o desempenho **nao se confirmou**. Tres lições:

1. **Backbone DINOv2 nao ajudou** com encoder frozen + retrieval. Features self-supervised mais ricas tornaram a head mais facil de sobre-ajustar a galeria + validacao.
2. **K maior amplificou o overfitting**, nao o resolveu. Mais contexto de retrieval = mais correlacao galeria↔val que nao transfere para test.
3. **O gargalo de M06 nao e qualidade visual nem volume de retrieval** — e a regularizacao do mecanismo de cross-attention contra a galeria fixa. Hard negatives nos supports (recuperados-mas-nao-parentes) sao a proxima hipotese a testar.

Este resultado negativo entra no TCC como ablacao: confirma que melhorias "obvias" (mais features, mais K) nao salvam um modelo cujo problema e regularizacao.

---

### Comparacao Final — FIW (Modelos Parametricos vs Retrieval-Augmented vs VLMs)

| Modelo | Approach | Test AUC | Test Acc | TAR@FAR=0.01 | Trainable Params | Per-Relacao Min |
|--------|----------|---------:|---------:|-------------:|-----------------:|----------------:|
| **Modelo 02 R031** | ViT + Cross-Attention | **0.850** | 74.4% | ~0.13 | ~86M | 88.4% (gmgs) |
| **Modelo 03 R002** | ConvNeXt + ViT Hybrid | **0.850** | 47.9%* | 0.130 | ~176M | — |
| **Modelo 03 R006** | Hybrid + Full Unfreeze | 0.848 | 50.5%* | 0.132 | ~176M | — |
| **Modelo 05 R001** | DINOv2 + LoRA + DiffAttn + relation head | 0.806 | **72.6%** | **0.152** | **8.47M** | 39.8% (gfgs) |
| **Modelo 05 R002** | M05 + regularizacao mais forte (LR/2, dropout↑, λ_rel↑) | 0.799 | 72.4% | 0.095 | 8.47M | 38.8% (gfgs) |
| **Modelo 05 R003** | M05 + stratified sampler (4-6× mais updates por classe rara) | 0.809 | 71.0% | 0.098 | 8.47M | 28.6% (gfgs) |
| **Modelo 05 R004** | M05 + partial unfreeze (last 4 DINOv2 blocks) | 0.812 | 72.9% | 0.119 | 36.8M | — |
| **Modelo 05 R005** | M05 + full unfreeze (LR=5e-6 M02-style) | **0.822** | 72.0% | 0.100 | 93.5M | 23.5% (gfgs) |
| **Modelo 05 R006** | M05 + heavy contrastive loss (peso 0.9) | 0.814 | 72.9% | 0.094 | 8.47M | 45.9% (gfgs) |
| **Modelo 05 R007** | M05 + DINOv2 + M02-trained-ViT hybrid backbone | 0.810 | 71.9% | 0.136 | 11.58M | 32.7% (gfgs) |
| **Modelo 06 R001** | Retrieval-Augmented (frozen ViT-B/16, K=32) | 0.776 | 69.8% | 0.062 | 8.16M | 61.2% (gmgs) |
| **Modelo 06 R002** | Retrieval-Augmented (frozen DINOv2, K=64) | 0.731 | 66.2% | 0.042 | ~12M | 41.3% (gmgs) |
| Codex VLM zero-shot binario | gpt-5.4-mini (`medium`) | — | 59.0% bal. | — | 0 (zero-shot) | non-kin: 38.1% |
| Claude Sonnet zero-shot binario | claude-sonnet-4-6 | — | 72.3% bal. | — | 0 (zero-shot) | non-kin: 63.9% |

*Threshold=0.5 default no evaluate.py — accuracy nao comparavel diretamente. Usar AUC.

**Insights:**
- Modelos parametricos puros (M02, M03) vencem em AUC absoluto (0.850).
- **Modelo 05 R001 tem o melhor TAR@FAR=0.01 (0.152)** — em regimes de threshold rigoroso, supera todos os outros, incluindo M02/M03. Util para aplicacoes que exigem baixa taxa de falsos positivos.
- **Modelo 05 atinge Val AUC 0.9116** (maior do projeto), mas o gap val→teste de -0.105 nao e fechado por regularizacao mais forte (R002 manteve gap em -0.106). Sugere divergencia estrutural entre distribuicoes de familia val/teste, nao overfitting do modelo.
- **Modelo 06 e 10x-20x menor em parametros treinaveis** e ainda assim atinge 0.776 AUC (Run 001).
- VLMs zero-shot agora foram avaliados tambem em verificacao binaria. O GPT fica em 58.98% de balanced accuracy e rejeita mal pares non-kin (38.1% de especificidade), enquanto o Claude chega a 72.28% de balanced accuracy. Mesmo com a tarefa binaria, os VLMs continuam abaixo dos modelos supervisionados calibrados no dominio.
- **Run 002 do M06 (DINOv2 + K=64) regrediu** em todas as metricas de teste, mostrando que o gargalo de M06 nao e backbone ou volume de retrieval, e sim regularizacao da cross-attention contra a galeria fixa.

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

### Experimento Exploratorio com VLM (Codex) — FIW, verificacao binaria

Como complemento aos modelos treinados, foi executado um experimento **zero-shot** com um VLM do ecossistema Codex, sem qualquer etapa de treinamento ou fine-tuning. A avaliacao foi refeita como **verificacao binaria**, para equiparar a formulacao da tarefa aos modelos supervisionados: dado um par de faces, o modelo deve responder apenas se existe ou nao parentesco biologico.

O resultado atual substitui, para fins de comparacao metodologica, o experimento anterior de classificacao 11-way. O experimento 11-way continua util como diagnostico historico de confusoes entre relacoes, mas nao e mais o numero principal do GPT.

**Configuracao do experimento:**
- **Modelo:** `gpt-5.4-mini` via Codex CLI
- **Modo:** zero-shot, `reasoning_effort=medium`
- **Task:** verificacao binaria (`kin` vs `non_kin`)
- **Amostra total:** **6.000 pares unicos** = 3.000 positivos + 3.000 non-kin
- **Execucao:** 750 pares iniciais + 5.250 pares complementares
- **Controle de duplicatas:** o lote complementar excluiu o manifesto dos 750 pares iniciais; o consolidado final tem `0` duplicatas
- **Entrada:** cada par foi convertido em uma imagem composta (face esquerda + face direita)
- **Artefatos:** `data/codex_vlm_fiw_binary_6000_medium_combined/`

### Resultados do GPT VLM binario (6.000 pares / 12.000 faces)

| Metrica | Valor |
|---------|------:|
| **Accuracy / Balanced accuracy** | **58.98%** |
| **Precision** | 0.563 |
| **Recall / Sensibilidade kin** | 0.799 |
| **Specificity / Rejeicao non-kin** | 0.381 |
| **F1** | 0.661 |
| Confidence media | 0.721 |
| Confidence media (acertos) | 0.735 |
| Confidence media (erros) | 0.700 |

### Matriz de Confusao — GPT VLM binario

| Classe real | Predito kin | Predito non-kin | Total |
|-------------|------------:|----------------:|------:|
| kin | 2.396 | 604 | 3.000 |
| non-kin | 1.857 | 1.143 | 3.000 |

### Interpretacao (GPT)

O GPT recupera bem pares positivos (`recall=0.799`), mas rejeita mal pares non-kin (`specificity=0.381`). O comportamento dominante e um **viés de aceitacao**: o modelo tende a responder `kin`, gerando 1.857 falsos positivos em 3.000 pares non-kin. Por isso, apesar de resolver a mesma tarefa binaria dos modelos supervisionados, o GPT zero-shot ainda nao oferece evidencia de substituicao de modelos treinados e calibrados para parentesco facial.

O ponto metodologico principal e que a discrepancia anterior foi removida: agora o VLM tambem recebe pares positivos e negativos e decide apenas entre `kin` e `non_kin`. A diferenca remanescente e de protocolo: o GPT nao e treinado no FIW, nao escolhe limiar em validacao e nao passa por validacao cruzada por familia.

### Adaptacao ao Dominio por Prompt no VLM (Codex) — experimento 11-way anterior

Como extensao do baseline zero-shot **11-way anterior**, foi testada uma forma de **adaptacao ao dominio baseada em inferencia**, sem fine-tuning do modelo. A ideia foi adaptar o comportamento do VLM ao dominio de parentesco facial por meio de:

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

Este bloco fica mantido como diagnostico historico da formulacao multiclasse. Ele nao substitui o resultado binario de 6.000 pares descrito acima.

---

### Experimento Zero-Shot com Claude Sonnet — FIW, verificacao binaria

Como segundo baseline zero-shot, o **Claude Sonnet** (Anthropic) tambem foi avaliado em verificacao binaria. A tarefa e equivalente a do GPT no espaco de saida (`kin` vs `non_kin`), embora os manifestos nao sejam exatamente pareados imagem a imagem.

**Configuracao:**
- **Modelo:** `claude-sonnet-4-6` (zero-shot direto por visao)
- **Entrada:** imagem composta lado a lado de dois rostos
- **Task:** verificacao binaria (`kin` vs `non_kin`)
- **Amostra:** 6.000 pares = 3.000 positivos + 3.000 non-kin
- **Artefatos:** `data/claude_vlm_fiw_binary_6000/`

### Resultados — Claude Sonnet binario (6.000 pares)

| Metrica | Valor |
|---------|------:|
| **Accuracy / Balanced accuracy** | **72.28%** |
| **Precision** | 0.691 |
| **Recall / Sensibilidade kin** | 0.806 |
| **Specificity / Rejeicao non-kin** | 0.639 |
| **F1** | 0.744 |
| ROC AUC | 0.789 |
| Average Precision | 0.742 |

### Matriz de Confusao — Claude Sonnet binario

| Classe real | Predito kin | Predito non-kin | Total |
|-------------|------------:|----------------:|------:|
| kin | 2.419 | 581 | 3.000 |
| non-kin | 1.082 | 1.918 | 3.000 |

### Interpretacao Comparativa

No enquadramento binario, o Claude Sonnet supera o GPT em balanced accuracy (72.28% vs 58.98%) e principalmente em especificidade (63.9% vs 38.1%). Os dois VLMs recuperam pares positivos em nivel semelhante, mas o GPT aceita muito mais pares non-kin como parentes. A comparacao confirma que a formulacao binaria reduz a discrepancia metodologica anterior, mas ainda nao torna os VLMs substitutos dos modelos supervisionados: a ausencia de treino no dominio, validacao de limiar e validacao cruzada continua sendo uma limitacao central.

**Artefatos Claude:** `data/claude_vlm_fiw_binary_6000/` | **Artefatos GPT:** `data/codex_vlm_fiw_binary_6000_medium_combined/`

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

8. **Baselines VLM zero-shot confirmam a necessidade de supervisao**: na verificacao binaria, o GPT (`gpt-5.4-mini`, `medium`) ficou em **58.98%** de balanced accuracy e o Claude Sonnet em **72.28%**. O GPT, em particular, rejeita mal pares non-kin (specificity = **0.381**), produzindo muitos falsos positivos.

9. **Adaptacao ao dominio por prompt em VLMs nao ajudou no experimento 11-way anterior**: o few-shot + prompt estruturado piorou o desempenho do Codex (33.1% -> 26.7%), engessando a decisao em heuristicas de idade/genero que amplificam vieses em vez de corrigir. Esse diagnostico fica separado do novo resultado binario de 6.000 pares.

10. **Retrieval-augmentation (Modelo 06) resolve o problema de classes raras** mas com AUC abaixo dos parametricos: o modelo retrieval-augmented com encoder congelado atinge Test AUC=0.776 (vs 0.850 dos parametricos), mas com **per-relacao muito mais uniforme** (61-87% em todas as 11 classes). Isso confirma que dar acesso explicito a exemplos de treino similares ajuda em classes com poucos exemplos, mas a complexidade extra da galeria + cross-attention nao supera os modelos parametricos quando ha dados suficientes. Apenas 8.16M parametros sao treinaveis (vs 86M-176M).

11. **Backbone melhor e K maior nao salvam o Modelo 06** (Run 002): trocar para DINOv2 e dobrar K para 64, mantendo encoder congelado, **piorou** o teste de 0.776 para 0.731 (-0.045 AUC). O gap val→teste cresceu de 0.06 para 0.09. Indica que o gargalo do retrieval-augmented nao e qualidade visual nem volume de contexto, e sim **regularizacao da cross-attention contra a galeria fixa**: features mais ricas + mais supports facilitam o overfitting na validacao em vez de melhorar generalizacao. Hard negatives nos supports e a hipotese a testar em runs futuras.

12. **Modelo 05 (DINOv2 + LoRA + DiffAttn) eleva o teto de Val AUC mas com gap val→teste alto:** Run 001 atingiu **Val AUC=0.9116** (maior do projeto) e **TAR@FAR=0.01=0.152** (melhor de todos), mas Test AUC ficou em 0.806 — gap de **-0.105**. A combinacao de DINOv2 self-supervised + LoRA rank=8 + differential cross-attention captura discriminacao mais forte que os modelos parametricos plenos. Apenas 8.47M parametros treinaveis (vs 86-176M dos parametricos plenos). Para regimes de threshold rigoroso (TAR@FAR=0.01), M05 R001 e a melhor opcao do projeto. O sinal multitarefa de classificacao de relacao (lambda=0.2) nao foi suficiente para resolver as classes de avo/avoa (40-52%) — e a relacao gfgs e a pior (39.8%), invertendo o padrao do M02 onde gfgs era a melhor.

13. **Regularizacao mais forte no Modelo 05 nao fecha o gap val→teste** (Run 002): aumentar relation_loss_weight para 0.4, adicionar LoRA dropout 0.1 e dropout 0.2, e cortar LR pico pela metade **manteve o gap em -0.106** (vs -0.105 da R001) e **reduziu TAR@FAR=0.01 de 0.152 para 0.095**. Aumentos pequenos (+3-6 pp) em classes intra-geracao (bb/ss/sibs) e mae/pai-filho(a) ao custo de precisao em thresholds rigorosos. **Implicacao central:** se overfitting fosse a causa do gap, R002 deveria fecha-lo; como nao fechou, o gap e provavelmente **estrutural — divergencia entre as distribuicoes de familia val e teste do RFIW Track-I**, nao propriedade do modelo. Isso desloca a investigacao para validacao cruzada k-fold e analise da composicao dos splits, em vez de ablacoes adicionais de hiperparametros. Aumentar lambda da relation-CE para 0.4 nao consertou as classes de avo/avoa, sugerindo que o gargalo nessas classes nao e o sinal multitarefa, mas insuficiencia de exemplos no treino.

14. **Class-balanced sampling tambem nao resolve as classes de avo** (Run 003 do Modelo 05): substituir o sampler aleatorio por `WeightedRandomSampler` que entrega 50% positivos divididos igualmente entre 11 relacoes do FIW + 50% negativos, dando **4-6× mais updates por epoch as classes raras** (gfgs vai de 1.03% para 4.54%), nao melhorou o desempenho dessas classes — gfgs caiu de 39.8% para 28.6%, todas as 11 classes regrediram, TAR@FAR=0.01 caiu para 0.098. **Implicacoes que reescrevem o diagnostico do M05:** (a) starvation de dados **nao** era o gargalo principal — mais updates sobre features fixas nao adiciona informacao que as features nao codificam; (b) o val→teste gap manteve-se em **-0.100**, igual a R001 (-0.105) e R002 (-0.106) — **tres intervencoes diferentes, mesmo gap, confirmando que o gap e propriedade do split RFIW Track-I, nao do modelo**; (c) o gargalo das classes de avo e arquitetonico — DINOv2 frozen extrai features que ja nao discriminam essas classes; (d) **R001 permanece como o checkpoint Pareto-otimal do M05** para uso de alta precisao (TAR@FAR=0.01 = 0.152, melhor de todo o projeto). A direcao que resta para mover o teto e descongelar parcialmente o DINOv2 — Run 004 planejada.

15. **Sete runs do M05, sete hipoteses rejeitadas, teto persistente em 0.81-0.82 AUC**: alem de R001-R003, foram testadas Run 004 (partial unfreeze dos ultimos 4 blocos DINOv2, AUC=0.812), Run 005 (full unfreeze com LR=5e-6 estilo M02, **AUC=0.822** — melhor M05 e menor val→teste gap em -0.071), Run 006 (heavy contrastive loss com peso 0.9, AUC=0.814) e Run 007 (**hibrido DINOv2 + M02-trained-ViT como backbones complementares frozen, AUC=0.810**). **R007 testou a hipotese mais ambiciosa**: combinar DINOv2 (general visual rico) com o ViT do M02 R031 (kinship-tuned via supervised contrastive) deveria gerar features complementares. Resultado: H0 confirmado, AUC identico a R001 sem o backbone face-specific. **Conclusao arquitetonica do M05:** as features kinship-fortes do M02 R031 nao sao do ViT alone — vem do **sistema completo treinado-em-conjunto** (ViT + FaCoR cross-attention + supervised contrastive + threshold 0.90 calibrados juntos). Reusar so os pesos do ViT em outra arquitetura quebra o ecossistema. **Para superar M02 (Test AUC 0.85) com filosofia M05, seria necessario abandonar o regime "frozen backbone + adapter"** e adotar full fine-tune com receita FRoundation (CosFace/ArcFace loss, LoRA rank 16, LR 1e-4) — mas isso replicaria a literatura sem novidade arquitetonica. **R001 permanece o checkpoint Pareto-otimal** do M05 (TAR@FAR=0.01 = 0.152, ainda melhor de todo o projeto). R005 e o melhor em ROC-AUC absoluto (0.822). R007 valida a tese comparativa do TCC com um achado negativo concreto: **fusao de backbones congelados nao recupera a performance de um sistema treinado-em-conjunto**.

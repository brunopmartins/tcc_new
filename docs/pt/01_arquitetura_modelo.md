# Arquitetura do Modelo ViT + FaCoR Cross-Attention

## Visao Geral

O modelo implementa uma arquitetura de **verificacao de parentesco** baseada em Vision Transformer (ViT) com mecanismo de cross-attention inspirado no FaCoR (Face Contrastive learning with Relation). O objetivo e determinar se duas imagens faciais pertencem a pessoas da mesma familia.

**Arquivo:** `models/02_vit_facor_crossattn/model.py`

---

## Fluxo Completo do Forward Pass

```
Imagem 1 [B, 3, 224, 224]     Imagem 2 [B, 3, 224, 224]
        |                              |
   ViT Backbone                   ViT Backbone (compartilhado)
        |                              |
  Patch Tokens [B, 195, 768]    Patch Tokens [B, 195, 768]
  CLS Token [B, 768]            CLS Token [B, 768]
        |                              |
        +--------- Cross-Attention ----+
        |          (bidirecional)       |
  Patches1 [B, 195, 768]       Patches2 [B, 195, 768]
        |                              |
   Global Avg Pool              Global Avg Pool
        |                              |
  feat1 [B, 768]                feat2 [B, 768]
        |                              |
   Channel Attention            Channel Attention
        |                              |
     + CLS Token                  + CLS Token
        |                              |
   Projection Head              Projection Head
        |                              |
  emb1 [B, 512]                emb2 [B, 512]
  (L2 normalizado)              (L2 normalizado)
```

---

## Componentes Detalhados

### 1. Backbone ViT (Vision Transformer)

**Linhas 150-238 em model.py**

O backbone utiliza um ViT pre-treinado do ImageNet (via biblioteca `timm`). O modelo padrao e `vit_base_patch16_224`:

- **Entrada:** imagem RGB 224x224
- **Patch embedding:** divide a imagem em patches de 16x16, gerando 196 patches (14x14)
- **Patch tokens:** cada patch vira um vetor de 768 dimensoes
- **CLS token:** token especial adicionado para representacao global
- **Saida:** `patch_tokens` [B, 195, 768] e `cls_token` [B, 768]

O mesmo backbone e compartilhado entre as duas imagens (siames), garantindo que ambas sejam projetadas no mesmo espaco de features.

**Opcao de congelamento (freeze):**
- `freeze_vit=True`: congela todos os pesos do ViT, treinando apenas o cross-attention e a projection head
- `unfreeze_after_epoch`: apos N epocas, descongela os ultimos K blocos do ViT para fine-tuning

```python
# Linhas 182-184: Congelamento do ViT
if freeze_vit:
    for param in self.vit.parameters():
        param.requires_grad = False
```

### 2. Cross-Attention Bidirecional

**Classe `CrossAttentionModule` — Linhas 25-113**

Este e o componente central do FaCoR. Ele permite que as features de uma face "olhem" para as features da outra, encontrando correspondencias faciais relevantes para parentesco.

**Mecanismo:**

1. **Queries de x1, Keys/Values de x2:**
   - Q1 = Linear(x1), K2 = Linear(x2), V2 = Linear(x2)
   - Atencao: `attn = softmax(Q1 * K2^T / sqrt(d_head))`
   - Saida: `out1 = attn * V2` (features de x1 informadas por x2)

2. **Caminho simetrico (x2 -> x1):**
   - Q2 = Linear(x2), K1 = Linear(x1), V1 = Linear(x1)
   - Mesmo calculo, direcao oposta

3. **Residual + LayerNorm + FFN:**
   - `out1 = LayerNorm(x1 + out1_proj)`
   - `out1 = LayerNorm(out1 + FFN(out1))`
   - FFN: Linear(768->3072) + GELU + Dropout + Linear(3072->768)

**Parametros:**
- `num_heads`: 8 (numero de cabecas de atencao)
- `num_layers`: 2 (numero de camadas de cross-attention empilhadas)
- `head_dim`: 96 (768 / 8)

**Intuicao:** A cross-attention permite que o modelo compare regioes especificas dos rostos (olhos, nariz, formato do rosto) entre as duas imagens, identificando semelhancas familiares.

### 3. Channel Attention (Atencao de Canal)

**Classe `ChannelAttention` — Linhas 116-147**

Mecanismo de squeeze-and-excitation que repondera os canais das features:

1. **Squeeze:** Global Average Pooling sobre a sequencia -> [B, 768]
2. **Excitation:** FC(768 -> 48) -> ReLU -> FC(48 -> 768) -> Sigmoid
3. **Reescala:** features * pesos_sigmoid

```python
# Linhas 135-145
weights = self.fc(pooled)      # [B, 768] -> [B, 48] -> [B, 768]
weights = torch.sigmoid(weights)
return x * weights.unsqueeze(1)  # Repondera cada canal
```

**Objetivo:** Aprender quais canais/features sao mais informativos para a tarefa de parentesco.

### 4. Projection Head (Cabeca de Projecao)

**Linhas 283-288 em ViTFaCoRModel**

Projeta as features de 768 dimensoes para o espaco de embeddings de 512 dimensoes:

```
Linear(768 -> 512) -> LayerNorm -> ReLU -> Dropout -> Linear(512 -> 512)
```

Seguido de **normalizacao L2**, colocando todos os embeddings na superficie de uma hiperesfera unitaria. Isso e essencial para que a similaridade do cosseno funcione corretamente.

### 5. Classifier Head (Opcional)

**Classe `ViTFaCoRClassifier` — Linhas 302-342**

Modo alternativo que adiciona uma cabeca de classificacao binaria:

1. Concatena: `[emb1, emb2, emb1-emb2, emb1*emb2]` -> [B, 2048]
2. Classifica: Linear(2048->256) -> ReLU -> Dropout -> Linear(256->1)
3. Saida: logit para BCEWithLogitsLoss

**Quando usar:** para treinamento com loss BCE em vez de contrastive loss.

---

## Funcoes Auxiliares

### `build_vit_facor_model()` (Linhas 345-370)

Funcao factory que cria o modelo correto baseado nos argumentos:

```python
model = build_vit_facor_model(
    vit_model="vit_base_patch16_224",
    embedding_dim=512,
    num_cross_attn_layers=2,
    cross_attn_heads=8,
    dropout=0.1,
    freeze_vit=False,
    use_classifier_head=False,
)
```

### `parse_model_outputs()` (Linhas 373-411)

Normaliza as saidas do modelo para um formato padrao:
- Modelo embedding: retorna `(emb1, emb2, attn_map)`
- Modelo classifier: retorna `(logits, emb1, emb2, attn_map)`

---

## Dimensoes em Cada Etapa

| Etapa | Formato | Dimensao |
|-------|---------|----------|
| Imagem de entrada | [B, 3, 224, 224] | 224x224 pixels RGB |
| Patch embeddings do ViT | [B, 196, 768] | 196 = 14x14 patches |
| Patch tokens (sem CLS) | [B, 195, 768] | 195 patches |
| CLS token | [B, 768] | Representacao global |
| Apos cross-attention | [B, 195, 768] | Features cruzadas |
| Apos pooling global | [B, 768] | Feature agregada |
| Apos channel attention | [B, 768] | Canais reponderados |
| Apos fusao com CLS | [B, 768] | Representacao combinada |
| Apos projection head | [B, 512] | Espaco de embedding |
| Embedding final (L2 norm) | [B, 512] | Esfera unitaria |

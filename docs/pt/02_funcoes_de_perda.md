# Funcoes de Perda (Loss Functions)

**Arquivo:** `models/shared/losses.py`

## Visao Geral

As funcoes de perda guiam o aprendizado do modelo. Para verificacao de parentesco, queremos que embeddings de pares kin (parentes) fiquem proximos e pares non-kin (nao-parentes) fiquem distantes no espaco de embeddings.

---

## 1. ContrastiveLoss (Perda Contrastiva Classica)

**Linhas 11-49**

Formula matematica:

```
L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(0, margin - D)^2
```

Onde:
- `Y = 1` para pares non-kin, `Y = 0` para pares kin
- `D` = distancia entre embeddings (euclidiana ou cosseno)
- `margin` = margem minima para pares negativos (padrao: 1.0)

**Comportamento:**
- **Pares kin (Y=0):** minimiza a distancia D (aproxima embeddings)
- **Pares non-kin (Y=1):** empurra a distancia para alem da margem

```python
# Linhas 30-40: Calculo da perda
if self.distance == "cosine":
    dist = 1 - F.cosine_similarity(emb1, emb2)
else:
    dist = F.pairwise_distance(emb1, emb2)

loss_pos = (1 - labels) * 0.5 * dist.pow(2)
loss_neg = labels * 0.5 * torch.clamp(self.margin - dist, min=0).pow(2)
return (loss_pos + loss_neg).mean()
```

**Esta e a loss usada nos melhores runs (030, 031, 032)** com `distance="cosine"` e `margin=0.3`.

---

## 2. CosineContrastiveLoss (InfoNCE / NT-Xent)

**Linhas 52-104**

Loss baseada em InfoNCE (Noise-Contrastive Estimation), popular em aprendizado contrastivo auto-supervisionado:

```
L = -log( exp(sim_pos / T) / sum(exp(sim_all / T)) )
```

Onde:
- `sim_pos` = similaridade do cosseno do par positivo
- `sim_all` = similaridades com todos os outros exemplos do batch
- `T` = temperatura (controla a "dureza" da distribuicao)

**Funcionamento:**
1. Concatena embeddings: `[emb1, emb2]` -> [2B, D]
2. Calcula matriz de similaridade: `sim = X * X^T / T`
3. Pares positivos: (i, i+B) sao verdadeiros pares
4. Loss: maximiza similaridade do par positivo vs todos os negativos

**Problema encontrado:** Esta loss **ignora os labels** — trata todo par (i, i+B) como positivo. Funciona para contrastivo auto-supervisionado, mas nao para verificacao de parentesco onde temos labels explicitos. Este bug foi identificado no run 18 e motivou a troca para `ContrastiveLoss`.

---

## 3. TripletLoss (Perda Tripla)

**Linhas 107-163**

Trabalha com tripletas (ancora, positivo, negativo):

```
L = max(0, D(ancora, positivo) - D(ancora, negativo) + margin)
```

**Hard Mining (mineracao de exemplos dificeis):**
- Seleciona o positivo mais distante (`hardest_positive`)
- Seleciona o negativo mais proximo (`hardest_negative`)
- Forca o modelo a lidar com os casos mais desafiadores

```python
# Linhas 130-148: Hard mining
for i in range(batch_size):
    # Positivo mais dificil: parente mais distante
    pos_mask = (labels == labels[i]) & (torch.arange(batch_size) != i)
    hardest_pos = dist_matrix[i][pos_mask].max()

    # Negativo mais dificil: nao-parente mais proximo
    neg_mask = labels != labels[i]
    hardest_neg = dist_matrix[i][neg_mask].min()

    triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
```

---

## 4. FairContrastiveLoss (Perda Contrastiva Justa)

**Linhas 166-244**

Incorpora margens demograficas para reduzir vies:

```
L = InfoNCE + alpha * bias_regularization
```

- Aplica margens diferentes por grupo demografico (idade, genero, etnia)
- Objetivo: manter performance similar entre todos os grupos
- Util quando o dataset tem representacao desigual de demografias

---

## 5. RelationGuidedContrastiveLoss

**Linhas 247-307**

Usa o mapa de atencao do cross-attention para modular a temperatura:

```
T_dinamica = T_base * (1 + alpha * atencao_media)
```

- **Alta atencao** -> temperatura mais alta -> loss mais suave (modelo esta "confiante")
- **Baixa atencao** -> temperatura mais baixa -> loss mais dura (modelo precisa aprender mais)

```python
# Linhas 279-280
attn_weights = attention_map.mean(dim=[1, 2, 3])  # Media global
temperature = self.base_temperature * (1 + self.alpha * attn_weights)
```

---

## 6. CombinedLoss

**Linhas 310-338**

Combinacao ponderada de multiplas losses:

```python
combined = CombinedLoss(
    losses=[ContrastiveLoss(), TripletLoss()],
    weights=[0.7, 0.3],
)
```

---

## Funcao Factory

### `get_loss()` (Linhas 341-351)

Cria a loss pelo nome:

```python
loss = get_loss("contrastive", margin=0.3, temperature=0.3)
```

Opcoes: `"bce"`, `"contrastive"`, `"cosine_contrastive"`, `"triplet"`, `"relation_guided"`, `"fair_contrastive"`

---

## Qual Loss Usar?

| Loss | Melhor Para | Observacao |
|------|-------------|------------|
| **ContrastiveLoss** | Verificacao de parentesco supervisionada | **Recomendada** — usa labels, distancia cosseno, margem ajustavel |
| CosineContrastiveLoss | Auto-supervisionado | Ignora labels — NAO usar para kin verification |
| TripletLoss | Hard mining | Boa alternativa, mas requer batches grandes |
| RelationGuidedContrastiveLoss | Multi-relacao (FIW) | Usa atencao como sinal — experimental |
| FairContrastiveLoss | Equidade demografica | Para mitigar vieses |

Para o projeto TCC, a **ContrastiveLoss com distancia cosseno e margem 0.3** produziu os melhores resultados.

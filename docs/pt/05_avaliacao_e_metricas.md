# Avaliacao e Metricas

**Arquivos:**
- `models/shared/evaluation.py` — calculo de metricas
- `models/shared/protocol.py` — protocolo de avaliacao com threshold

---

## Protocolo de Avaliacao

O protocolo segue o padrao da literatura de verificacao de parentesco:

```
1. Treina o modelo no conjunto de treino
2. Coleta predicoes no conjunto de VALIDACAO
3. Encontra threshold otimo no conjunto de validacao (maximiza F1)
4. Aplica o MESMO threshold no conjunto de TESTE
5. Reporta metricas do teste
```

**Por que esse protocolo?** O threshold nao pode ser otimizado no conjunto de teste — isso seria "vazamento de dados" (data leakage). O conjunto de validacao serve como proxy para selecionar o ponto de operacao.

---

## Coleta de Predicoes

### `collect_predictions()` (Linhas 258-295)

```python
def collect_predictions(model, dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_relations = []

    with torch.no_grad():
        for batch in dataloader:
            img1 = batch["img1"].to(device)
            img2 = batch["img2"].to(device)

            outputs = model(img1, img2)

            # Converte saida do modelo para score [0, 1]
            scores = _extract_prediction_tensor(outputs)

            all_predictions.extend(scores.cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_relations.extend(batch["relation"])

    return {
        "predictions": np.array(all_predictions),
        "labels": np.array(all_labels),
        "relations": all_relations,
    }
```

### `_extract_prediction_tensor()` (Linhas 207-255)

Converte a saida do modelo para um score entre 0 e 1:

```python
def _extract_prediction_tensor(outputs):
    if isinstance(outputs, tuple) and len(outputs) >= 2:
        emb1, emb2 = outputs[0], outputs[1]
        # Similaridade do cosseno: [-1, 1] -> [0, 1]
        cosine_sim = F.cosine_similarity(emb1, emb2)
        return (cosine_sim + 1) / 2
    elif outputs has logits:
        return torch.sigmoid(logits)
```

**Interpretacao do score:**
- `score = 1.0` -> modelo tem certeza que sao parentes
- `score = 0.0` -> modelo tem certeza que NAO sao parentes
- `score = 0.5` -> modelo esta incerto

---

## Busca de Threshold Otimo

### `find_optimal_threshold()` (Linhas 334-359)

Busca em grade (grid search) pelo threshold que maximiza a metrica escolhida:

```python
def find_optimal_threshold(predictions, labels, metric="f1"):
    best_threshold = 0.5
    best_score = 0.0

    # Busca de 0.10 a 0.95, passo 0.05
    for threshold in np.arange(0.10, 0.96, 0.05):
        binary_preds = (predictions >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(labels, binary_preds)
        elif metric == "accuracy":
            score = accuracy_score(labels, binary_preds)
        elif metric == "balanced":
            score = balanced_accuracy_score(labels, binary_preds)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```

**Nos melhores runs, o threshold otimo ficou em 0.85-0.90**, indicando que o modelo tende a dar scores altos para todos os pares.

---

## Metricas Calculadas

### Classe KinshipMetrics (Linhas 28-204)

#### Metricas Basicas

| Metrica | Descricao | Formula |
|---------|-----------|---------|
| **Accuracy** | Proporcao de acertos | (VP + VN) / (VP + VN + FP + FN) |
| **Balanced Accuracy** | Media da acuracia por classe | (TPR + TNR) / 2 |
| **Precision** | Dos que previu como kin, quantos sao? | VP / (VP + FP) |
| **Recall** | Dos que sao kin, quantos detectou? | VP / (VP + FN) |
| **F1** | Media harmonica de precision e recall | 2 * P * R / (P + R) |
| **ROC AUC** | Area sob a curva ROC | Probabilidade de rankear positivo > negativo |
| **Average Precision** | Area sob a curva precision-recall | Integral da curva PR |

Onde: VP = Verdadeiro Positivo, VN = Verdadeiro Negativo, FP = Falso Positivo, FN = Falso Negativo, TPR = Taxa de Verdadeiros Positivos, TNR = Taxa de Verdadeiros Negativos.

#### TAR @ FAR (Linhas 104-121)

**True Accept Rate at False Accept Rate** — metrica de seguranca:

| FAR | Significado |
|-----|-------------|
| TAR@FAR=0.001 | De cada 1000 nao-parentes, aceita apenas 1. Qual % dos parentes aceita? |
| TAR@FAR=0.01 | De cada 100 nao-parentes, aceita apenas 1. Qual % dos parentes aceita? |
| TAR@FAR=0.1 | De cada 10 nao-parentes, aceita apenas 1. Qual % dos parentes aceita? |

**Como calcula:**

```python
def _compute_tar_at_far(predictions, labels, far_targets=[0.001, 0.01, 0.1]):
    fpr, tpr, thresholds = roc_curve(labels, predictions)

    results = {}
    for target_far in far_targets:
        # Encontra o ponto na curva ROC onde FPR <= target_far
        idx = np.searchsorted(fpr, target_far, side="right") - 1
        results[f"tar@far={target_far}"] = tpr[idx]

    return results
```

#### Metricas por Relacao (Linhas 123-155)

Calcula metricas separadas para cada tipo de relacao (fd, fs, md, ms, bb, ss, etc.):

```python
def _compute_per_relation_metrics(predictions, labels, relations, threshold):
    results = {}
    for rel in unique_relations:
        mask = [r == rel for r in relations]
        rel_preds = predictions[mask]
        rel_labels = labels[mask]

        results[rel] = {
            "accuracy": accuracy_score(rel_labels, rel_preds >= threshold),
            "f1": f1_score(rel_labels, rel_preds >= threshold),
            "count": len(rel_labels),
        }
    return results
```

**Exemplo de saida (run 031):**

| Relacao | Acc | F1 | N |
|---------|-----|------|-----|
| bb | 95.5% | 0.977 | 860 |
| fs | 95.3% | 0.976 | 1135 |
| gmgs | 88.4% | 0.939 | 121 |

---

## Protocolo Completo de Avaliacao

### `evaluate_with_validation_threshold()` (protocol.py, Linhas 159-200)

```python
def evaluate_with_validation_threshold(model, val_loader, test_loader, device):
    # 1. Coleta predicoes na validacao
    val_bundle = collect_predictions(model, val_loader, device)

    # 2. Encontra threshold otimo na validacao
    threshold, threshold_score = find_optimal_threshold(
        val_bundle["predictions"],
        val_bundle["labels"],
        metric="f1",
    )

    # 3. Calcula metricas na validacao com o threshold
    val_metrics = compute_metrics_from_predictions(
        predictions=val_bundle["predictions"],
        labels=val_bundle["labels"],
        threshold=threshold,
    )

    # 4. Coleta predicoes no teste
    test_bundle = collect_predictions(model, test_loader, device)

    # 5. Aplica MESMO threshold no teste
    test_metrics = compute_metrics_from_predictions(
        predictions=test_bundle["predictions"],
        labels=test_bundle["labels"],
        threshold=threshold,  # <-- Mesmo threshold da validacao!
    )

    return {
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
```

---

## Agregacao de Metricas (Cross-Validation)

### `aggregate_numeric_metrics()` (protocol.py, Linhas 203-225)

Para k-fold CV, calcula media e desvio padrao:

```python
def aggregate_numeric_metrics(fold_results):
    # Extrai valores numericos de cada fold
    all_values = {}
    for result in fold_results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                all_values.setdefault(key, []).append(value)

    # Calcula mean +- std
    aggregated = {}
    for key, values in all_values.items():
        aggregated[f"mean_{key}"] = np.mean(values)
        aggregated[f"std_{key}"] = np.std(values)

    return aggregated
```

**Exemplo de saida (run 026, 5-fold CV no KinFaceW):**

```
Mean Accuracy:   73.56% +- 5.84%
Mean F1:         0.771  +- 0.028
Mean ROC AUC:    0.842  +- 0.012
```

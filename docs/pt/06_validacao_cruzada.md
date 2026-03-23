# Validacao Cruzada (K-Fold Cross-Validation)

**Arquivo:** `models/02_vit_facor_crossattn/AMD/train_cv.py`

---

## O que e Validacao Cruzada?

Em datasets pequenos (como KinFaceW com ~500 pares), um unico split train/test pode dar resultados instáveis dependendo de quais amostras caem em cada split. A validacao cruzada resolve isso:

1. Divide os dados em **K folds** (partes iguais)
2. Para cada fold: treina em K-1 partes, testa em 1 parte
3. Repete K vezes, rotacionando qual parte e o teste
4. Reporta **media +- desvio padrao** das metricas

```
Fold 1: [TESTE] [treino] [treino] [treino] [treino]
Fold 2: [treino] [TESTE] [treino] [treino] [treino]
Fold 3: [treino] [treino] [TESTE] [treino] [treino]
Fold 4: [treino] [treino] [treino] [TESTE] [treino]
Fold 5: [treino] [treino] [treino] [treino] [TESTE]
```

---

## Splits Disjuntos

### KinFaceW: Pair-Disjoint

Nenhum **par** de imagens aparece em mais de um fold. Cada par (pai-filho, mae-filha, etc.) e atribuido a exatamente um fold.

### FIW: Family-Disjoint

Nenhuma **familia** aparece em mais de um fold. Todas as imagens de uma familia vao para o mesmo fold. Isso e mais rigoroso e evita que o modelo "decore" rostos familiares.

```python
# Atribuicao deterministia de folds
test_ids = {id for i, id in enumerate(all_ids) if i % n_folds == fold_k}
fold_train = [id for i, id in enumerate(all_ids) if i % n_folds != fold_k]
train_ids, val_ids = _split_train_val_ids(fold_train, seed)
```

---

## Pipeline de Cada Fold

### `run_fold()` (Linhas 129-225)

```python
def run_fold(fold_k, n_folds, args):
    # 1. Cria dataloaders para este fold
    train_loader, val_loader, test_loader = create_cv_fold_loaders(
        config, fold_k=fold_k, n_folds=n_folds,
        dataset_type=args.train_dataset,
    )

    # 2. Cria modelo (novo para cada fold)
    model = build_vit_facor_model(...)

    # 3. Cria loss e trainer
    loss_fn = ViTFaCoRLoss(args.loss, args.temperature, args.margin)
    trainer = ViTFaCoRROCmTrainer(model, loss_fn, train_loader, val_loader, ...)

    # 4. Treina
    trainer.train()

    # 5. Carrega melhor checkpoint
    load_best_checkpoint(model, checkpoint_dir)

    # 6. Avalia com threshold da validacao
    results = evaluate_with_validation_threshold(
        model, val_loader, test_loader, device
    )

    return results["test_metrics"]
```

### Agregacao Final (Linhas 228-307)

```python
def main():
    all_fold_results = []

    for fold_k in range(n_folds):
        print(f"Fold {fold_k + 1} / {n_folds}")
        fold_metrics = run_fold(fold_k, n_folds, args)
        all_fold_results.append(fold_metrics)

    # Calcula media e desvio padrao
    summary = aggregate_numeric_metrics(all_fold_results)

    # Salva resultados
    save_json(output_path / "cv_summary.json", {
        "n_folds": n_folds,
        "fold_results": all_fold_results,
        **summary,  # mean_accuracy, std_accuracy, etc.
    })
```

---

## Como Executar

### Via run_pipeline.sh

```bash
CV_FOLDS=5 EPOCHS=50 LOSS=contrastive MARGIN=0.3 \
LEARNING_RATE=1e-5 SCHEDULER=cosine \
bash AMD/run_pipeline.sh
```

### Diretamente

```bash
python AMD/train_cv.py \
    --train_dataset kinface \
    --n_folds 5 \
    --epochs 50 \
    --lr 1e-5 \
    --loss contrastive \
    --margin 0.3
```

---

## Exemplo de Resultado (Run 026 — KinFaceW-I)

```json
{
  "n_folds": 5,
  "mean_accuracy": 0.7356,
  "std_accuracy": 0.0584,
  "mean_f1": 0.7707,
  "std_f1": 0.0283,
  "mean_roc_auc": 0.8420,
  "std_roc_auc": 0.0123,
  "fold_results": [
    {"accuracy": 0.6215, "f1": 0.7216, "roc_auc": 0.8240},
    {"accuracy": 0.7523, "f1": 0.7706, "roc_auc": 0.8310},
    {"accuracy": 0.7570, "f1": 0.7833, "roc_auc": 0.8558},
    {"accuracy": 0.7877, "f1": 0.8085, "roc_auc": 0.8494},
    {"accuracy": 0.7594, "f1": 0.7692, "roc_auc": 0.8498}
  ]
}
```

**Interpretacao:**
- **AUC medio de 0.842 +- 0.012** indica boa capacidade discriminativa
- **Desvio padrao baixo** sugere modelo estavel entre folds
- **Fold 1** e outlier (62.2% acc) — provavel que tenha pares mais dificeis

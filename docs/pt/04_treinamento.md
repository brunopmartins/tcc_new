# Pipeline de Treinamento

**Arquivos:**
- `models/02_vit_facor_crossattn/AMD/train.py` — script principal de treinamento
- `models/shared/AMD/trainer.py` — loop de treinamento otimizado para AMD ROCm
- `models/02_vit_facor_crossattn/AMD/run_pipeline.sh` — orquestrador do pipeline

---

## Visao Geral do Pipeline

```
run_pipeline.sh
    |
    +-- [1/3] train.py       -> Treina o modelo, salva melhor checkpoint
    +-- [2/3] test.py        -> Avalia no conjunto de teste
    +-- [3/3] evaluate.py    -> Analise completa com visualizacoes
```

---

## Script de Treinamento (train.py)

### Argumentos Principais

**Linhas 58-136**

| Argumento | Padrao | Descricao |
|-----------|--------|-----------|
| `--train_dataset` | kinface | Dataset de treino ("kinface" ou "fiw") |
| `--test_dataset` | kinface | Dataset de teste |
| `--epochs` | 100 | Numero maximo de epocas |
| `--batch_size` | 32 | Tamanho do batch |
| `--lr` | 1e-5 | Taxa de aprendizado |
| `--weight_decay` | 1e-5 | Regularizacao L2 |
| `--scheduler` | cosine | Escalonador de LR ("cosine", "plateau", "none") |
| `--warmup_epochs` | 5 | Epocas de aquecimento |
| `--loss` | cosine_contrastive | Funcao de perda |
| `--margin` | 0.5 | Margem para perda contrastiva |
| `--temperature` | 0.3 | Temperatura para losses baseadas em InfoNCE |
| `--negative_ratio` | 1.0 | Razao negativos/positivos no treino |
| `--freeze_vit` | False | Congela backbone ViT |
| `--unfreeze_after_epoch` | 0 | Descongela apos N epocas |
| `--dropout` | 0.1 | Taxa de dropout |
| `--patience` | 50 | Paciencia para early stopping |

### Wrapper de Loss (ViTFaCoRLoss)

**Linhas 139-166**

Adapta as saidas do modelo para a funcao de perda:

```python
class ViTFaCoRLoss(nn.Module):
    def forward(self, emb1, emb2, labels, attn_map=None):
        if self.loss_type == "bce":
            similarity = torch.sum(emb1 * emb2, dim=1)
            return self.loss_fn(similarity, labels)
        elif self.loss_type == "relation_guided" and attn_map is not None:
            attn_weights = attn_map.mean(dim=[1, 2, 3])
            return self.loss_fn(emb1, emb2, attn_weights)
        else:
            return self.loss_fn(emb1, emb2, labels)
```

### Trainer Customizado (ViTFaCoRROCmTrainer)

**Linhas 168-235**

Estende o `ROCmTrainer` com:
- **Descongelamento agendado:** descongela os ultimos blocos do ViT apos N epocas
- **Calculo de loss customizado:** passa embeddings E mapa de atencao para a loss

```python
class ViTFaCoRROCmTrainer(ROCmTrainer):
    def on_epoch_start(self, epoch):
        # Descongela ultimos blocos do ViT apos a epoca configurada
        if epoch == self.unfreeze_after_epoch:
            self._unfreeze_vit_tail()

    def _unfreeze_vit_tail(self):
        # Descongela apenas os ultimos K blocos do transformer
        blocks = list(model.vit.blocks)
        for block in blocks[-self.unfreeze_last_vit_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
```

---

## Loop de Treinamento (trainer.py)

**Arquivo:** `models/shared/AMD/trainer.py`

### Classe ROCmTrainer

**Linhas 30-346**

Trainer otimizado para GPUs AMD com ROCm (HIP).

### Inicializacao (Linhas 38-106)

1. **Otimizador:** AdamW com weight decay
2. **Scheduler:** CosineAnnealingLR ou ReduceLROnPlateau
3. **Mixed Precision:** AMP (Automatic Mixed Precision) com GradScaler
4. **Metrica monitorada:** `roc_auc` (padrao)

```python
self.optimizer = AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
    eps=1e-8,
)

self.scheduler = CosineAnnealingLR(
    self.optimizer,
    T_max=max(config.num_epochs - config.warmup_epochs, 1),
    eta_min=config.min_lr,  # 1e-7
)
```

### Epoca de Treinamento (Linhas 109-159)

```python
def train_epoch(self):
    self.model.train()

    for batch in train_loader:
        img1 = batch["img1"].to(device)     # [B, 3, 224, 224]
        img2 = batch["img2"].to(device)     # [B, 3, 224, 224]
        labels = batch["label"].to(device)  # [B]

        # Zero gradientes
        self.optimizer.zero_grad(set_to_none=True)

        # Forward pass com AMP
        with autocast():
            outputs = self.model(img1, img2)  # (emb1, emb2, attn_map)
            loss = self._compute_loss(outputs, labels)

        # Backward com gradient scaling
        self.scaler.scale(loss).backward()

        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Passo do otimizador
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

### Validacao (Linhas 169-183)

Valida com **busca de threshold otimo**:

```python
def validate(self):
    bundle = collect_predictions(self.model, self.val_loader, self.device)

    # Encontra threshold que maximiza F1
    threshold, _ = find_optimal_threshold(
        bundle["predictions"], bundle["labels"], metric="f1"
    )

    # Calcula metricas com o threshold otimo
    metrics = compute_metrics_from_predictions(
        predictions=bundle["predictions"],
        labels=bundle["labels"],
        threshold=threshold,
    )
    return metrics
```

### Loop Principal (Linhas 225-305)

```python
def train(self):
    for epoch in range(num_epochs):
        # 1. Hook pre-epoca (descongelamento agendado)
        self.on_epoch_start(epoch + 1)

        # 2. Warmup de learning rate (linear)
        if epoch < warmup_epochs:
            lr = base_lr * (epoch + 1) / warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

        # 3. Treina uma epoca
        train_loss = self.train_epoch()

        # 4. Valida
        val_metrics = self.validate()

        # 5. Atualiza scheduler
        if epoch >= warmup_epochs:
            self.scheduler.step()

        # 6. Salva melhor modelo
        current_auc = val_metrics.get("roc_auc", 0)
        if current_auc > best_auc + min_delta:
            best_auc = current_auc
            self.save_checkpoint("best.pt", val_metrics, epoch)
            patience_counter = 0
        else:
            patience_counter += 1

        # 7. Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # 8. Safeguards (parada automatica por metricas degeneradas)
        if self._check_safeguards(epoch + 1):
            break
```

### Safeguards (Protecoes Automaticas)

**Linhas 189-223**

Duas protecoes contra treinamento degenerado:

1. **Declinio continuo de AUC** (10 epocas consecutivas):
   ```python
   # Se AUC cai por 10 epocas seguidas, para o treinamento
   if len(auc_history) >= 10:
       recent = auc_history[-10:]
       if all(recent[i] >= recent[i+1] for i in range(9)):
           print("Safeguard: AUC declining for 10 epochs")
           return True
   ```

2. **Acuracia degenerada** (50% por 8 epocas em dataset balanceado):
   ```python
   # Se acuracia = 50% por 8 epocas, threshold esta degenerado
   if len(acc_history) >= 8:
       recent = acc_history[-8:]
       if all(abs(a - 0.5) < 0.02 for a in recent):
           print("Safeguard: degenerate accuracy")
           return True
   ```

---

## Escalonadores de Learning Rate

### Cosine Annealing (Padrao)

```
LR
 ^
 |  /\
 | /  \
 |/    \_________
 +----------------> epocas
   warmup  decay
```

- **Warmup:** LR sobe linearmente de 0 ate `base_lr` em `warmup_epochs`
- **Decay:** LR decai seguindo cosseno ate `min_lr`

### ReduceLROnPlateau

- Reduz LR por fator 0.5 se a metrica monitorada nao melhora por 5 epocas
- Mais adaptativo, mas pode ficar preso em minimos locais

---

## Gradient Clipping

**Linhas 138-142**

Limita a norma dos gradientes para evitar explosao:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm=1.0)
```

Essencial para estabilidade com fine-tuning de ViT, onde gradientes podem ser muito grandes nas primeiras camadas.

---

## Mixed Precision (AMP)

**Linhas 89-94**

Usa precisao mista (FP16/FP32) para:
- **Economia de memoria:** operacoes FP16 usam metade da VRAM
- **Velocidade:** GPUs AMD (e NVIDIA) sao mais rapidas em FP16

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(img1, img2)
    loss = compute_loss(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

O `GradScaler` ajusta a escala dos gradientes para evitar underflow em FP16.

---

## Pipeline Shell (run_pipeline.sh)

**Arquivo:** `models/02_vit_facor_crossattn/AMD/run_pipeline.sh`

### Variaveis de Ambiente

```bash
TRAIN_DATASET=fiw         # Dataset
EPOCHS=50                 # Epocas
LEARNING_RATE=2e-6        # Taxa de aprendizado
SCHEDULER=cosine          # Escalonador
FREEZE_VIT=0              # Congelar ViT (0=nao, 1=sim)
NEGATIVE_RATIO=2.0        # Razao de negativos
CV_FOLDS=5                # Folds para validacao cruzada (0=desabilitado)
LOSS=contrastive          # Funcao de perda
DROPOUT=0.2               # Dropout
```

### Modo Validacao Cruzada

```bash
if [ "${CV_FOLDS}" -gt 0 ]; then
    python train_cv.py ${SHARED_TRAIN_ARGS[@]} --n_folds ${CV_FOLDS}
else
    python train.py ${SHARED_TRAIN_ARGS[@]}
    python test.py ...
    python evaluate.py ...
fi
```

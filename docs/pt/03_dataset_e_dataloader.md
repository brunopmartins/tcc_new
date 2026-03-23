# Dataset e Carregamento de Dados

**Arquivo principal:** `models/shared/dataset.py`
**Configuracao:** `models/shared/config.py`

## Visao Geral

O sistema suporta dois datasets de verificacao de parentesco:

| Dataset | Familias | Imagens | Relacoes | Escala |
|---------|----------|---------|----------|--------|
| **KinFaceW-I** | ~100 pares | ~1.000 | 4 (fd, fs, md, ms) | Pequeno |
| **FIW (Families in the Wild)** | 980 familias | 50.420 | 11 tipos | Grande |

---

## Classe Principal: KinshipPairDataset

**Linhas 77-559**

Dataset PyTorch que retorna pares de imagens faciais com label de parentesco.

### Inicializacao

```python
dataset = KinshipPairDataset(
    root_dir="/caminho/para/dataset",
    dataset_type="fiw",              # "kinface" ou "fiw"
    split="train",                    # "train", "val", "test"
    transform=transforms,            # augmentacao de imagens
    negative_ratio=2.0,              # razao negativos/positivos
    negative_sampling_strategy="relation_matched",  # "random" ou "relation_matched"
    split_seed=42,                    # semente para reprodutibilidade
)
```

### Saida de Cada Amostra

```python
sample = dataset[0]
# sample = {
#     "img1": tensor [3, 224, 224],   # primeira face
#     "img2": tensor [3, 224, 224],   # segunda face
#     "label": tensor(1.0),           # 1=parente, 0=nao-parente
#     "relation": "fs",               # tipo de relacao
# }
```

---

## Carregamento do FIW (RFIW Track-I)

### Protocolo Padrao RFIW

O FIW usa o protocolo do **RFIW Challenge** (Recognizing Families in the Wild):

1. **`train-pairs.csv`** — 6.983 pares a nivel de membro (ex: `F0001/MID1, F0001/MID3`)
2. **`val-pairs.csv`** — 2.186 pares de validacao
3. **`test-pairs.csv`** — 39.743 pares a nivel de face com labels (positivos e negativos)

### Expansao de Pares (Membro -> Face)

**Linhas 322-407 — `_load_fiw_rfiw_train_val()`**

Cada par de membros e expandido para pares de faces:

```
Par membro: F0001/MID1 <-> F0001/MID3

F0001/MID1 contem:          F0001/MID3 contem:
  P00001_face2.jpg            P00003_face1.jpg
  P00002_face3.jpg            P00004_face3.jpg
  P00007_face2.jpg

Pares de faces gerados (produto cartesiano, max 10):
  (P00001_face2, P00003_face1) -> kin
  (P00001_face2, P00004_face3) -> kin
  (P00002_face3, P00003_face1) -> kin
  ...
```

O parametro `max_face_pairs=10` limita o numero de pares por par de membros para controlar o tamanho do dataset.

### Resultado Final (com neg_ratio=2.0)

| Split | Positivos | Negativos | Total |
|-------|-----------|-----------|-------|
| Train | 33.207 | 66.414 | 99.621 |
| Val | 5.695 | 5.695 | 11.390 |
| Test | 6.432 | 6.993 | 13.425 |

---

## Amostragem de Negativos

### Estrategia "random" (Linhas 506-531)

Seleciona aleatoriamente duas familias diferentes e pega uma imagem de cada:

```python
fid1, fid2 = rng.sample(families, 2)  # Duas familias diferentes
img1 = rng.choice(images_by_family[fid1])
img2 = rng.choice(images_by_family[fid2])
```

### Estrategia "relation_matched" (Linhas 409-443)

Gera negativos preservando a distribuicao de tipos de relacao:

```python
# Seleciona tipo de relacao baseado na distribuicao dos positivos
rel = rng.choice(rel_choices)  # Ex: "fd" aparece proporcionalmente
# Seleciona imagens de familias diferentes
fid1, fid2 = rng.sample(families, 2)
```

**Vantagem:** negativos mais dificeis, pois preservam a estrutura demografica (pai-filha negativo vs pai-filha positivo).

---

## Carregamento do KinFaceW

### Estrutura do KinFaceW-I (Linhas 101-167)

```
KinFaceW-I/
  images/
    father-dau/      # Relacao pai-filha (fd)
      fd_001_1.jpg   # Pai do par 001
      fd_001_2.jpg   # Filha do par 001
    father-son/      # Relacao pai-filho (fs)
    mother-dau/      # Relacao mae-filha (md)
    mother-son/      # Relacao mae-filho (ms)
```

Cada par tem exatamente 2 imagens (pessoa 1 e pessoa 2). Os splits sao **pair-disjoint** (nenhum par aparece em mais de um split).

---

## Splits Deterministicos

**Linhas 36-45 — `_split_id_sets()`**

Divisao 70/15/15 com semente fixa:

```python
def _split_id_sets(ids, split_seed):
    ordered = _shuffled_ids(ids, split_seed)  # Shuffle deterministico
    n = len(ordered)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    return {
        "train": set(ordered[:n_train]),
        "val": set(ordered[n_train:n_train + n_val]),
        "test": set(ordered[n_train + n_val:]),
    }
```

**Garantia:** Com `split_seed=42`, os mesmos dados sempre vao para os mesmos splits, garantindo reprodutibilidade.

---

## Transformacoes de Imagem

**Linhas 562-577 — `get_transforms()`**

### Treinamento (com augmentacao)

```python
T.Compose([
    T.Resize((224, 224)),          # Redimensiona
    T.RandomHorizontalFlip(p=0.5), # Flip horizontal aleatorio
    T.RandomRotation(10),           # Rotacao aleatoria +-10 graus
    T.ToTensor(),                   # Converte para tensor [0,1]
    T.ColorJitter(                  # Variacao de cor
        brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.1
    ),
    T.Normalize(                    # Normalizacao ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
```

### Avaliacao (sem augmentacao)

```python
T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Nota:** `ToTensor()` vem antes de `ColorJitter` para evitar overflow com NumPy 2.0.

---

## Validacao Cruzada (K-Fold CV)

**Linhas 712-801 — `create_cv_fold_loaders()`**

Cria splits para k-fold cross-validation:

- **KinFaceW:** splits sao **pair-disjoint** (pares nunca compartilhados entre folds)
- **FIW:** splits sao **family-disjoint** (familias inteiras em cada fold)

```python
# Atribuicao de folds (deterministico)
test_ids = {id for i, id in enumerate(all_ids) if i % n_folds == fold_k}
train_ids = {id for i, id in enumerate(all_ids) if i % n_folds != fold_k}
train_ids, val_ids = _split_train_val_ids(train_ids, seed)
```

Para 5-fold CV: cada fold usa 80% para treino (dos quais 15% para validacao) e 20% para teste.

---

## Configuracao (DataConfig)

**Arquivo:** `models/shared/config.py`, Linhas 10-41

```python
@dataclass
class DataConfig:
    fiw_root: str = "/home/bruno/Desktop/tcc_new/datasets/FIW"
    kinface_i_root: str = "/home/bruno/Desktop/tcc_new/datasets/KinFaceW-I"
    image_size: int = 224
    split_seed: int = 42
    negative_ratio: float = 1.0
    num_workers: int = 4
    relation_types: List[str] = ["fd", "fs", "md", "ms", "bb", "ss",
                                  "sibs", "gfgd", "gfgs", "gmgd", "gmgs"]
```

Os 11 tipos de relacao do FIW:

| Codigo | Relacao |
|--------|---------|
| fd | pai-filha |
| fs | pai-filho |
| md | mae-filha |
| ms | mae-filho |
| bb | irmao-irmao |
| ss | irma-irma |
| sibs | irmaos (misto) |
| gfgd | avo(paterno)-neta |
| gfgs | avo(paterno)-neto |
| gmgd | avo(materna)-neta |
| gmgs | avo(materna)-neto |

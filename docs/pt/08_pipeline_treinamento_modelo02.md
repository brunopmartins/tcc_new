# Pipeline de Treinamento — Modelo 02 (ViT + FaCoR Cross-Attention)

## Visao Geral

O pipeline de treinamento do Modelo 02 e orquestrado por um unico script shell (`run_pipeline.sh`) que opera em dois modos, dependendo da configuracao:

**Modo Single-Run** (para experimentacao rapida):
```
run_pipeline.sh
    |
    +-- [1/3] train.py      -> Treina o modelo e salva o melhor checkpoint
    +-- [2/3] test.py       -> Avalia no conjunto de teste com metricas detalhadas
    +-- [3/3] evaluate.py   -> Gera visualizacoes e analise completa
```

**Modo Validacao Cruzada (K-Fold)** (para avaliacao final robusta):
```
run_pipeline.sh  (CV_FOLDS=5)
    |
    +-- [1/1] train_cv.py   -> Treina e avalia K modelos independentes
                                reporta media +- desvio padrao das metricas
```

O modo e selecionado pela variavel `CV_FOLDS`:
```bash
# Single-run (padrao)
bash AMD/run_pipeline.sh

# 5-fold cross-validation
CV_FOLDS=5 bash AMD/run_pipeline.sh
```

Cada execucao cria uma pasta numerada (`output/001/`, `output/002/`, ...) contendo checkpoints, resultados e logs, garantindo que nenhum resultado anterior seja sobrescrito.

---

## Passo 1: Inicializacao do Ambiente (`run_pipeline.sh`)

### 1.1 Resolucao de Caminhos

O script determina automaticamente sua localizacao no sistema de arquivos:

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"    # .../AMD/
MODEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                   # .../02_vit_facor_crossattn/
PROJECT_ROOT="$(cd "${MODEL_ROOT}/../.." && pwd)"              # .../tcc_new/
```

Isso permite que o script seja executado de qualquer diretorio do projeto sem que os caminhos quebrem.

### 1.2 Configuracao da GPU AMD (ROCm)

A GPU utilizada (RX 6750 XT, arquitetura gfx1031) nao e oficialmente suportada pelo ROCm. Para contornar isso, o script define variaveis de ambiente que fazem o runtime tratar a GPU como um modelo compativel (gfx1030):

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0   # Faz a GPU ser reconhecida como gfx1030
export MIOPEN_FIND_MODE=FAST             # Busca rapida de algoritmos de convolucao
export HIP_VISIBLE_DEVICES=0             # Seleciona a GPU 0
```

### 1.3 Permissoes de Acesso a GPU

O acesso ao dispositivo `/dev/kfd` (Kernel Fusion Driver) requer que o usuario pertenca ao grupo `render`. O script verifica isso e, se necessario, reinicia a si mesmo com o grupo correto ativo:

```bash
if ! id -Gn | grep -qx render; then
    sudo usermod -aG render,video "$(whoami)"
fi
exec sg render -c "bash '${SELF}'"
```

Isso evita que o usuario precise fazer logout/login para que as permissoes tenham efeito.

### 1.4 Ambiente Virtual Python

O script verifica se o ambiente virtual (`.venv`) existe. Na primeira execucao, ele cria o venv e instala todas as dependencias:

```bash
python3 -m venv "${MODEL_ROOT}/.venv"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
pip install timm numpy pandas scikit-learn Pillow tqdm matplotlib seaborn
```

Nas execucoes seguintes, esta etapa pode ser pulada com `SKIP_INSTALL=1`.

### 1.5 Criacao da Pasta de Saida

O script encontra o maior numero de diretorio existente em `output/` e cria o proximo:

```bash
# Se existem output/001/, output/002/, o proximo sera output/003/
RUN_ID=$((RUN_ID + 1))
RUN_LABEL="$(printf '%03d' ${RUN_ID})"
mkdir -p "${RUN_DIR}/checkpoints" "${RUN_DIR}/results" "${RUN_DIR}/logs"
```

Estrutura criada:
```
output/003/
    checkpoints/   <- pesos do modelo (best.pt, final.pt)
    results/       <- metricas e graficos
    logs/          <- logs de saida do terminal
```

---

## Passo 2: Carregamento dos Dados (`dataset.py`)

### 2.1 Leitura dos Pares

Para o dataset FIW (Families in the Wild), o sistema utiliza o protocolo RFIW Track-I, que define listas pre-definidas de pares para treino e teste.

**Pares de treino** (`train-pairs.csv`):
- Contem pares positivos (parentes) no formato: `F0001/MID1, F0001/MID3, fd`
- Cada linha identifica dois membros de uma familia e o tipo de relacao (fd = pai-filha, fs = pai-filho, etc.)

**Expansao para nivel de face:**
Cada membro de familia pode ter multiplas fotos. O sistema expande os pares de membros para pares de faces:
```
Par de membros: (F0001/MID1, F0001/MID3)
  MID1 tem fotos: P00001_face1.jpg, P00001_face2.jpg
  MID3 tem fotos: P00003_face0.jpg, P00003_face1.jpg

  Pares gerados (maximo 10 por par de membros):
    (P00001_face1.jpg, P00003_face0.jpg) -> positivo
    (P00001_face1.jpg, P00003_face1.jpg) -> positivo
    (P00001_face2.jpg, P00003_face0.jpg) -> positivo
    (P00001_face2.jpg, P00003_face1.jpg) -> positivo
```

### 2.2 Geracao de Pares Negativos

Para cada par positivo, o sistema gera pares negativos (nao-parentes) utilizando a estrategia `relation_matched`:

1. Identifica a relacao do par positivo (ex: pai-filha)
2. Seleciona uma outra familia aleatoria
3. Cria um par negativo com membros que tenham a mesma relacao, mas de familias diferentes

Isso garante que o modelo nao aprenda atalhos baseados em diferenca de idade ou genero.

**Proporcao configuravel:**
- Treino: 2 negativos para cada positivo (ratio 2:1)
- Validacao/Teste: 1 negativo para cada positivo (ratio 1:1)

### 2.3 Divisao em Conjuntos

Os dados sao divididos em tres conjuntos **disjuntos por familia**:

| Conjunto | Familias | Proposito |
|----------|----------|-----------|
| Treino | 571 familias (85% das familias de treino) | Ajustar os pesos do modelo |
| Validacao | ~100 familias (15% das familias de treino) | Selecionar o melhor threshold e monitorar overfitting |
| Teste | 190 familias (pre-definidas pelo RFIW) | Avaliacao final, nunca usada durante o treinamento |

**Disjuncao por familia** significa que nenhuma pessoa aparece em mais de um conjunto. Isso evita que o modelo "decore" rostos especificos.

### 2.4 Transformacoes de Imagem

**Treino (com data augmentation):**
```python
transforms = [
    Resize(224),              # Redimensiona para 224x224
    ToTensor(),               # Converte para tensor PyTorch [0, 1]
    RandomHorizontalFlip(),   # Espelhamento horizontal (50% de chance)
    ColorJitter(0.2),         # Variacao aleatoria de brilho, contraste, saturacao
    RandomRotation(10),       # Rotacao aleatoria de ate 10 graus
    Normalize(mean, std),     # Normaliza com media/desvio padrao do ImageNet
]
```

**Validacao/Teste (sem augmentation):**
```python
transforms = [
    Resize(224),
    ToTensor(),
    Normalize(mean, std),
]
```

O data augmentation so e aplicado no treino para aumentar artificialmente a diversidade dos dados sem alterar os conjuntos de avaliacao.

### 2.5 Tamanhos Finais dos Conjuntos (FIW)

| Conjunto | Pares Positivos | Pares Negativos | Total |
|----------|----------------|-----------------|-------|
| Treino | ~33.207 | ~66.414 | ~99.621 |
| Validacao | ~5.695 | ~5.695 | ~11.390 |
| Teste | ~6.713 | ~6.713 | ~13.425 |

---

## Passo 3: Construcao do Modelo (`model.py`)

### 3.1 Backbone ViT (Vision Transformer)

O modelo utiliza um Vision Transformer pre-treinado no ImageNet (`vit_base_patch16_224`) como backbone:

```
Imagem 224x224 -> Dividida em patches 16x16 -> 196 patches
Cada patch -> Embedding de 768 dimensoes
+ Token [CLS] -> Representacao global da imagem
```

O ViT pre-treinado ja "entende" imagens gerais. O fine-tuning adapta esse conhecimento para reconhecimento facial de parentesco.

### 3.2 Cross-Attention Bidirecional

A inovacao principal do modelo: permite que cada face "olhe" para a outra para encontrar regioes de similaridade.

```
Face 1 (patches)  ->  Query ─┐
                              ├── Attention(Q1, K2, V2) -> Face 1 atualizada
Face 2 (patches)  ->  Key/Value ─┘

Face 2 (patches)  ->  Query ─┐
                              ├── Attention(Q2, K1, V1) -> Face 2 atualizada
Face 1 (patches)  ->  Key/Value ─┘
```

**Em termos simples:** a Face 1 pergunta "quais regioes da Face 2 sao parecidas comigo?" e vice-versa. Isso e feito em 2 camadas com 8 cabecas de atencao, permitindo que o modelo capture diferentes tipos de similaridade (formato do nariz, olhos, mandibula, etc.).

### 3.3 Channel Attention (Squeeze-and-Excitation)

Apos a cross-attention, um modulo de channel attention pondera a importancia de cada dimensao do embedding:

```python
# Reduz 768 dimensoes a 48 (768/16)
# Depois expande de volta a 768
# Resultado: pesos de 0 a 1 para cada dimensao
weights = sigmoid(Linear(relu(Linear(avg_pool(x), 768/16)), 768))
output = x * weights  # Repondera as dimensoes
```

### 3.4 Projecao Final

Os tokens de ambas as faces sao combinados e projetados para o espaco de embeddings:

```
[CLS token + media dos patches atualizados] -> 768+768 = 1536 dimensoes
  -> Linear(1536, 512)
  -> LayerNorm
  -> GELU
  -> Dropout(0.25)
  -> Linear(512, 512)
  -> Normalizacao L2
```

O resultado e um vetor de 512 dimensoes com norma unitaria para cada face.

---

## Passo 4: Funcao de Perda (`losses.py`)

### 4.1 Perda Contrastiva Supervisionada

A funcao de perda utilizada e a **Supervised Contrastive Loss** com distancia do cosseno:

```
L = (1/2) * [ y * D(a,b)^2  +  (1-y) * max(0, m - D(a,b))^2 ]

onde:
  D(a,b) = 1 - cos(a, b)    (distancia do cosseno, entre 0 e 2)
  y = 1 se sao parentes, 0 se nao sao
  m = margem (0.3 nos melhores runs)
```

**Interpretacao intuitiva:**
- **Pares positivos (parentes):** a loss penaliza quando a distancia e grande. O modelo e incentivado a aproximar os embeddings.
- **Pares negativos (nao-parentes):** a loss penaliza quando a distancia e menor que a margem. O modelo e incentivado a afastar os embeddings ate pelo menos a distancia `m`.

A margem `m = 0.3` define a "zona de seguranca" minima entre pares negativos.

---

## Passo 5: Loop de Treinamento (`trainer.py`)

### 5.1 Otimizador

```python
optimizer = AdamW(
    model.parameters(),
    lr=2e-6,              # Taxa de aprendizado (muito baixa para fine-tuning)
    weight_decay=5e-5,    # Regularizacao L2
    eps=1e-8,
)
```

**AdamW** e uma variante do Adam que aplica weight decay corretamente (desacoplado dos gradientes). O weight decay funciona como regularizacao, penalizando pesos muito grandes para evitar overfitting.

### 5.2 Escalonamento de Learning Rate

O treinamento usa duas fases de controle da taxa de aprendizado:

**Fase 1 — Warmup Linear (epocas 1-8):**
```
LR sobe linearmente de 0 ate 2e-6

Epoca 1: LR = 2e-6 * (1/8) = 2.5e-7
Epoca 2: LR = 2e-6 * (2/8) = 5.0e-7
...
Epoca 8: LR = 2e-6 * (8/8) = 2.0e-6  (pico)
```

O warmup evita que o modelo receba atualizacoes muito grandes no inicio, quando os pesos ainda estao longe do otimo para a tarefa.

**Fase 2 — Cosine Annealing (epocas 9-50):**
```
LR decai seguindo uma curva cossenoidal de 2e-6 ate 1e-7

     LR
      ^
      |    /\
      |   /  \
      |  /    \.
      | /      '._____
      +-------------------> epocas
        warmup   cosine decay
```

O decaimento suave permite que o modelo faca ajustes finos cada vez menores nos pesos.

### 5.3 Mixed Precision (AMP)

O treinamento utiliza **Automatic Mixed Precision** para economizar memoria e acelerar o processamento:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():                      # Forward pass em FP16 (16 bits)
    outputs = model(img1, img2)
    loss = loss_fn(emb1, emb2, labels)

scaler.scale(loss).backward()          # Backward com escala para evitar underflow
scaler.unscale_(optimizer)             # Desfaz escala antes do clipping
clip_grad_norm_(model.parameters(), 1.0)  # Limita gradientes
scaler.step(optimizer)                 # Atualiza pesos
scaler.update()                        # Ajusta fator de escala
```

**Por que usar?**
- Operacoes em FP16 usam metade da memoria da GPU
- GPUs modernas (incluindo AMD) processam FP16 mais rapido que FP32
- O `GradScaler` previne que gradientes muito pequenos se tornem zero em FP16

### 5.4 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

Limita a norma total dos gradientes a 1.0. Isso previne a "explosao de gradientes", que pode ocorrer especialmente nas primeiras camadas do ViT durante fine-tuning.

### 5.5 Fluxo de Uma Epoca

Para cada epoca, o treinamento segue esta sequencia:

```
1. on_epoch_start()        -> Verificar se deve descongelar blocos do ViT
2. Ajustar learning rate   -> Warmup linear ou cosine decay
3. train_epoch()           -> Iterar sobre todos os batches de treino
   Para cada batch:
     a. Carregar img1, img2, labels na GPU
     b. Zero gradientes
     c. Forward pass (com AMP)
     d. Calcular loss
     e. Backward pass (com gradient scaling)
     f. Clip gradientes
     g. Atualizar pesos
4. validate()              -> Avaliar no conjunto de validacao
5. scheduler.step()        -> Atualizar learning rate
6. Verificar best model    -> Salvar se AUC melhorou
7. Verificar early stop    -> Parar se paciencia esgotou
8. Verificar safeguards    -> Parar se treinamento degenerou
```

### 5.6 Validacao com Threshold Otimo

A validacao **nao usa threshold fixo**. A cada epoca:

```python
def validate():
    # 1. Coleta predicoes (scores de similaridade) para todos os pares de validacao
    bundle = collect_predictions(model, val_loader, device)

    # 2. Busca o threshold que maximiza o F1-score na validacao
    threshold, _ = find_optimal_threshold(
        bundle["predictions"], bundle["labels"], metric="f1"
    )
    # Busca em grade: testa thresholds de 0.10 a 0.95, passo 0.05

    # 3. Calcula metricas usando o threshold otimo
    metrics = compute_metrics_from_predictions(
        predictions=bundle["predictions"],
        labels=bundle["labels"],
        threshold=threshold,
    )
    return metrics
```

**Por que buscar o threshold?** O modelo produz scores de similaridade entre 0 e 1. Precisamos de um ponto de corte para decidir "parente" vs "nao-parente". Em vez de usar um valor arbitrario como 0.5, buscamos o valor que maximiza o F1-score no conjunto de validacao.

### 5.7 Salvamento de Checkpoints

O melhor modelo e salvo quando a metrica monitorada (ROC AUC) melhora:

```python
if current_auc > best_auc + min_delta:    # min_delta = 0.0001
    best_auc = current_auc
    patience_counter = 0
    save_checkpoint("best.pt", val_metrics, epoch)
else:
    patience_counter += 1
```

O checkpoint contem:
- `model_state_dict`: pesos do modelo
- `optimizer_state_dict`: estado do otimizador (momentos do Adam)
- `scheduler_state_dict`: estado do escalonador de LR
- `scaler_state_dict`: estado do gradient scaler (AMP)
- `metrics`: metricas de validacao na hora do save
- `epoch`: numero da epoca
- `best_metric`: melhor AUC ate o momento
- `history`: historico completo de loss e metricas

### 5.8 Early Stopping

Se a AUC nao melhora por `patience` epocas consecutivas, o treinamento para:

```python
if patience_counter >= patience:     # patience = 25
    print(f"Early stopping at epoch {epoch}")
    break
```

Isso evita desperdicio de tempo de GPU quando o modelo ja convergiu.

### 5.9 Safeguards (Protecoes Automaticas)

Duas protecoes adicionais contra treinamento degenerado:

**Safeguard 1 — Declinio sustentado de AUC:**
```python
# Se AUC esta abaixo do pico por 10 epocas consecutivas, para
if all(v < peak_auc - 0.005 for v in last_10_aucs):
    print("Stopping: AUC declining for 10 epochs")
```

**Safeguard 2 — Acuracia degenerada:**
```python
# Se acuracia = 50% por 8 epocas consecutivas, o threshold esta quebrado
if all(abs(a - 0.5) < 0.0001 for a in last_8_accuracies):
    print("Stopping: degenerate accuracy")
```

Estes safeguards foram adicionados apos observar que alguns runs ficavam "presos" em estados degenerados, desperdicando horas de GPU.

---

## Passo 6: Avaliacao Pos-Treinamento (`train.py` — secao final)

Apos o loop de treinamento, o script carrega o melhor checkpoint e realiza a **avaliacao com protocolo padrao**:

### 6.1 Protocolo de Avaliacao

```python
# 1. Carrega o melhor modelo salvo durante o treinamento
load_best_checkpoint(model, checkpoint_dir, device)

# 2. Avalia seguindo o protocolo padrao da literatura
protocol_results = evaluate_with_validation_threshold(
    model, val_loader, test_loader, device
)
```

O protocolo funciona em 5 etapas:

```
1. Coleta predicoes no conjunto de VALIDACAO
2. Busca o threshold otimo no conjunto de validacao (maximiza F1)
3. Calcula metricas de validacao com esse threshold
4. Coleta predicoes no conjunto de TESTE
5. Aplica o MESMO threshold no teste (sem reotimizar!)
```

**Ponto critico:** o threshold e determinado na validacao e aplicado intacto no teste. Reotimizar o threshold no teste seria "vazamento de dados" (data leakage), pois estariamos usando informacao do teste para calibrar o modelo.

### 6.2 Metricas Calculadas

| Metrica | Descricao |
|---------|-----------|
| **Accuracy** | Proporcao de predicoes corretas |
| **Balanced Accuracy** | Media da acuracia por classe (evita vies em dados desbalanceados) |
| **Precision** | Dos que o modelo classificou como parentes, quantos realmente sao? |
| **Recall** | Dos que realmente sao parentes, quantos o modelo detectou? |
| **F1-Score** | Media harmonica de precision e recall |
| **ROC AUC** | Area sob a curva ROC — mede a capacidade discriminativa |
| **TAR@FAR** | Taxa de aceitacao verdadeira em taxas de falsa aceitacao fixas |

### 6.3 Salvamento de Resultados

```python
# Salva metadados no checkpoint para reproducibilidade
update_checkpoint_metadata(checkpoint_path, protocol_metadata)

# Salva resultados em JSON
save_json("protocol_summary.json", {
    "validation_metrics": val_metrics,
    "test_metrics": test_metrics,
    "threshold": threshold,
    "train_dataset": "fiw",
    "split_seed": 42,
    ...
})

# Salva resultados em texto legivel
with open("test_results_rocm.txt", "w") as f:
    for key, value in test_metrics.items():
        f.write(f"{key}: {value:.4f}\n")
```

---

## Passo 7: Teste Independente (`test.py`)

O script `test.py` e o segundo estagio do pipeline. Ele carrega o melhor checkpoint e avalia no conjunto de teste de forma independente.

### 7.1 Reconstrucao do Modelo

O modelo e reconstruido a partir das configuracoes salvas no checkpoint:

```python
checkpoint = torch.load("best.pt")
model_config = checkpoint["model_config"]

model = build_vit_facor_model(
    vit_model=model_config["vit_model"],           # "vit_base_patch16_224"
    embedding_dim=model_config["embedding_dim"],     # 512
    num_cross_attn_layers=model_config["cross_attn_layers"],  # 2
    cross_attn_heads=model_config["cross_attn_heads"],        # 8
    dropout=model_config["dropout"],                 # 0.25
)
model.load_state_dict(checkpoint["model_state_dict"])
```

### 7.2 Threshold do Checkpoint

O threshold utilizado e o que foi determinado na validacao e armazenado no checkpoint:

```python
threshold = get_checkpoint_threshold(checkpoint, default=0.5)
```

Isso garante consistencia — o mesmo threshold usado no final do treinamento e usado aqui.

### 7.3 Visualizacao de Attention Maps

O test.py pode gerar visualizacoes dos mapas de cross-attention:

```
[Face 1] [Face 2] [Mapa de Atencao] [Overlay na Face 1]
```

Os mapas mostram quais regioes da Face 2 o modelo considerou mais importantes ao comparar com a Face 1. Em pares de parentes, tipicamente se observam padroes mais fortes nas regioes de olhos, nariz e mandibula.

---

## Passo 8: Avaliacao Completa (`evaluate.py`)

O script `evaluate.py` e o terceiro e ultimo estagio. Gera analises visuais detalhadas.

### 8.1 Curva ROC

Plota a curva ROC (Receiver Operating Characteristic):
- Eixo X: Taxa de Falsos Positivos (FPR)
- Eixo Y: Taxa de Verdadeiros Positivos (TPR)
- Diagonal: classificador aleatorio

A area sob a curva (AUC) indica a capacidade discriminativa do modelo. AUC = 1.0 seria perfeito; AUC = 0.5 seria aleatorio.

### 8.2 Matriz de Confusao

Gera a matriz de confusao mostrando:
```
                    Predito
                  Kin    Non-Kin
Real  Kin     [  VP  |   FN  ]
      Non-Kin [  FP  |   VN  ]
```

Onde VP = Verdadeiros Positivos, FN = Falsos Negativos, etc.

### 8.3 Analise de Padroes de Atencao

Compara a intensidade media de atencao entre pares de parentes e nao-parentes:

```python
for cada par no teste:
    if parentes:
        kin_attentions.append(media_da_atencao)
    else:
        nonkin_attentions.append(media_da_atencao)

# Gera grafico comparativo
bar(["Kin Pairs", "Non-Kin Pairs"], [media_kin, media_nonkin])
```

Se o modelo aprendeu corretamente, espera-se que pares de parentes tenham intensidade de atencao mais alta (o modelo "presta mais atencao" porque encontra mais similaridades).

---

## Passo 9: Validacao Cruzada K-Fold (`train_cv.py`)

O pipeline utiliza **validacao cruzada de 5 folds** para a avaliacao final dos modelos. Esse modo e ativado pela variavel `CV_FOLDS=5` e substitui os tres scripts (train/test/evaluate) por um unico script (`train_cv.py`) que executa o treinamento completo K vezes.

### 9.1 Por que Usar Validacao Cruzada?

Em um unico split treino/teste, os resultados dependem de **quais amostras caem em cada conjunto**. Uma divisao pode ser "facil" (pares mais distinguiveis no teste) e outra "dificil". A validacao cruzada resolve esse problema:

1. Divide os dados em **K partes iguais** (folds)
2. Para cada fold: treina em K-1 partes, testa na parte restante
3. Repete K vezes, rotacionando qual parte e o teste
4. Reporta **media +- desvio padrao** de todas as metricas

```
Fold 1: [TESTE] [treino] [treino] [treino] [treino]
Fold 2: [treino] [TESTE] [treino] [treino] [treino]
Fold 3: [treino] [treino] [TESTE] [treino] [treino]
Fold 4: [treino] [treino] [treino] [TESTE] [treino]
Fold 5: [treino] [treino] [treino] [treino] [TESTE]
```

O resultado final (ex: "AUC = 0.842 +- 0.012") indica nao apenas o desempenho medio, mas tambem a **estabilidade** do modelo — um desvio padrao baixo significa que o modelo generaliza bem independentemente dos dados especificos usados para treino.

### 9.2 Splits Disjuntos

A divisao em folds respeita a estrutura dos dados para evitar vazamento de informacao:

**KinFaceW — Disjuncao por par:**
Nenhum par de imagens aparece em mais de um fold. Cada par (pai-filho, mae-filha, etc.) e atribuido a exatamente um fold.

**FIW — Disjuncao por familia:**
Nenhuma familia aparece em mais de um fold. Todas as imagens e pares de uma familia vao para o mesmo fold. Isso e mais rigoroso, pois evita que o modelo "decore" rostos familiares vistos durante o treino.

```python
# Atribuicao deterministica de folds
all_ids = sorted(family_ids)   # Ex: ["F0001", "F0002", ..., "F0761"]
random.shuffle(all_ids)        # Shuffle deterministico (seed fixa)

# Familia i vai para o fold (i % K)
test_ids = {id for i, id in enumerate(all_ids) if i % n_folds == fold_k}
```

### 9.3 Pipeline de Cada Fold

Cada fold executa o pipeline completo de treinamento de forma independente:

```python
def run_fold(fold_k, n_folds, args):
    # 1. Cria dataloaders especificos para este fold
    train_loader, val_loader, test_loader = create_cv_fold_loaders(
        config, fold_k=fold_k, n_folds=n_folds
    )

    # 2. Cria um modelo NOVO (pesos reinicializados)
    model = build_vit_facor_model(...)

    # 3. Treina o modelo (mesmo loop descrito nos Passos 4-5)
    trainer = ViTFaCoRROCmTrainer(model, loss_fn, train_loader, val_loader, ...)
    trainer.train()

    # 4. Carrega o melhor checkpoint deste fold
    load_best_checkpoint(model, checkpoint_dir)

    # 5. Avalia com protocolo padrao (threshold da validacao no teste)
    results = evaluate_with_validation_threshold(
        model, val_loader, test_loader, device
    )

    return results["test_metrics"]
```

**Ponto importante:** cada fold treina um modelo completamente novo. Os pesos nao sao compartilhados entre folds. Isso garante que a avaliacao de cada fold e independente.

### 9.4 Agregacao de Resultados

Apos os K folds, o sistema calcula a media e o desvio padrao de cada metrica:

```python
def main():
    all_fold_results = []

    for fold_k in range(n_folds):       # 0, 1, 2, 3, 4
        fold_metrics = run_fold(fold_k, n_folds, args)
        all_fold_results.append(fold_metrics)

    # Calcula media e desvio padrao
    summary = aggregate_numeric_metrics(all_fold_results)

    # Salva resultado final
    save_json("cv_summary.json", {
        "n_folds": 5,
        "mean_accuracy": 0.7356,    # Media dos 5 folds
        "std_accuracy": 0.0584,     # Desvio padrao
        "mean_f1": 0.7707,
        "std_f1": 0.0283,
        "mean_roc_auc": 0.8420,
        "std_roc_auc": 0.0123,
        "fold_results": [...]       # Resultado individual de cada fold
    })
```

### 9.5 Exemplo de Resultado (5-Fold CV)

```
Fold 1: Acc=62.2%  F1=0.722  AUC=0.824
Fold 2: Acc=75.2%  F1=0.771  AUC=0.831
Fold 3: Acc=75.7%  F1=0.783  AUC=0.856
Fold 4: Acc=78.8%  F1=0.809  AUC=0.849
Fold 5: Acc=75.9%  F1=0.769  AUC=0.850
─────────────────────────────────────────
Media:  Acc=73.6%  F1=0.771  AUC=0.842
Desvio:     +-5.8%     +-0.028   +-0.012
```

**Interpretacao:**
- **AUC medio de 0.842** indica boa capacidade discriminativa
- **Desvio padrao de 0.012 na AUC** indica estabilidade entre folds
- **Fold 1** apresenta resultado inferior — provavelmente contem pares mais dificeis de distinguir

### 9.6 Estrutura de Saida (Modo CV)

```
output/026/
  checkpoints/
    fold_0/best.pt       <- Melhor modelo do fold 0
    fold_1/best.pt       <- Melhor modelo do fold 1
    fold_2/best.pt       <- ...
    fold_3/best.pt
    fold_4/best.pt
  results/
    cv_summary.json      <- Media, desvio padrao, resultados por fold
  logs/
    train.log            <- Log completo de todos os folds
```

### 9.7 Execucao

```bash
cd models/02_vit_facor_crossattn

CV_FOLDS=5 \
EPOCHS=50 \
LOSS=contrastive \
MARGIN=0.3 \
LEARNING_RATE=1e-5 \
SCHEDULER=cosine \
bash AMD/run_pipeline.sh
```

Tempo estimado total (5 folds x ~50 epocas): depende do dataset e da GPU.

---

## Passo 10: Estrutura Final de Saida

Ao final da execucao completa, a pasta de saida contem:

```
output/003/
  checkpoints/
    best.pt                      <- Melhor modelo (maior AUC na validacao)
    final.pt                     <- Modelo na ultima epoca
    epoch_5.pt, epoch_10.pt, ... <- Checkpoints periodicos (a cada 5 epocas)
  results/
    test_results_rocm.txt        <- Metricas finais em texto
    protocol_summary.json        <- Metricas completas + metadados em JSON
    test_metrics_rocm.txt        <- Metricas do test.py
    metrics_rocm.json            <- Metricas do evaluate.py
    roc_curve_rocm.png           <- Grafico da curva ROC
    confusion_matrix_rocm.png   <- Grafico da matriz de confusao
    attention_maps/              <- Visualizacoes de cross-attention
    attention_intensity_comparison.png  <- Comparacao kin vs non-kin
  logs/
    train.log                    <- Log completo do treinamento
    test.log                     <- Log do teste
    evaluate.log                 <- Log da avaliacao
```

---

## Resumo do Fluxo Completo

### Modo Single-Run

```
[1] run_pipeline.sh
    |-- Configura GPU AMD (ROCm)
    |-- Cria ambiente virtual Python
    |-- Cria pasta de saida numerada
    |
    [2] train.py
    |   |-- Carrega dataset FIW (RFIW Track-I)
    |   |   |-- Expande pares de membros para pares de faces
    |   |   |-- Gera negativos relation_matched (2:1)
    |   |   |-- Divide em treino/validacao (disjunto por familia)
    |   |
    |   |-- Constroi modelo ViT + Cross-Attention
    |   |   |-- ViT pre-treinado (backbone)
    |   |   |-- Cross-attention bidirecional (2 camadas, 8 cabecas)
    |   |   |-- Channel attention + projecao para 512 dimensoes
    |   |
    |   |-- Loop de treinamento
    |   |   |-- Warmup linear (8 epocas)
    |   |   |-- Cosine annealing (epocas 9-50)
    |   |   |-- Mixed precision (FP16) + gradient clipping
    |   |   |-- Validacao com busca de threshold otimo
    |   |   |-- Salvamento do melhor modelo (por AUC)
    |   |   |-- Early stopping (paciencia = 25)
    |   |   |-- Safeguards contra treinamento degenerado
    |   |
    |   |-- Avaliacao final com protocolo padrao
    |       |-- Threshold da validacao aplicado no teste
    |       |-- Metricas: Accuracy, F1, AUC, TAR@FAR
    |
    [3] test.py
    |   |-- Carrega melhor checkpoint
    |   |-- Avalia no teste com threshold salvo
    |   |-- (Opcional) Gera mapas de atencao
    |
    [4] evaluate.py
        |-- Carrega melhor checkpoint
        |-- Gera curva ROC
        |-- Gera matriz de confusao
        |-- Analisa padroes de atencao (kin vs non-kin)
```

### Modo Validacao Cruzada (5-Fold)

```
[1] run_pipeline.sh (CV_FOLDS=5)
    |-- Configura GPU AMD (ROCm)
    |-- Cria ambiente virtual Python
    |-- Cria pasta de saida numerada
    |
    [2] train_cv.py
        |-- Divide dados em 5 folds (disjuntos por familia)
        |
        |-- Fold 1/5:
        |   |-- Cria modelo novo (pesos pre-treinados)
        |   |-- Treina no treino do fold 1
        |   |-- Avalia no teste do fold 1 (protocolo padrao)
        |   |-- Salva metricas do fold 1
        |
        |-- Fold 2/5: (mesmo processo, dados diferentes)
        |-- Fold 3/5: ...
        |-- Fold 4/5: ...
        |-- Fold 5/5: ...
        |
        |-- Agrega resultados
            |-- Calcula media +- desvio padrao de cada metrica
            |-- Salva cv_summary.json
```

---

## Exemplos de Execucao

### Single-Run (experimentacao)

```bash
cd models/02_vit_facor_crossattn

SKIP_INSTALL=1 \
TRAIN_DATASET=fiw \
EPOCHS=50 \
PATIENCE=25 \
LOSS=contrastive \
MARGIN=0.3 \
TEMPERATURE=0.3 \
NEGATIVE_RATIO=2.0 \
TRAIN_NEGATIVE_STRATEGY=relation_matched \
EVAL_NEGATIVE_RATIO=1.0 \
EVAL_NEGATIVE_STRATEGY=random \
LEARNING_RATE=2e-6 \
WEIGHT_DECAY=5e-5 \
SCHEDULER=cosine \
WARMUP_EPOCHS=8 \
DROPOUT=0.25 \
bash AMD/run_pipeline.sh
```

Tempo estimado por epoca (RX 6750 XT, batch_size=32): ~36 minutos.

### 5-Fold Cross-Validation (avaliacao final)

```bash
cd models/02_vit_facor_crossattn

SKIP_INSTALL=1 \
CV_FOLDS=5 \
TRAIN_DATASET=fiw \
EPOCHS=50 \
PATIENCE=25 \
LOSS=contrastive \
MARGIN=0.3 \
LEARNING_RATE=2e-6 \
WEIGHT_DECAY=5e-5 \
SCHEDULER=cosine \
WARMUP_EPOCHS=8 \
DROPOUT=0.25 \
bash AMD/run_pipeline.sh
```

Tempo estimado total: 5 folds x ~50 epocas x ~36 min/epoca. Com early stopping, tipicamente completa em menos tempo.

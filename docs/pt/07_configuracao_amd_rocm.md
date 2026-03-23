# Configuracao AMD ROCm

## Visao Geral

O projeto utiliza GPU AMD com a plataforma **ROCm (Radeon Open Compute)**, que e a alternativa da AMD ao CUDA da NVIDIA. O PyTorch suporta ROCm via backend HIP.

### Hardware Utilizado

| Componente | Especificacao |
|-----------|---------------|
| GPU | AMD Radeon RX 6750 XT |
| Arquitetura | RDNA2 (gfx1031) |
| VRAM | 12 GB GDDR6 |
| ROCm | 5.7 |
| PyTorch | 2.3.1+rocm5.7 |

---

## Configuracao do Ambiente

### Variavel de Ambiente Essencial

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

A RX 6750 XT (gfx1031) nao e oficialmente suportada pelo ROCm. Esta variavel faz o runtime tratar a GPU como gfx1030 (RX 6900 XT), que e suportada e compativel.

### Outras Variaveis (definidas no run_pipeline.sh)

```bash
export HIP_VISIBLE_DEVICES=0         # Seleciona GPU 0
export ROCR_VISIBLE_DEVICES=0        # Mesmo, nivel driver
export MIOPEN_FIND_MODE=FAST         # Busca rapida de algoritmos convolucionais
```

---

## Otimizacoes para ROCm

### Mixed Precision (AMP)

```python
from torch.cuda.amp import autocast, GradScaler

# ROCm usa o mesmo API do CUDA
scaler = GradScaler()
with autocast():
    outputs = model(img1, img2)
    loss = loss_fn(outputs, labels)
```

**Nota:** O aviso "Torch was not compiled with memory efficient attention" e normal em ROCm — o scaled_dot_product_attention usa uma implementacao alternativa.

### Limpeza de Cache

```python
# Libera VRAM nao utilizada periodicamente
torch.cuda.empty_cache()
```

Feita a cada 10 epocas para evitar fragmentacao de memoria.

---

## Grupo de Permissoes (render)

O run_pipeline.sh reinicia automaticamente com o grupo `render` para acesso a GPU:

```bash
if ! id -nG | grep -qw render; then
    exec sg render -c "bash '$0'"
fi
```

Isso e necessario porque o acesso ao dispositivo `/dev/kfd` (Kernel Fusion Driver) requer permissao do grupo `render`.

---

## Tempos de Treinamento

| Dataset | Pares/Epoca | Tempo/Epoca | Batch Size |
|---------|------------|-------------|------------|
| KinFaceW (1 fold) | ~2.000 | ~20s | 32 |
| FIW (single run) | ~99.000 | ~36min | 32 |

A RX 6750 XT processa ~3.5 batches/segundo para o modelo ViT-Base + Cross-Attention.

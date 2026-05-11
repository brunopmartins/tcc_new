# AdaFace IR-101 weights

This directory holds the pretrained AdaFace IR-101 weights. **Not committed to
git** (~250 MB).

## Primary source — HuggingFace mirror of cvlface AdaFace

The AdaFace authors (Kim et al., CVPR 2022) distribute weights through the
[mk-minchul/AdaFace](https://github.com/mk-minchul/AdaFace) repo, which uses
Google Drive. The same author also uploaded HuggingFace mirrors under the
`minchul/cvlface_adaface_*` namespace.

We use **`minchul/cvlface_adaface_ir101_webface4m`** by default — IR-101
trained with AdaFace on WebFace4M (4M faces, more diverse than MS1M).

### Download

```bash
pip install huggingface_hub

# Set your HF token if the repo requires it (most do not for this one)
# export HF_TOKEN=<your_token>

huggingface-cli download \
    minchul/cvlface_adaface_ir101_webface4m \
    pretrained_model/model.pt \
    --local-dir models/10_adaface_facor/weights/_hf_cache

mv models/10_adaface_facor/weights/_hf_cache/pretrained_model/model.pt \
   models/10_adaface_facor/weights/adaface_ir101_webface4m.pth

rm -rf models/10_adaface_facor/weights/_hf_cache
```

Equivalent Python:

```python
from huggingface_hub import hf_hub_download
import shutil

p = hf_hub_download(
    repo_id="minchul/cvlface_adaface_ir101_webface4m",
    filename="pretrained_model/model.pt",
    local_dir="models/10_adaface_facor/weights/_hf_cache",
)
shutil.move(p, "models/10_adaface_facor/weights/adaface_ir101_webface4m.pth")
```

## Alternative AdaFace IR-101 mirrors

| HF repo                                          | Pretrain set     | Notes                                |
|--------------------------------------------------|------------------|--------------------------------------|
| `minchul/cvlface_adaface_ir101_webface4m`        | WebFace4M (4M)   | **Default for M10**                  |
| `minchul/cvlface_adaface_ir101_webface12m`       | WebFace12M (12M) | Larger pretrain — may be even better |
| `minchul/cvlface_adaface_ir101_ms1mv2`           | MS1MV2 (5.8M)    | Same pretrain set as M08's ArcFace   |
| `minchul/cvlface_adaface_ir101_ms1mv3`           | MS1MV3 (5.2M)    | Cleaned MS1M variant                  |

All four share the same architecture (the cvlface `Backbone` IR-101 class) and
load identically through `adaface_iresnet.load_adaface_state_dict`. Swap by
setting `ADAFACE_WEIGHTS=...` when calling `run_pipeline.sh`.

## Fallback — reuse ArcFace MS1MV3 weights

If AdaFace weights are unreachable, the **same module file** (model.py) loader
infrastructure could be repurposed to load InsightFace's ArcFace IR-100
weights, but the architectures are different:

- AdaFace cvlface uses `BasicBlockIR` (MaxPool shortcut, PReLU mid-block) with
  state-dict keys `net.input_layer.*`, `net.body.0..48.res_layer.*`,
  `net.output_layer.*`.
- InsightFace iresnet100 (used by M08) has different block names and lacks
  the cvlface wrapper prefix.

**ArcFace weights cannot be dropped into AdaFaceIR101 without an architecture
translation pass.** If AdaFace is truly unavailable, the simpler fallback is
to subclass M08's `iresnet100` and modify *its* forward to expose `layer4`
output — but that diverges from M10's design.

For this project the AdaFace WebFace4M weights are confirmed accessible
(verified via `HfApi.list_models` with the provided HF token), so the
fallback path is not exercised. See `models/10_adaface_facor/RUN_LOG.md`
(when training runs are performed) for the actual loading log.

## Verifying the download

```bash
cd models/10_adaface_facor
python -c "
import sys; sys.path.insert(0, '.')
from adaface_iresnet import adaface_ir101
m = adaface_ir101('weights/adaface_ir101_webface4m.pth')
import torch
x = torch.randn(2, 3, 112, 112)
emb = m(x)
print('OK — emb shape', tuple(emb.shape), '(expected (2, 512))')
"
```

If you see `OK — emb shape (2, 512)` with **no `missing` or `unexpected`
keys** printed, the weights are loaded cleanly.

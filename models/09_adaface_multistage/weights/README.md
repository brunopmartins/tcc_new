# Weights — Model 09

M09 uses the **same AdaFace IR-101 WebFace4M checkpoint as M10**. To avoid
duplicating a 249 MB file, symlink the M10 weight into place:

```bash
ln -sf ../../10_adaface_facor/weights/adaface_ir101_webface4m.pth \
       models/09_adaface_multistage/weights/adaface_ir101_webface4m.pth
```

If M10's weights are missing, fetch them first:

```bash
huggingface-cli download minchul/cvlface_adaface_ir101_webface4m \
    pretrained_model/model.pt \
    --local-dir models/10_adaface_facor/weights/_hf_cache
mv models/10_adaface_facor/weights/_hf_cache/pretrained_model/model.pt \
   models/10_adaface_facor/weights/adaface_ir101_webface4m.pth
```

The cvlface checkpoint wraps the network as `Wrapper.net`, so all keys are
prefixed `net.`. `adaface_iresnet.load_adaface_state_dict` strips that
automatically on load — no preprocessing needed.

Source: <https://huggingface.co/minchul/cvlface_adaface_ir101_webface4m>
(MIT license, Minchul Kim).

# M11 weights

M11 reuses M10's AdaFace IR-101 (WebFace4M) checkpoint via symlink:

```
adaface_ir101_webface4m.pth -> ../../10_adaface_facor/weights/adaface_ir101_webface4m.pth
```

If you do not have M10's weights set up, download from the AdaFace official
release (or the project's curated mirror) and place at
`models/10_adaface_facor/weights/adaface_ir101_webface4m.pth`.

SHA256 of the expected file is recorded in M10's `weights/README.md`.

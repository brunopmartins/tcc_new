# Run Log — Model 02: ViT + FaCoR Cross-Attention

This file logs every training run for this model with the exact configuration and final metrics. The goal is reproducibility: anyone (human or agent) reading this should be able to recreate any run by copying its launch command.

Newest run on top.

---

## Historical runs (026-032) — pre-RUN_LOG

These runs predate the RUN_LOG.md system. Their results are summarized in [docs/pt/09_visao_geral_pesquisa.md](../../docs/pt/09_visao_geral_pesquisa.md) under "Evolucao dos Resultados (Modelo 02 — FIW)". Configuration details and per-epoch trajectories are recoverable from `output/0XX/logs/train.log`.

Best historical run: **Run 031** — Test ROC AUC = **0.850**, Test Acc = 74.4%.
- Config: no freeze, LR=5e-6, warmup=5, dropout=0.2
- This is the canonical Model 02 result used in all comparison tables.

For new runs, follow the template in [prompts/train_model.md](../../prompts/train_model.md) (local) and append entries above this section.

---

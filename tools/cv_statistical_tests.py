#!/usr/bin/env python3
"""Paired statistical tests over 5-fold CV results.

For each pair of models, runs a paired t-test (parametric) and a
Wilcoxon signed-rank test (non-parametric) on the per-fold metric
vectors. Sample size is n=5 (low statistical power — see caveats in
the report).

Models covered (only those with 5 family-disjoint folds):
- M02 R031 (output/033)
- M06 R001 (output/003)
- M12 R006 (output/011)
- M12 R010 (output/012)
- M12 R011 (output/014)
- B0 (deterministic; all 5 folds identical)
- M05 R005 (output/010, partial — only included if all 5 folds present)

Output: markdown report at the path given via --out (default
cv_statistical.md at the project root).
"""
from __future__ import annotations

import argparse
import json
import re
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


PROJECT_ROOT = Path("/home/bruno/Desktop/tcc_new")

# (model_name, root_dir_with_fold_subdirs, results_filename)
CV_RUNS = [
    ("M02 R031",  PROJECT_ROOT / "models/02_vit_facor_crossattn/output/033",     "metrics_rocm.json"),
    ("M06 R001",  PROJECT_ROOT / "models/06_retrieval_augmented_kinship/output/003", "test_metrics_rocm.json"),
    ("M12 R006",  PROJECT_ROOT / "models/12_rgck_net/output/011",                  "test_metrics_rocm.txt"),
    ("M12 R010",  PROJECT_ROOT / "models/12_rgck_net/output/012",                  "test_metrics_rocm.txt"),
    ("M12 R011",  PROJECT_ROOT / "models/12_rgck_net/output/014",                  "test_metrics_rocm.txt"),
    ("M05 R005",  PROJECT_ROOT / "models/05_dinov2_lora_diffattn/output/010",       "test_metrics_rocm.json"),
]

# Deterministic B0 (5 folds, all identical — model frozen, test set fixed).
# Numbers taken from models/00_adaface_cosine/output/cv/cv_metrics.txt.
B0_VALUES = {
    "roc_auc": 0.7991,
    "average_precision": 0.8093,
    "tar@far=0.001": 0.0706,
    "tar@far=0.01": 0.2178,
    "tar@far=0.1": 0.5238,
    "f1": 0.7121,
    "accuracy": 0.6660,
    "balanced_accuracy": 0.6739,
    "precision": 0.6065,
    "recall": 0.8623,
}


METRICS = [
    "roc_auc", "average_precision",
    "tar@far=0.001", "tar@far=0.01", "tar@far=0.1",
    "f1", "accuracy", "balanced_accuracy",
    "precision", "recall",
]


def parse_metrics_txt(path: Path) -> Dict[str, float]:
    """Parse the M12 'key: value' text format."""
    out: Dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\s*([\w@./=]+)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def load_fold_metrics(fold_dir: Path, results_filename: str) -> Optional[Dict[str, float]]:
    """Return the metric dict for one fold, or None if missing.

    Tries the primary results_filename first, then falls back to the other
    known names so that runs whose folds were written by different scripts
    still load cleanly.
    """
    candidates = [results_filename, "metrics_rocm.json",
                  "test_metrics_rocm.json", "test_metrics_rocm.txt"]
    # de-dup preserving order
    seen = set()
    for fname in [c for c in candidates if not (c in seen or seen.add(c))]:
        p = fold_dir / "results" / fname
        if not p.exists():
            continue
        if p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))
        return parse_metrics_txt(p)
    return None


def load_cv_metrics(name: str, root: Path, fname: str, n_folds: int = 5) -> Optional[Dict[str, List[float]]]:
    """Return {metric: [val_per_fold]} or None if not all folds available."""
    per_fold = []
    for k in range(n_folds):
        fold_dir = root / f"fold_{k}"
        m = load_fold_metrics(fold_dir, fname)
        if m is None:
            return None
        per_fold.append(m)

    out: Dict[str, List[float]] = {}
    for metric in METRICS:
        vals = [pf.get(metric) for pf in per_fold]
        if all(isinstance(v, (int, float)) for v in vals):
            out[metric] = [float(v) for v in vals]
    return out


def fmt_mean_std(vals: List[float]) -> str:
    arr = np.asarray(vals, dtype=float)
    return f"{arr.mean():.4f} ± {arr.std(ddof=1):.4f}"


def paired_tests(a: List[float], b: List[float]) -> Dict[str, float]:
    """Run paired t-test + Wilcoxon signed-rank on two equal-length vectors."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)
    diff = a_arr - b_arr

    # paired t-test
    try:
        t_res = stats.ttest_rel(a_arr, b_arr)
        t_stat, t_p = float(t_res.statistic), float(t_res.pvalue)
    except Exception:
        t_stat, t_p = float("nan"), float("nan")
    if np.isnan(t_stat) and np.all(diff == 0):
        # both sides identical (e.g., B0 vs itself) → undefined t
        t_stat, t_p = 0.0, 1.0

    # Wilcoxon signed-rank
    try:
        if np.allclose(diff, 0):
            w_stat, w_p = 0.0, 1.0
        else:
            w_res = stats.wilcoxon(a_arr, b_arr, zero_method="wilcox")
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
    except Exception:
        w_stat, w_p = float("nan"), float("nan")

    return {
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)) if len(diff) > 1 else 0.0,
        "t_stat": t_stat,
        "t_pvalue": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_pvalue": w_p,
        "n": len(a_arr),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "cv_statistical.md")
    p.add_argument("--metrics", nargs="+", default=METRICS,
                   help="Metrics to report.")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading per-fold CV metrics...")
    cv_data: Dict[str, Dict[str, List[float]]] = {}

    # Trained models
    for name, root, fname in CV_RUNS:
        m = load_cv_metrics(name, root, fname)
        if m is None:
            print(f"  SKIP {name}: not all 5 folds present in {root}")
            continue
        cv_data[name] = m
        print(f"  loaded {name}: {len(m)} metrics × 5 folds")

    # B0 deterministic: replicate value 5 times for each metric
    cv_data["B0"] = {k: [v] * 5 for k, v in B0_VALUES.items()}
    print(f"  loaded B0 (deterministic, n=5 replicated)")

    model_names = list(cv_data.keys())
    print(f"\nModels in analysis: {model_names}")

    # ------------------------------------------------------------------ #
    # Build the markdown report
    # ------------------------------------------------------------------ #
    lines: List[str] = []
    L = lines.append

    L("# Testes estatísticos pareados sobre CV de 5 dobras")
    L("")
    L("Resultados de **t-test pareado** (paramétrico) e **Wilcoxon signed-rank** "
      "(não paramétrico) entre as médias por dobra dos modelos avaliados sob "
      "validação cruzada de 5 dobras família-disjunta.")
    L("")

    # CV mean ± std per model
    L("## Médias por modelo (CV de 5 dobras)")
    L("")
    head = "| Modelo | " + " | ".join(args.metrics) + " |"
    sep  = "|" + ("---|" * (len(args.metrics) + 1))
    L(head)
    L(sep)
    for name in model_names:
        row = [name]
        for metric in args.metrics:
            if metric in cv_data[name]:
                row.append(fmt_mean_std(cv_data[name][metric]))
            else:
                row.append("—")
        L("| " + " | ".join(row) + " |")
    L("")

    # Caveats
    L("## Caveats metodológicos")
    L("")
    L("- **n = 5** dobras pareadas. Para o teste-t pareado isso dá df = 4 e "
      "poder estatístico baixo: diferenças reais menores que ~1 desvio por "
      "dobra (≈ 0,004 AUC no M12) tendem a não atingir significância "
      "mesmo quando consistentes.")
    L("- Wilcoxon signed-rank com n=5 tem apenas 2⁵ = 32 permutações possíveis "
      "(menor p-value alcançável ≈ 0,031 quando todas as diferenças têm o "
      "mesmo sinal). p-values exatos abaixo disso são impossíveis com esse "
      "tamanho de amostra.")
    L("- O B0 é **determinístico** (modelo congelado + conjunto de teste fixo). "
      "Suas 5 réplicas têm desvio zero, então o t-test pareado contra o B0 "
      "vira essencialmente um teste de uma amostra sobre a diferença A − B0 "
      "(ainda interpretável).")
    L("- Os modelos não foram repetidos com sementes diferentes. A leitura é, "
      "portanto, sobre **partição de famílias**, não sobre estabilidade ao "
      "longo de inicializações.")
    L("")

    # Pairwise tests per metric
    L("## Comparações pareadas")
    L("")
    n_pairs = len(model_names) * (len(model_names) - 1) // 2
    L(f"Total de {n_pairs} pares × {len(args.metrics)} métricas = "
      f"{n_pairs * len(args.metrics)} testes. p < 0,05 marcado com **negrito**, "
      "p < 0,10 marcado com *itálico*.")
    L("")

    for metric in args.metrics:
        L(f"### {metric}")
        L("")
        L("| A | B | Δ (A − B) | t (df=4) | p (t) | Wilcoxon W | p (Wilcoxon) |")
        L("|---|---|---:|---:|---:|---:|---:|")
        for a, b in combinations(model_names, 2):
            if metric not in cv_data[a] or metric not in cv_data[b]:
                continue
            r = paired_tests(cv_data[a][metric], cv_data[b][metric])
            def fmt_p(p):
                if not np.isfinite(p):
                    return "—"
                if p < 0.05:
                    return f"**{p:.4f}**"
                if p < 0.10:
                    return f"*{p:.4f}*"
                return f"{p:.4f}"
            L(
                f"| {a} | {b} | {r['mean_diff']:+.4f} | "
                f"{r['t_stat']:+.3f} | {fmt_p(r['t_pvalue'])} | "
                f"{r['wilcoxon_stat']:.1f} | {fmt_p(r['wilcoxon_pvalue'])} |"
            )
        L("")

    # Headline interpretation for AUC
    if "roc_auc" in args.metrics:
        L("## Síntese — Test ROC AUC")
        L("")
        L("Pares com p (t-test) < 0,10 em AUC (cobertura completa para a tabela acima):")
        L("")
        sig = []
        for a, b in combinations(model_names, 2):
            if "roc_auc" not in cv_data[a] or "roc_auc" not in cv_data[b]:
                continue
            r = paired_tests(cv_data[a]["roc_auc"], cv_data[b]["roc_auc"])
            if np.isfinite(r["t_pvalue"]) and r["t_pvalue"] < 0.10:
                sig.append((a, b, r))
        if not sig:
            L("- Nenhum par com p < 0,10 em AUC.")
        else:
            for a, b, r in sig:
                L(f"- **{a} vs {b}**: Δ AUC = {r['mean_diff']:+.4f}, "
                  f"t = {r['t_stat']:+.2f}, p = {r['t_pvalue']:.4f} "
                  f"(Wilcoxon p = {r['wilcoxon_pvalue']:.4f})")
        L("")

    # Footer
    L("---")
    L("")
    L(f"Gerado por `tools/cv_statistical_tests.py`. Para reproduzir:")
    L("")
    L("```bash")
    L("python tools/cv_statistical_tests.py --out cv_statistical.md")
    L("```")

    out_path = args.out
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

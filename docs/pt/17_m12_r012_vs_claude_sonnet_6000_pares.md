# M12 R012 vs Claude Sonnet nos mesmos 6000 pares

**Data:** 2026-06-07
**Modelo:** M12 RGCK-Net, checkpoint da última single run **R012**
(`models/12_rgck_net/output/016/checkpoints/best.pt`, época 3, threshold
salvo no checkpoint = **0.400**).
**Script:** [`tools/eval_m12_on_manifest.py`](../../tools/eval_m12_on_manifest.py)
(roda o M12 sobre um manifesto explícito de pares, em vez de
`KinshipPairDataset(split="test")`).
**Conjunto de pares:** o manifesto da avaliação VLM da Claude Sonnet
(`data/claude_vlm_fiw_binary_6000/manifest.json`), 6000 pares.

> **Aviso de escopo.** Esta avaliação **não** usa o teste completo de
> 13.425 pares do protocolo RFIW Track-I do M12. Ela usa exatamente os
> 6000 pares que a Claude Sonnet já avaliou, para permitir uma
> comparação no mesmo conjunto de pares. O AUC aqui (0,888) **não** é o
> headline de teste do M12 (0,8813 single-run / 0,8761 ± 0,0029 em CV);
> é o desempenho do mesmo checkpoint neste conjunto balanceado
> 3000/3000, cujos negativos têm construção diferente da do teste
> oficial.

---

## Validações obrigatórias (todas confirmadas)

Registradas em `data/m12_r012_on_claude_vlm_6000_pairs/run.log` e em
`metrics.json`:

| Validação | Resultado |
|---|---|
| Manifesto com exatamente 6000 pares | ✓ |
| Composição 3000 kin / 3000 non_kin | ✓ |
| sample_ids únicos (sem amostras duplicadas) | ✓ (6000) |
| Sem pares-imagem duplicados | ✓ (6000 pares únicos) |
| Pares avaliados pelo M12 == pares do manifesto da Claude | ✓ (o script lê o manifesto direto) |
| sample_ids do manifesto == sample_ids do predictions.csv da Claude | ✓ (6000/6000 idênticos) |
| `metrics.json` calculado sobre exatamente 6000 predições | ✓ |
| Imagens resolvidas na árvore alinhada (FIW_aligned), sem fallback | ✓ (0 faltando) |

Notas:
- O preprocessing do M12 foi replicado fielmente: imagens carregadas de
  `datasets/FIW_aligned` (cada caminho `datasets/FIW/<rel>` remapeado
  para `datasets/FIW_aligned/<rel>`), `Resize(224)` → `ToTensor` →
  `Normalize(0.5, 0.5)`, idêntico a `AMD/test.py`.
- No conjunto da Claude, os negativos ficam todos em um bucket único
  (`non-kin`), não estratificados por relação. As relações de parentesco
  (bb, fd, …) contêm apenas pares positivos. Logo, a acurácia por
  relação de parentesco equivale ao recall daquela relação, e o bucket
  `non_kin` mede a specificity.

---

## Resultado — M12 R012 vs Claude Sonnet (6000 pares)

| Métrica | **M12 R012** | Claude Sonnet | Δ (M12 − Claude) |
|---|---:|---:|---:|
| Accuracy | **0,7998** | 0,7228 | +0,0770 |
| Balanced accuracy | **0,7998** | 0,7228 | +0,0770 |
| Precision | **0,7549** | 0,6909 | +0,0640 |
| Recall | **0,8880** | 0,8063 | +0,0817 |
| Specificity | **0,7117** | 0,6393¹ | +0,0724 |
| F1 | **0,8161** | 0,7442 | +0,0719 |
| ROC AUC | **0,8883** | 0,7888 | +0,0995 |
| Average Precision | **0,8817** | 0,7421 | +0,1396 |
| TAR@FAR=0,001 | **0,0663** | 0,0420 | +0,0243 |
| TAR@FAR=0,01 | **0,2587** | 0,0420 | +0,2167 |
| TAR@FAR=0,1 | **0,6337** | 0,2973 | +0,3364 |

¹ Specificity da Claude derivada de `2·balanced_acc − recall` (o
`metrics.json` da Claude não traz specificity explícita).

A Claude Sonnet produz score de confiança, então AUC/AP/TAR existem e
são comparáveis. O **M12 vence em todas as métricas**, com folga grande
em AUC (+0,099), AP (+0,140) e nas TAR de operação (+0,22 em FAR=0,01,
+0,34 em FAR=0,1).

**Matriz de confusão (linhas = verdadeiro, colunas = predito):**

| | predito kin | predito non_kin |
|---|---:|---:|
| **M12** verdadeiro kin | 2664 | 336 |
| **M12** verdadeiro non_kin | 865 | 2135 |

---

## Acurácia por relação

No conjunto da Claude cada relação de parentesco contém só positivos
(os negativos ficam no bucket `non_kin`, que corresponde à specificity).

| Relação | N | M12 | Claude Sonnet |
|---|---:|---:|---:|
| bb | 416 | 0,9111 | 0,7813 |
| fd | 409 | 0,8557 | 0,8093 |
| fs | 558 | 0,8961 | 0,8154 |
| ms | 495 | 0,8606 | 0,8081 |
| md | 474 | 0,9262 | 0,8544 |
| ss | 306 | 0,9216 | 0,8072 |
| sibs | 111 | 0,9820 | 0,7477 |
| gfgd | 68 | 0,8235 | 0,8382 |
| gfgs | 42 | 0,7857 | 0,6429 |
| gmgd | 59 | 0,7458 | 0,7627 |
| gmgs | 62 | 0,7419 | 0,7097 |
| non_kin (specificity) | 3000 | 0,7117 | — |

O M12 vence em 9 das 11 relações de parentesco; a Claude fica
ligeiramente à frente só em gfgd (+1,5 pp) e gmgd (+1,7 pp), duas das
classes menores (68 e 59 pares).

---

## Leitura geral

- O M12 R012 supera a Claude Sonnet em **todas** as métricas no conjunto
  de 6000 pares que a Claude avaliou, inclusive nas que dependem de
  ranking (AUC +0,099, AP +0,140).
- A maior vantagem está em rejeitar não-parentes (specificity 0,712 vs
  0,639) e na separação geral do score (AUC e AP), que é justamente onde
  a VLM zero-shot é mais fraca.
- Este AUC de 0,888 é específico deste conjunto balanceado com negativos
  não estratificados; não substitui o número de teste oficial do M12
  (0,8813 single-run / 0,8761 ± 0,0029 em CV no teste de 13.425 pares).
  É o mesmo checkpoint, mesmo threshold (0,400) e mesmo preprocessing.

---

## Artefatos gerados

- `data/m12_r012_on_claude_vlm_6000_pairs/` — M12 no conjunto da Claude
  (`manifest.json`, `predictions.csv`, `metrics.json`, `run.log`).
- Script: [`tools/eval_m12_on_manifest.py`](../../tools/eval_m12_on_manifest.py).

Referência VLM usada como fonte dos pares (não foi re-executada):
- `data/claude_vlm_fiw_binary_6000/`

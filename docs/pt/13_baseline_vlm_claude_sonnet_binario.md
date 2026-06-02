# Baseline VLM — Claude Sonnet em verificação binária (n = 6 000)

Este documento complementa
[`10_baseline_vlm_zero_shot.md`](10_baseline_vlm_zero_shot.md), que
reporta o piloto de 75 pares com Claude Sonnet 4.6 em classificação
fechada de 11 relações. O experimento descrito aqui responde a uma
pergunta diferente e foi feito sob protocolo diretamente comparável aos
modelos supervisionados.

| Aspecto                | Piloto antigo (75 pares)              | Este experimento (6 000 pares)        |
|------------------------|---------------------------------------|---------------------------------------|
| Tarefa                 | Classificação fechada de 11 relações  | Verificação binária parente/não-parente |
| Amostra                | 75 pares (≈ 7 por classe)             | 6 000 pares = 3 000 positivos + 3 000 não-parentes |
| Fonte                  | Manifesto do `codex_vlm_fiw_150`      | `KinshipPairDataset(test, split_seed=42, negative_ratio=1.0)` — o mesmo pool que B0, M02, M05, M06 e M12 |
| Comparável a sup.      | Não (tarefa diferente)                | Sim (mesma tarefa, mesmo pool)        |
| Modelo                 | `claude-sonnet-4-6`                   | `claude-sonnet-4-6`                   |
| Protocolo              | Zero-shot direto                      | Zero-shot direto                      |
| Custo da API           | < US\$ 1                              | ≈ US\$ 10 (≈ R\$ 53)                  |
| Wall-clock             | minutos                               | ≈ 9 h serial                          |

Estes números são o que pode entrar na tabela principal de resultados
ao lado dos modelos supervisionados. O piloto de 75 pares permanece
útil como leitura exploratória sobre classificação fina de relação,
mas não substitui a comparação binária.

## Por que refazer

O TCC compara cinco abordagens supervisionadas (B0, M02 ViT-FaCoR,
M05 DINOv2-LoRA, M06 com recuperação e M12 AdaFace-Regional) em uma
tarefa binária com classe `non-kin` explícita. O Claude Sonnet do
piloto antigo não estava nessa mesma tarefa — ele escolhia entre 11
relações positivas e nunca via um não-parente. Isso impedia
comparação direta. O experimento aqui descrito coloca o Sonnet sob o
mesmo protocolo dos modelos supervisionados.

## Prompt

```
You are an expert in facial kinship analysis. Given a side-by-side
image of two faces, decide whether the two people are biologically
related — that is, whether ANY direct kin relationship exists between
them (parent-child, sibling, grandparent-grandchild). Base your
decision only on the visual cues in the image. Respond ONLY with a
JSON object: {"is_related": true|false, "confidence": <0-1 float>}.
No other text. The confidence reflects how certain you are about your
is_related answer, where 0 means 'no information' and 1 means 'fully
certain'.
```

## Mapeamento decisão → escore contínuo

O Sonnet emite uma decisão binária mais um valor de confiança no
intervalo `[0, 1]`. Para calcular ROC AUC, AP e TAR@FAR, esses dois
campos são combinados em um único escore contínuo:

| Decisão                  | Escore                              | Faixa        |
|--------------------------|-------------------------------------|--------------|
| `is_related = true`      | `0,5 + 0,5 × confidence`            | `[0,5; 1,0]` |
| `is_related = false`     | `0,5 − 0,5 × confidence`            | `[0,0; 0,5]` |

Escore mais alto significa "mais certeza de parentesco". O limiar
natural é 0,5 — é o que separa as duas decisões. Como o Sonnet não
tem um conjunto de validação próprio, esse 0,5 é o limiar usado para
F1, acurácia, precisão e revocação. As métricas que não dependem de
limiar (AUC, AP, TAR@FAR) são calculadas diretamente sobre o escore
contínuo.

## Resultados principais

| Métrica           | Sonnet binário (n=6000) | B0 (CV)  | M02 R031 (CV)       | M12 R011 (CV média)   | M12 R011 (conjunto CV) |
|-------------------|------------------------:|---------:|--------------------:|----------------------:|-----------------------:|
| Test ROC AUC      | **0,7888**              | 0,7991   | 0,8462 ± 0,0040     | 0,8761 ± 0,0029       | **0,8839**             |
| Test AP           | 0,7421                  | 0,8093   | 0,8131              | 0,8562 ± 0,0031       | 0,8657                 |
| TAR@FAR = 0,001   | 0,0420                  | 0,0706   | 0,0219              | 0,0677 ± 0,0147       | 0,0801                 |
| TAR@FAR = 0,01    | 0,0420 †                | 0,2178   | 0,1349              | 0,2069 ± 0,0107       | 0,2172                 |
| TAR@FAR = 0,1     | 0,2973                  | 0,5238   | 0,4964              | 0,5951 ± 0,0083       | 0,6063                 |
| F1                | 0,7442                  | 0,7121   | 0,7774              | 0,7979 ± 0,0024       | 0,8054                 |
| Acurácia          | 0,7228                  | 0,6660   | 0,7466              | 0,7858 ± 0,0036       | 0,7925                 |
| Precisão          | 0,6909                  | 0,6065   | 0,6724              | 0,7282 ± 0,0090       | 0,7313                 |
| Revocação         | 0,8063                  | 0,8623   | 0,9225              | 0,8826 ± 0,0144       | 0,8961                 |

† O valor de TAR@FAR=0,001 e TAR@FAR=0,01 coincidiu em 0,042. Não é um
erro de implementação — é uma consequência direta de o Sonnet emitir
confiança em um conjunto pequeno de valores discretos (0,5, 0,6, 0,7,
0,8, 0,9, 0,92). Entre FAR=0,001 e FAR=0,01 o limiar deslocaria através
de poucos pontos de score, e nesse intervalo a curva fica plana.

## Acurácia por relação

Considerando apenas a classe correta (relação positiva ou `non-kin`),
a fração de decisões em que o Sonnet acertou o lado da fronteira:

| Classe   | N    | Acurácia |
|----------|-----:|---------:|
| md       | 474  | 85,4 %   |
| gfgd     | 68   | 83,8 %   |
| fs       | 558  | 81,5 %   |
| fd       | 409  | 80,9 %   |
| ms       | 495  | 80,8 %   |
| ss       | 306  | 80,7 %   |
| bb       | 416  | 78,1 %   |
| gmgd     | 59   | 76,3 %   |
| sibs     | 111  | 74,8 %   |
| gmgs     | 62   | 71,0 %   |
| gfgs     | 42   | 64,3 %   |
| **non-kin** | 3 000 | **63,9 %** |

## Como ler estes números

1. **Sonnet zero-shot fica abaixo da linha de base B0 sem treinamento.**
   AUC 0,7888 vs 0,7991 do B0 (AdaFace congelado + cosseno). A diferença
   está em torno de uma vez o desvio entre dobras do B0 (que é 0 por
   construção), mas é estável: o B0 não usa nem treino nem prompt, só
   embeddings de identidade. Que o Sonnet não bata esse piso é um
   resultado forte para a seção de discussão dos VLMs.

2. **A especificidade em `non-kin` é o gargalo (63,9 %).** O Sonnet
   tende a responder "sim, parentesco" sempre que duas faces têm pistas
   visuais genéricas em comum (idade próxima, mesmo gênero, iluminação
   parecida). É exatamente o padrão de erro que o protocolo binário foi
   feito para revelar e que a tarefa de 11 classes do piloto antigo
   ocultava.

3. **Distribuição de escore grosseira.** O Sonnet emite confiança em
   poucos valores discretos. Isso impede operação útil em regimes de
   falsa aceitação estrita: TAR@FAR=0,001 e TAR@FAR=0,01 ficaram iguais
   (0,042) porque entre esses dois pontos não há resolução suficiente
   de escore para rejeitar mais falsos positivos. Para uso prático com
   custo alto de falso positivo, esse VLM zero-shot é inadequado.

4. **O piloto antigo de 75 pares permanece útil em outro plano.** Ele
   responde "dado que o par é positivo, o Sonnet acerta qual relação?",
   o que é uma análise diagnóstica interessante. Os números aqui são os
   que devem entrar na tabela comparativa do TCC. Os do piloto antigo
   permanecem em
   [`10_baseline_vlm_zero_shot.md`](10_baseline_vlm_zero_shot.md) como
   leitura complementar.

## Comparação metodológica

| Aspecto       | Modelos supervisionados                                   | Este experimento                                                            |
|---------------|-----------------------------------------------------------|-----------------------------------------------------------------------------|
| Tarefa        | Verificação binária                                       | Verificação binária                                                         |
| Pool          | `KinshipPairDataset(test, seed=42, negative_ratio=1.0)`   | Subamostragem do mesmo pool: 3 000 positivos + 3 000 negativos              |
| Limiar        | Selecionado em validação (F1-ótimo por dobra)             | 0,5 fixo (fronteira natural do mapeamento; sem conjunto de validação)        |
| AUC / AP / TAR | A partir do escore contínuo                              | A partir do escore contínuo derivado                                        |
| Validação cruzada | 5 dobras família-disjuntas (M02, M05, M06, M12)        | Passada única sobre 6 000 pares — modelo determinístico no nível do prompt  |

A diferença metodológica relevante é a do limiar. Pelas métricas
independentes de limiar (AUC, AP, TAR@FAR) a comparação é direta.
Métricas dependentes de limiar (F1, acurácia, precisão, revocação)
mudariam se o Sonnet tivesse um conjunto de validação para ajustar o
ponto operacional, e por isso são reportadas com a ressalva
correspondente.

## Reprodução

```bash
ANTHROPIC_API_KEY="$(< ~/.anthropic_api_key)" \
/home/bruno/Desktop/tcc_new/models/12_rgck_net/.venv/bin/python \
  tools/run_claude_vlm_fiw_binary.py \
  --output-dir data/claude_vlm_fiw_binary_6000 \
  --num-positives 3000 --num-negatives 3000 \
  --model claude-sonnet-4-6 \
  --delay 0.3 \
  --aligned-root datasets/FIW_aligned
```

A amostragem é determinística pela semente. A resposta da API não é
estritamente determinística mesmo com temperatura zero, então uma
reexecução pode produzir pequenas variações nos números — o ranking
das classes e as conclusões qualitativas, contudo, são estáveis.

## Artefatos

- `data/claude_vlm_fiw_binary_6000/manifest.json` — 6 000 pares (caminho absoluto + rótulo + ptype)
- `data/claude_vlm_fiw_binary_6000/config.json` — prompt, modelo, parâmetros
- `data/claude_vlm_fiw_binary_6000/predictions.csv` — uma linha por par com `sample_id, label, ptype, predicted, confidence, score, raw_text`
- `data/claude_vlm_fiw_binary_6000/metrics.json` — pacote completo de métricas + breakdown por relação
- `data/claude_vlm_fiw_binary_6000/README.md` — versão em inglês deste mesmo documento (mais curta)

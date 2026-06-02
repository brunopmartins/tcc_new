# Experimento com VLM Codex para Verificacao Binaria de Parentesco

## Objetivo

Registrar o experimento complementar com um **VLM do ecossistema Codex** em configuracao **zero-shot** para a tarefa de verificacao binaria de parentesco facial.

A formulacao usada neste experimento e:

```text
kin vs non_kin
```

Essa mudanca foi necessaria para alinhar a avaliacao do VLM ao protocolo dos modelos supervisionados da pesquisa, que tambem decidem se um par possui ou nao relacao de parentesco, em vez de classificar a relacao genealogica exata.

---

## Motivacao

Na versao anterior, o VLM era avaliado como classificador multiclasse entre 11 relacoes positivas do FIW:

```text
bb, ss, sibs, fd, fs, md, ms, gfgd, gfgs, gmgd, gmgs
```

Esse enquadramento gerava uma discrepancia metodologica: os modelos supervisionados da pesquisa realizam **verificacao binaria**, enquanto o VLM realizava **classificacao multiclasse** sem considerar a classe `non_kin`.

Para tornar a comparacao metodologicamente mais coerente, o experimento foi refeito com a mesma pergunta central dos demais modelos:

> As duas faces pertencem a pessoas com relacao de parentesco?

---

## Definicao da Task

Dado um par de faces `(x1, x2)`, o VLM deve escolher exatamente uma das classes:

```text
kin
non_kin
```

Onde:

- `kin` indica que o par possui relacao familiar.
- `non_kin` indica que o par nao possui relacao familiar.

O modelo nao deve inferir ou retornar o tipo especifico de relacao familiar. A saida esperada e apenas uma decisao binaria acompanhada de confianca.

---

## Dataset e Protocolo

### Dataset

O experimento usa o **FIW Track-I**, com pares extraidos de:

```text
datasets/FIW/track-I/test-pairs.csv
```

### Amostragem

Foram avaliados **6000 pares unicos**, balanceados entre as duas classes:

| Classe | Pares |
|---|---:|
| `kin` | 3000 |
| `non_kin` | 3000 |
| Total | 6000 |

O run foi composto por:

| Etapa | Pares |
|---|---:|
| Rodada inicial | 750 |
| Complemento | 5250 |
| Total combinado | 6000 |

O complemento foi executado com exclusao do manifesto inicial, evitando repeticao dos pares ja avaliados. A validacao do manifesto combinado indicou:

| Checagem | Resultado |
|---|---:|
| Pares totais | 6000 |
| Pares `kin` | 3000 |
| Pares `non_kin` | 3000 |
| Duplicatas detectadas | 0 |

---

## Preparacao das Entradas

Cada exemplo foi convertido em uma imagem composta com:

1. Face da pessoa A na esquerda.
2. Face da pessoa B na direita.
3. Layout fixo para todos os pares.

Essa representacao simplifica o prompt e garante que o VLM receba sempre a mesma estrutura visual.

---

## Prompting

O prompt foi fixo para todas as amostras, preservando o carater zero-shot do experimento.

Formato conceitual do prompt:

```text
You will receive one face pair image containing a left face and a right face.
Decide whether the two people are biologically related.
Choose exactly one label: kin or non_kin.
Return only a JSON object with:
{
  "predicted_label": "kin" or "non_kin",
  "confidence": <0_to_1>
}
```

O prompt nao solicita a relacao especifica (`fd`, `md`, `sibs` etc.), pois isso recriaria a discrepancia com o protocolo binario dos modelos supervisionados.

---

## Configuracao do Modelo

| Campo | Valor |
|---|---|
| Modelo | `gpt-5.4-mini` |
| Familia | Codex VLM |
| Modo | zero-shot |
| Reasoning effort | `medium` |
| Saida | JSON estruturado |
| Tarefa | verificacao binaria |

---

## Artefatos de Reproducao

O resultado combinado esta salvo em:

```text
data/codex_vlm_fiw_binary_6000_medium_combined/
```

Principais arquivos:

| Arquivo | Conteudo |
|---|---|
| `config.json` | configuracao do run |
| `manifest.json` | pares avaliados |
| `predictions.csv` | ground truth, predicao e confianca |
| `metrics.json` | metricas agregadas |

As duas rodadas que compoem o resultado combinado estao em:

```text
data/codex_vlm_fiw_binary_1500_medium/
data/codex_vlm_fiw_binary_extra_5250_medium/
```

---

## Resultados

### Metricas globais

| Metrica | Valor |
|---|---:|
| Accuracy | 58.98% |
| Balanced accuracy | 58.98% |
| Precision | 0.563 |
| Recall / sensibilidade `kin` | 0.799 |
| Specificity / rejeicao `non_kin` | 0.381 |
| F1 | 0.661 |
| Confidence media | 0.721 |
| Confidence media em acertos | 0.735 |
| Confidence media em erros | 0.700 |

### Matriz de confusao

| Classe real | Predito `kin` | Predito `non_kin` | Total |
|---|---:|---:|---:|
| `kin` | 2396 | 604 | 3000 |
| `non_kin` | 1857 | 1143 | 3000 |

Em termos de verificacao binaria:

| Tipo | Quantidade |
|---|---:|
| Verdadeiros positivos | 2396 |
| Falsos negativos | 604 |
| Falsos positivos | 1857 |
| Verdadeiros negativos | 1143 |

---

## Interpretacao

O Codex VLM apresentou desempenho acima do acaso, com **balanced accuracy de 58.98%**, mas mostrou forte tendencia a predizer `kin`.

Essa tendencia aparece na diferenca entre:

| Indicador | Valor |
|---|---:|
| Recall para `kin` | 79.87% |
| Specificity para `non_kin` | 38.10% |

Ou seja, o modelo reconhece muitos pares positivos, mas rejeita mal pares negativos. Isso gera um numero elevado de falsos positivos: **1857 pares `non_kin` foram classificados como `kin`**.

O resultado sugere que, em zero-shot, o VLM usa pistas visuais gerais de semelhanca facial, idade, genero e aparencia global, mas ainda nao possui calibracao suficiente para separar parentesco real de semelhanca superficial.

---

## Comparabilidade com os Modelos Supervisionados

Com a reformulacao binaria, o experimento passa a ser mais comparavel aos modelos supervisionados da pesquisa do que a versao multiclasse anterior.

Ainda assim, a comparacao deve ser feita com cautela:

1. O VLM nao foi treinado no FIW.
2. O VLM nao possui ajuste de threshold em validacao.
3. O VLM nao usa validacao cruzada.
4. O VLM produz uma decisao zero-shot baseada no prompt, nao em uma funcao de similaridade supervisionada.

Portanto, o resultado deve ser usado como **baseline externo zero-shot**, nao como substituto direto dos modelos especializados.

---

## Nota Sobre o Experimento Multiclasse Anterior

A versao anterior deste documento descrevia uma tarefa de classificacao fechada em 11 relacoes positivas:

```text
bb, ss, sibs, fd, fs, md, ms, gfgd, gfgs, gmgd, gmgs
```

Essa formulacao foi mantida apenas como etapa exploratoria historica, pois nao inclui a classe `non_kin` e, por isso, nao mede a mesma tarefa dos modelos de verificacao binaria.

O resultado principal a ser usado na comparacao metodologica e o experimento binario de 6000 pares documentado acima.

---

## Conclusao

O experimento com Codex VLM mostra que um modelo multimodal generico consegue capturar parte do sinal visual de parentesco sem treinamento no dominio, mas ainda apresenta desempenho limitado para rejeitar pares `non_kin`.

O valor metodologico do experimento esta em estabelecer um baseline zero-shot binario: ele mede quanto da tarefa pode ser resolvido apenas com conhecimento visual geral e evidencia por que modelos supervisionados, calibrados especificamente para verificacao de parentesco facial, continuam necessarios.

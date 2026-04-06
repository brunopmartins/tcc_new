# Experimento com VLM Codex para Classificacao de Parentesco

## Objetivo

Definir um experimento complementar aos modelos treinados usando um **VLM do ecossistema Codex** como baseline **zero-shot** para classificacao de parentesco facial. A proposta e medir quanto um modelo multimodal generico consegue inferir a relacao familiar apenas a partir de pistas visuais, **sem treinamento supervisionado no dominio**.

---

## Motivacao

Os modelos 01-04 desta pesquisa exigem treinamento, validacao e ajuste de hiperparametros. Um VLM, por outro lado, permite uma avaliacao imediata:

1. **Sem treinamento**: basta preparar os pares e definir um prompt fixo.
2. **Baseline de baixo custo**: o experimento pode ser executado rapidamente como referencia externa.
3. **Analise qualitativa**: os erros do VLM revelam quais relacoes dependem mais de semelhanca facial fina e menos de pistas amplas de idade/genero.

O papel deste experimento nao e substituir os modelos especializados, mas sim fornecer um **baseline zero-shot** e uma camada extra de interpretacao metodologica.

---

## Hipotese

Um VLM generico deve conseguir resolver razoavelmente bem relacoes em que **idade** e **genero** oferecem pistas fortes, como:

- `fd` e `fs`
- `md` e `ms`
- `sibs`

Por outro lado, espera-se maior dificuldade em relacoes que exigem distinguir **graus genealogicos** mais sutis:

- `bb` vs `fs`
- `ss` vs `md`
- `gfgd`, `gfgs`, `gmgd`, `gmgs`

---

## Definicao da Task

O experimento usa o problema como **classificacao fechada da relacao de parentesco**.

Dado um par de faces `(x1, x2)`, o VLM deve escolher exatamente uma classe em:

```text
bb, ss, sibs, fd, fs, md, ms, gfgd, gfgs, gmgd, gmgs
```

Onde:

- `bb` = brother-brother
- `ss` = sister-sister
- `sibs` = brother-sister
- `fd` = father-daughter
- `fs` = father-son
- `md` = mother-daughter
- `ms` = mother-son
- `gfgd` = grandfather-granddaughter
- `gfgs` = grandfather-grandson
- `gmgd` = grandmother-granddaughter
- `gmgs` = grandmother-grandson

Este enquadramento e diferente da verificacao binaria (`kin` vs `non-kin`) usada nos modelos treinados, e por isso deve ser tratado como **baseline complementar**, nao como substituto direto.

---

## Dataset e Protocolo

### Dataset recomendado

Usar o **FIW Track-I**, pois ele ja fornece pares rotulados por tipo de relacao e cobre as 11 classes necessarias.

### Split

Usar o **split oficial de teste** do FIW, mantendo o protocolo padrao do dataset.

### Filtragem

Antes da inferencia:

1. Verificar se os caminhos de imagem listados no split existem no dataset local.
2. Descartar entradas com paths quebrados ou inconsistentes.
3. Manter um **manifest** deterministicamente salvo em disco para reproducao.

### Balanceamento

Para evitar que relacoes abundantes dominem a analise:

1. Amostrar o mesmo numero de pares por relacao sempre que possivel.
2. Se alguma relacao tiver menos pares validos do que o alvo planejado, limitar essa classe ao maximo disponivel e redistribuir o deficit entre as demais relacoes.
3. Fixar uma seed unica para tornar a amostragem reprodutivel.

---

## Preparacao das Entradas

Cada exemplo deve ser convertido em uma **imagem composta** com:

1. Face da pessoa A na esquerda
2. Face da pessoa B na direita
3. Um layout fixo para todos os pares

Essa representacao simplifica o prompt e reduz ambiguidade na ordem das imagens.

Pipeline sugerido:

1. Ler o par `(img1, img2)`
2. Redimensionar/cortar ambas para um tamanho padrao
3. Combinar lado a lado em uma unica imagem
4. Enviar a imagem composta ao Codex VLM

---

## Prompting

O prompt deve ser **fixo** para todas as amostras. Isso evita tuning manual por relacao e preserva o caracter zero-shot.

Exemplo de prompt:

```text
You will receive one kinship pair image containing a left face and a right face.
Classify the family relation using only the visual evidence in that image.
Choose exactly one label from:
bb, ss, sibs, fd, fs, md, ms, gfgd, gfgs, gmgd, gmgs.
Return only a JSON object with:
{
  "predicted_relation": "<one_label>",
  "confidence": <0_to_1>
}
Do not use shell commands or inspect files.
```

Para throughput maior, tambem e possivel **batchar varias imagens compostas por chamada**, pedindo um array JSON na mesma ordem dos anexos.

---

## Configuracao do Modelo

O experimento deve fixar uma variante do Codex VLM e mantela constante durante toda a avaliacao.

Configuracao recomendada:

- **Modelo:** `gpt-5.4` ou `gpt-5.4-mini`
- **Modo:** zero-shot
- **Reasoning effort:** `low` ou `medium`
- **Saida estruturada:** JSON schema fixo

Para estudos futuros, pode-se comparar:

1. `gpt-5.4-mini` vs `gpt-5.4`
2. `low` vs `medium` reasoning
3. inferencia individual vs inferencia em lote

---

## Metricas

Como a task e multiclasse, as metricas principais devem ser:

1. **Accuracy global**
2. **Macro Precision**
3. **Macro Recall**
4. **Macro F1**
5. **Accuracy por relacao**
6. **Matriz de confusao**

Tambem vale registrar:

1. **Confidence media**
2. **Confidence media em acertos**
3. **Confidence media em erros**

Isso ajuda a medir se o VLM esta bem calibrado ou apenas responde com alta confianca independentemente da qualidade.

---

## Artefatos de Reproducao

O experimento deve salvar pelo menos:

1. `config.json` com modelo, seed, batch size e configuracao do run
2. `manifest.json` com os pares efetivamente avaliados
3. `predictions.csv` com ground truth, predicao e confianca
4. `metrics.json` com accuracy, macro-F1 e matriz de confusao

Esses artefatos tornam o VLM benchmarkavel da mesma forma que os runs dos modelos treinados.

---

## Limitacoes Esperadas

1. **Nao ha adaptacao ao dominio**: o VLM nao aprende com FIW.
2. **Sensibilidade a heuristicas superficiais**: idade e genero podem dominar a decisao.
3. **Nao e comparacao direta com AUC de verificacao**: a task aqui e multiclasse, nao binaria.
4. **Custo de inferencia por chamada**: embora nao haja treinamento, ha custo por lote de imagens.

Por isso, o VLM deve ser interpretado como **baseline zero-shot exploratorio**, nao como candidato principal para SOTA na tarefa.

---

## Como Posicionar na Dissertacao

Este experimento pode ser apresentado como:

1. **Baseline sem treinamento**
2. **Controle externo ao pipeline supervisionado**
3. **Analise complementar de erro**

Uma formulacao adequada seria:

> Alem dos modelos treinados especificamente para verificacao de parentesco, foi avaliado um VLM do ecossistema Codex em configuracao zero-shot, com o objetivo de medir o quanto um modelo multimodal generico consegue inferir a relacao familiar apenas por pistas visuais. O resultado serve como baseline complementar e evidencia a diferenca entre conhecimento visual geral e especializacao supervisionada no dominio de parentesco facial.

---

## Conclusao

O experimento com VLM Codex acrescenta valor metodologico porque responde a uma pergunta diferente da dos modelos 01-04:

**quanto do problema de parentesco facial pode ser resolvido sem treinamento, apenas com capacidade visual geral?**

Essa comparacao fortalece a narrativa da pesquisa ao mostrar, de forma concreta, onde um VLM generico acerta, onde falha, e por que modelos especializados continuam necessarios.

# Adaptacao ao Dominio por Prompt no VLM Codex

## Objetivo

Avaliar se uma forma de **adaptacao ao dominio baseada em inferencia**, sem qualquer fine-tuning do backbone, consegue melhorar o baseline zero-shot do Codex VLM para classificacao de parentesco facial no FIW.

O objetivo nao foi treinar um novo modelo, mas sim adaptar o comportamento do VLM ao dominio por meio de:

1. **few-shot in-context learning**
2. **prompt estruturado por decisao**
3. **calibracao leve em validacao**

---

## O que significa "adaptacao ao dominio" aqui

Neste experimento, o termo **adaptacao ao dominio** foi usado no sentido de **adaptacao inferencial/prompt-based**, e nao no sentido classico de treinamento adicional.

Mais especificamente:

- o modelo continua sendo o mesmo `gpt-5.4-mini`
- nao ha ajuste fino
- nao ha atualizacao de pesos
- o dominio e incorporado via:
  - exemplos few-shot do proprio FIW
  - definicoes mais explicitas das relacoes
  - saidas auxiliares que obrigam o modelo a explicitar gap geracional e genero
  - uma etapa leve de calibracao baseada em validacao

---

## Desenho Experimental

### Modelo

- **Runtime:** Codex CLI
- **Modelo:** `gpt-5.4-mini`
- **Reasoning effort:** `low`

### Split de Validacao

Para selecao da configuracao adaptada, foi usado o **`FIW/track-I/val-pairs.csv`**.

Como esse arquivo armazena pares em nivel de membro com a coluna `face_pairs`, cada par foi expandido para pares de faces validos no dataset local. Em seguida foi feita uma amostragem estratificada com:

- **1 exemplo demo por relacao** para o conjunto de exemplos few-shot
- **10 pares de validacao por relacao** para selecao de configuracao

Isso resultou em:

- **11 demos canonicos** disponiveis
- **110 pares de validacao** balanceados

### Split de Teste

Para manter comparabilidade direta com o baseline zero-shot ja documentado, o teste reutilizou exatamente o mesmo manifesto:

- `data/codex_vlm_fiw_1500/manifest.json`

Ou seja:

- **750 pares positivos**
- **1500 imagens**
- mesma distribuicao por relacao do baseline Codex zero-shot

---

## Configuracoes Testadas em Validacao

Foram comparadas quatro configuracoes:

| Configuracao | Shots | Prompt |
|--------------|-------|--------|
| `seven_shot_v1` | 7 | Estruturado, mais conciso |
| `seven_shot_v2` | 7 | Estruturado, com tabela de decisao mais rigida |
| `eleven_shot_v1` | 11 | Um exemplo por relacao, prompt conciso |
| `eleven_shot_v2` | 11 | Um exemplo por relacao, prompt mais rigido |

### Conjunto de 7 shots

O conjunto reduzido foi composto pelas relacoes:

- `bb`
- `ss`
- `sibs`
- `fd`
- `ms`
- `gfgd`
- `gmgs`

A ideia foi cobrir:

- mesma geracao
- uma geracao de diferenca
- duas geracoes de diferenca
- combinacoes centrais de genero

---

## Prompt Estruturado

O prompt adaptado exigia que o modelo retornasse, para cada par:

- `predicted_relation`
- `runner_up_relation`
- `confidence`
- `generation_gap`
- `older_face_side`
- `left_gender_guess`
- `right_gender_guess`

A intencao era forcar uma decomposicao intermediaria do problema:

1. decidir se o par parece de **mesma geracao**, **uma geracao de diferenca** ou **duas geracoes de diferenca**
2. estimar os generos
3. usar semelhanca facial fina apenas para desempatar dentro do subconjunto plausivel de relacoes

---

## Calibracao Leve

Com base nas saidas estruturadas, foi tentada uma calibracao leve em validacao.

A calibracao testou:

- `none`
- `gap_only`
- `gap_plus_low_conf` com thresholds `0.55`, `0.65` e `0.75`

O principio era simples:

- se o `predicted_relation` estivesse incoerente com o `generation_gap`, usar `runner_up_relation` ou uma relacao derivada dos sinais auxiliares
- opcionalmente aplicar essa correcao tambem em casos de baixa confianca

---

## Resultados em Validacao

| Configuracao | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|--------------|----------|-----------------|--------------|----------|
| `seven_shot_v1` | **26.4%** | **0.239** | **0.264** | **0.243** |
| `eleven_shot_v1` | 24.5% | 0.228 | 0.245 | 0.228 |
| `seven_shot_v2` | 22.7% | 0.232 | 0.227 | 0.223 |
| `eleven_shot_v2` | 16.4% | 0.154 | 0.164 | 0.153 |

Melhor configuracao selecionada:

- **Prompt:** `seven_shot_v1`
- **Shots:** `7`

### Resultado da Calibracao em Validacao

Nenhuma variante de calibracao melhorou a saida bruta:

| Politica | Accuracy | Macro F1 |
|----------|----------|----------|
| `none` | **26.4%** | **0.243** |
| `gap_only` | 26.4% | 0.243 |
| `gap_plus_low_conf @ 0.55` | 26.4% | 0.243 |
| `gap_plus_low_conf @ 0.65` | 26.4% | 0.243 |
| `gap_plus_low_conf @ 0.75` | 26.4% | 0.243 |

Portanto, a configuracao escolhida para o teste final foi:

- **prompt:** `seven_shot_v1`
- **calibracao:** `none`

---

## Resultados no Teste Held-Out

### Comparacao com o Baseline Zero-Shot

| Metrica | Zero-shot baseline | Adaptado por prompt | Adaptado + calibracao |
|---------|--------------------|---------------------|-----------------------|
| **Accuracy** | **33.1%** | 26.7% | 26.7% |
| **Macro Precision** | **0.245** | 0.203 | 0.203 |
| **Macro Recall** | **0.320** | 0.260 | 0.260 |
| **Macro F1** | **0.257** | 0.223 | 0.223 |

### Deltas em Relacao ao Baseline

- **Accuracy:** `-6.4` pontos
- **Macro Precision:** `-0.0419`
- **Macro Recall:** `-0.0600`
- **Macro F1:** `-0.0344`

O resultado final foi **negativo**: a adaptacao ao dominio por prompt piorou o desempenho em vez de melhora-lo.

---

## Efeito por Relacao

### Relacoes que Melhoraram

| Relacao | Zero-shot | Adaptado |
|---------|-----------|----------|
| `sibs` | 58.6% | **67.1%** |
| `bb` | 8.3% | **15.3%** |
| `ss` | 20.0% | **22.9%** |
| `gmgd` | 0.0% | **4.3%** |

### Relacoes que Pioraram

| Relacao | Zero-shot | Adaptado |
|---------|-----------|----------|
| `md` | **76.4%** | 33.3% |
| `fd` | **72.9%** | 47.1% |
| `ms` | **48.6%** | 35.7% |
| `fs` | **67.1%** | 60.0% |

### Relacoes que Continuaram Sem Solucao

- `gfgd`: `0.0%`
- `gfgs`: `0.0%`
- `gmgs`: `0.0%`

---

## Padrões de Erro

### Baseline Zero-Shot

Os erros dominantes do baseline eram fortemente orientados por idade/genero:

- `gfgs -> md` (`41`)
- `gfgd -> ms` (`39`)
- `gmgs -> gfgd` (`32`)
- `gmgs -> fd` (`25`)

### Versao Adaptada

Os erros mais frequentes passaram a ser:

- `gfgd -> ms` (`42`)
- `gfgs -> md` (`35`)
- `gmgs -> gfgd` (`29`)
- `md -> ss` (`25`)
- `ss -> sibs` (`23`)
- `ms -> sibs` (`20`)

Isso mostra que a adaptacao:

- **nao resolveu** o problema principal das relacoes avo/avoa-neto(a)
- aumentou a rigidez em torno de categorias de **mesma geracao**
- introduziu novas confusoes em relacoes **mae-filha** e **mae-filho**

---

## O que os Sinais Auxiliares Mostram

Os campos estruturados do prompt ajudam a explicar por que a calibracao nao funcionou:

- a estimativa de **gap geracional** acertou apenas **54.4%** dos 750 pares
- uma relacao derivada diretamente desses sinais ficou disponivel em **99.3%** dos casos
- essa relacao derivada coincidiu com a predicao final em **91.6%** dos pares

Interpretacao:

- o modelo passou a seguir seus sinais intermediarios com alta consistencia
- esses sinais, porem, **nao eram confiaveis o suficiente**
- por isso a calibracao praticamente nao teve liberdade para corrigir nada

Em outras palavras, o prompt estruturado fez o VLM ficar mais **deterministico**, mas nao mais **acurado**.

---

## Recorte Diagnostico nas Relacoes de Avo(a)-Neto(a)

### Por que repetir o teste nesse recorte

Esse recorte foi criado porque, no teste held-out de `750` pares, as relacoes de avo(a)-neto(a) foram o ponto mais fragil de todo o baseline. Como a adaptacao por prompt foi desenhada justamente para explicitar **gap geracional**, fazia sentido verificar se essa vantagem apareceria ao menos em um subconjunto pequeno onde nao ha diluicao por classes mais faceis como `md`, `fd` e `fs`.

A comparacao, portanto, nao teve o objetivo de substituir o teste principal, mas sim de responder a uma pergunta mais especifica:

- quando o experimento olha apenas para `gfgd`, `gfgs`, `gmgd` e `gmgs`, o prompt adaptado finalmente ajuda?

### Protocolo do recorte

- manifesto de origem: `data/codex_vlm_fiw_1500/manifest.json`
- seed de amostragem: `20260413`
- amostra: `3` pares por relacao de avo(a)-neto(a)
- relacoes avaliadas: `gfgd`, `gfgs`, `gmgd`, `gmgs`
- total: `12` pares / `24` imagens
- comparacao: baseline **zero-shot** original vs prompt adaptado **`seven_shot_v1`**

Os IDs escolhidos e o manifesto completo foram salvos junto aos artefatos do recorte, para que essa analise possa ser refeita sem alterar o conjunto.

### Resultados do recorte

| Metrica | Zero-shot | Adaptado (`seven_shot_v1`) |
|---------|-----------|----------------------------|
| Accuracy exata | `0/12` = **0.0%** | `0/12` = **0.0%** |
| Macro F1 ativo (4 relacoes) | `0.000` | `0.000` |
| Predicao no bucket avo(a)-neto(a) | `1/12` = **8.3%** | `2/12` = **16.7%** |
| Confianca media | `0.794` | `0.768` |
| Campo `generation_gap = two_generation_apart` | n/a | `2/12` = **16.7%** |

### Interpretacao do recorte

O resultado e severo, mas muito esclarecedor: mesmo quando o teste e reduzido apenas para as quatro relacoes de avo(a)-neto(a), o prompt adaptado **nao consegue produzir nenhum acerto exato**. Isso reforca a leitura do experimento maior: o problema nao era apenas a media global ser puxada para baixo por algumas classes dificeis; a falha nas relacoes de duas geracoes de distancia e real e persistente.

Ao mesmo tempo, o recorte mostra um detalhe importante. O baseline zero-shot colocou apenas `1` dos `12` pares dentro do bucket correto de avo(a)-neto(a), enquanto a versao adaptada colocou `2` de `12`. Ou seja, houve um ganho muito pequeno na percepcao de que alguns pares pareciam estar separados por duas geracoes. Esse ganho, porem, foi insuficiente para identificar a relacao correta.

O padrao mais forte apareceu em `gfgd` e `gfgs`:

- os `3` pares de `gfgd` continuaram sendo previstos como `ms`
- os `3` pares de `gfgs` continuaram sendo previstos como `md`

Isso e especialmente relevante para a hipotese do prompt adaptado. Se a estrategia realmente estivesse aprendendo a usar o **gap geracional** como filtro principal, era razoavel esperar pelo menos uma migracao parcial dessas classes para o bucket avo(a)-neto(a). Nao foi o que aconteceu.

As unicas mudancas qualitativas apareceram em dois pares:

- um caso de `gmgd` passou de `fd` para `gfgs`
- um caso de `gmgs` permaneceu no bucket avo(a)-neto(a), mas como `gfgd`

Esses dois exemplos mostram que o prompt estruturado conseguiu, em raros momentos, empurrar a predicao para o **bucket geracional correto**. Ainda assim, ele errou a combinacao fina entre sexo da geracao mais velha e sexo do neto(a), que e exatamente o nivel de detalhe exigido por `gfgd`, `gfgs`, `gmgd` e `gmgs`.

O proprio campo auxiliar `generation_gap` confirma essa leitura. Na versao adaptada, o modelo declarou `two_generation_apart` em apenas `2` de `12` pares. Nos outros `10`, ele mesmo descreveu os pares como `one_generation_apart`, o que praticamente sela o erro antes da classificacao final.

Em resumo, a comparacao foi util porque isola o caso mais dificil e mostra com clareza onde a adaptacao falha:

- ela quase nunca corrige o nivel geracional
- quando corrige o bucket, ainda nao resolve a relacao exata
- portanto, o ganho e apenas parcial e nao altera a conclusao central do estudo

---

## Sweep de Prompts Direcionados no Mesmo Recorte

### Motivacao

Depois do primeiro recorte de `12` pares, ficou uma duvida importante: o fracasso vinha principalmente do espaco de rotulos `11-way`, onde o VLM escorrega para `fd`, `fs`, `md` e `ms`, ou o problema persistiria mesmo com prompts mais agressivos e especializados para avo(a)-neto(a)?

Para responder isso, foi feito um segundo sweep no **mesmo manifesto de 12 pares**, sem trocar amostra, variando apenas o prompt.

### Familias de prompt testadas

Foram comparados dois grupos novos de prompt:

- **11-way com guardrails**, ainda permitindo todas as `11` relacoes, mas forçando mais explicitamente o VLM a separar `same_generation`, `one_generation_apart` e `two_generation_apart`
- **4-way oracular**, restringindo o espaco de resposta apenas a `gfgd`, `gfgs`, `gmgd` e `gmgs`

Importante:

- os prompts **11-way** continuam comparaveis com a tarefa original
- os prompts **4-way** sao apenas **diagnosticos**, porque usam a informacao externa de que este recorte contem apenas relacoes de avo(a)-neto(a)

### Configuracoes novas

| Configuracao | Tipo | Demos | Observacao |
|--------------|------|-------|------------|
| `directed_11way_guardrail_v1` | 11-way | nao | reforca "duas geracoes" antes da decisao final |
| `directed_11way_guardrail_v2` | 11-way | nao | usa tabela de eliminacao mais rigida |
| `oracle_4way_zero_shot_v1` | 4-way | nao | mapeia genero do idoso + genero do mais jovem |
| `oracle_4way_zero_shot_v2` | 4-way | nao | mesmo espaco 4-way, com regras mais estritas |
| `fewshot_oracle_4way_v1` | 4-way | sim | um demo por relacao avo(a)-neto(a) |

### Resultado agregado

| Configuracao | Accuracy exata | Macro F1 ativo | Bucket avo(a)-neto(a) | `two_generation_apart` declarado |
|--------------|----------------|----------------|------------------------|----------------------------------|
| zero-shot referencia | `0/12` = **0.0%** | `0.000` | `8.3%` | n/a |
| `seven_shot_v1` referencia | `0/12` = **0.0%** | `0.000` | `16.7%` | `16.7%` |
| `directed_11way_guardrail_v1` | `0/12` = **0.0%** | `0.000` | `25.0%` | `25.0%` |
| `directed_11way_guardrail_v2` | `0/12` = **0.0%** | `0.000` | `33.3%` | `33.3%` |
| `oracle_4way_zero_shot_v1` | `1/12` = **8.3%** | `0.071` | `100%` | `91.7%` |
| `oracle_4way_zero_shot_v2` | `1/12` = **8.3%** | `0.071` | `100%` | `100%` |
| `fewshot_oracle_4way_v1` | `0/12` = **0.0%** | `0.000` | `100%` | `100%` |

### Como interpretar esses resultados

O sweep mostra tres coisas diferentes.

Primeiro, os prompts **11-way** realmente empurram o modelo um pouco mais na direcao correta do **bucket geracional**. O melhor deles (`directed_11way_guardrail_v2`) saiu de `16.7%` para `33.3%` de predicoes dentro do bucket avo(a)-neto(a). Ainda assim, isso nao se converteu em nenhum acerto exato. Em outras palavras, mais estrutura ajuda o VLM a "suspeitar" um pouco mais de duas geracoes de distancia, mas nao o suficiente para sair das heuristicas erradas.

Segundo, os prompts **4-way** mostram que parte do problema realmente estava no espaco amplo de rotulos. Quando o modelo fica proibido de responder `fd`, `fs`, `md` ou `ms`, ele finalmente para de colapsar para pai/mae-filho(a). Isso fica visivel porque o bucket avo(a)-neto(a) vai para `100%`. Mas aqui existe uma ressalva metodologica importante: esse `100%` nao significa sucesso real, pois e praticamente garantido pela propria restricao do label space. O numero que importa nesses prompts passa a ser a **accuracy exata**, e ela sobe apenas para `1/12`.

Terceiro, o experimento `fewshot_oracle_4way_v1` mostra que **mais contexto nao necessariamente ajuda**. Mesmo com um demo por relacao de avo(a)-neto(a), o resultado voltou para `0/12`, com confianca media acima de `0.90`. Isso sugere que os exemplos few-shot nao corrigiram a heuristica visual do modelo; ao contrario, podem ter reforcado uma regra errada aplicada com mais conviccao.

### O que os erros dos prompts 4-way revelam

Os prompts 4-way foram os mais informativos porque removem a desculpa do "bucket errado". Neles, o VLM e obrigado a escolher entre apenas quatro classes:

- `gfgd`
- `gfgs`
- `gmgd`
- `gmgs`

Mesmo assim, o padrao de erro continuou fortemente sistematico:

- os `3` casos de `gfgd` foram classificados como `gmgs`
- os `3` casos de `gfgs` foram classificados como `gmgd`
- os `3` casos de `gmgd` foram distribuidos entre `gfgd` e `gfgs`
- os `3` casos de `gmgs` tiveram apenas `1` acerto

Isso indica que o gargalo nao e apenas "perceber que sao duas geracoes de distancia". O modelo ainda falha em algo mais fino:

- identificar corretamente qual face deve ser tratada como a mais velha
- inferir o genero da geracao mais velha com estabilidade
- mapear de forma consistente genero da pessoa mais velha + genero da pessoa mais jovem para a relacao final

Ou seja, ao restringir o problema para `4` classes, o erro deixa de ser "pai/mae vs avo/avoa" e passa a ser um erro de **papel genealogico fino**.

### Leitura final do sweep

Esse segundo teste refina a interpretacao anterior.

- Se o prompt continua `11-way`, ele melhora um pouco a chance de cair no bucket correto, mas nao resolve a classificacao final.
- Se o prompt vira `4-way`, o bucket deixa de ser o problema principal, mas a discriminacao entre `gfgd`, `gfgs`, `gmgd` e `gmgs` continua muito fraca.
- Se adicionamos demos few-shot dentro do espaco `4-way`, o modelo nao melhora e ainda fica mais confiante nos erros.

Portanto, aumentar a variedade de prompts direcionados foi util porque mostra que existe um teto claro para esse tipo de adaptacao inferencial:

- o prompt consegue mudar o **tipo de erro**
- mas nao consegue entregar uma separacao fina confiavel entre as relacoes de avo(a)-neto(a)

---

## Conclusao

Neste repositório, a tentativa de **adaptacao ao dominio baseada em prompt** para o Codex VLM:

- foi metodologicamente valida
- manteve separacao clara entre validacao e teste
- foi totalmente reproduzivel
- mas **nao melhorou o baseline zero-shot**

Esse e um resultado util para a pesquisa porque mostra que:

1. mais estrutura no prompt nao implica melhor desempenho
2. few-shot em FIW nao foi suficiente para aproximar o VLM da tarefa fine-grained
3. o gargalo continua sendo distinguir **graus genealogicos** mais do que apenas idade/genero
4. a especializacao supervisionada segue sendo necessaria
5. mesmo num recorte focado de `12` pares apenas com relacoes de avo(a)-neto(a), o modelo ficou em `0/12`, o que reforca que o gargalo nao esta na mistura de classes do teste geral
6. ao restringir o problema para apenas `4` relacoes de avo(a)-neto(a), o modelo chegou a no maximo `1/12`, mostrando que o limite nao e so o label space amplo, mas a propria discriminacao visual fina entre subtipos genealogicos

Portanto, a contribuicao deste experimento nao esta em "superar o baseline", mas em demonstrar de forma controlada que uma adaptacao inferencial aparentemente razoavel pode **piorar** o desempenho global ao reforcar heuristicas superficiais.

---

## Artefatos

Todos os artefatos do experimento foram salvos em:

- `data/codex_vlm_fiw_domain_adapt_1500/`
- `data/codex_vlm_fiw_grandparent_slice_12/`
- `data/codex_vlm_fiw_grandparent_prompt_sweep_12/`

Arquivos principais:

- `validation_results.json`
- `calibration_results.json`
- `selected_configuration.json`
- `comparison_summary.json`
- `test/seven_shot_v1/predictions.csv`
- `test/seven_shot_v1/metrics_raw.json`
- `tools/run_codex_vlm_fiw_domain_adapt.py`
- `pair_level_comparison.csv`
- `tools/run_codex_vlm_fiw_grandparent_slice.py`
- `prompt_sweep_summary.json`
- `tools/run_codex_vlm_fiw_grandparent_prompt_sweep.py`

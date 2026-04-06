# Baseline Zero-Shot com Modelo de Linguagem Visual (VLM)

## Visão Geral

Como experimento complementar às abordagens supervisionadas treinadas especificamente para verificação de parentesco, este trabalho inclui uma avaliação *zero-shot* utilizando um Modelo de Linguagem Visual (VLM) de grande escala — o **Claude Sonnet** da Anthropic. O objetivo é estabelecer um ponto de referência não supervisionado que representa o limite superior do conhecimento visual geral sem treinamento específico para a tarefa.

Este baseline é **complementar**, não competitivo: ele responde à pergunta *"o que um modelo de visão de propósito geral consegue inferir sobre parentesco apenas olhando para os rostos?"*, sem qualquer exemplo de treinamento ou adaptação ao domínio.

---

## Motivação

Modelos de linguagem visual pré-treinados em bilhões de pares imagem-texto acumulam conhecimento implícito sobre aparência humana, relações familiares e características faciais. Testar esses modelos em modo zero-shot permite:

1. **Quantificar o gap** entre conhecimento visual geral e o aprendizado supervisionado específico para parentesco.
2. **Identificar quais relações** são naturalmente discrimináveis pela aparência (e.g., pai-filho *vs.* avô-neto).
3. **Fornecer um baseline interpretável** — o modelo usa apenas raciocínio visual sem acesso a metadados, embeddings treinados ou pares de referência.

---

## Metodologia

### Protocolo de Avaliação

- **Conjunto de dados:** FIW (*Families in the Wild*) — split de teste do Track-I.
- **Tarefa:** Classificação fechada (*closed-set*) de relação de parentesco entre dois rostos.
- **Classes:** 11 relações de parentesco FIW — `bb` (irmão-irmão), `ss` (irmã-irmã), `sibs` (irmão-irmã), `fd` (pai-filha), `fs` (pai-filho), `md` (mãe-filha), `ms` (mãe-filho), `gfgd` (avô-neta), `gfgs` (avô-neto), `gmgd` (avó-neta), `gmgs` (avó-neto).
- **Modo:** Zero-shot — nenhum exemplo de treinamento, ajuste fino ou adaptação ao domínio.

### Construção das Amostras

Os pares de avaliação são amostrados de forma **balanceada** entre as 11 relações, garantindo representação uniforme. Apenas pares positivos (indivíduos com relação de parentesco confirmada) são utilizados, e somente imagens cujos arquivos existam localmente são incluídas. A amostragem é reprodutível via semente fixa (`seed=20260406`).

Para cada par amostrado, as duas imagens faciais são compostas lado a lado em uma folha de par (*pair sheet*) padronizada de 512×274 pixels, sem informações textuais além do identificador do par.

### Inferência com VLM

O modelo recebe cada folha de par e uma instrução em linguagem natural solicitando a classificação da relação de parentesco dentro do conjunto fechado de 11 classes. O *prompt* do sistema instrui o modelo a:

- Analisar **apenas as evidências visuais** presentes na imagem.
- Retornar **exatamente uma** das 11 etiquetas de relação.
- Fornecer um **escore de confiança** entre 0 e 1.

Nenhum exemplo adicional (*few-shot*), metadado de família ou informação auxiliar é fornecido — apenas as duas faces e a lista de classes válidas.

### Métricas Reportadas

- **Acurácia geral** (*overall accuracy*)
- **Precisão macro**, **Recall macro**, **F1 macro** — média não ponderada entre as 11 classes
- **Acurácia por relação** — desempenho individual em cada um dos 11 tipos de parentesco
- **Matriz de confusão** — padrões de erro sistemático entre relações
- **Confiança média** — separação entre predições corretas e incorretas

---

## Análise Qualitativa das Relações

### Relações Mais Discrimináveis Visualmente

As relações com maior diferença de idade e gênero entre os indivíduos tendem a ser mais facilmente identificáveis:

- **Pai-filha (`fd`) e pai-filho (`fs`):** A combinação de diferença de gênero (fd) ou semelhança fenotípica com gap etário acentuado (fs) fornece pistas visuais robustas.
- **Irmão-irmã (`sibs`):** A presença simultânea de similaridade fenotípica e diferença de gênero é discriminativa.

### Relações de Maior Ambiguidade

- **Relações entre irmãos do mesmo sexo (`bb`, `ss`):** Sem diferença de gênero e com faixas etárias similares, a distinção de parentesco lateral é altamente ambígua.
- **Relações de segundo grau (`gfgd`, `gfgs`, `gmgd`, `gmgs`):** A separação geracional de dois níveis introduz grande variabilidade fenotípica, e a sobreposição visual com relações pai-filho é elevada — modelos sem treinamento específico confundem sistematicamente avós com pais.

### Limitações Inerentes ao Zero-Shot

A classificação de parentesco facial é uma tarefa **intrinsecamente difícil** mesmo para humanos: estudos mostram que humanos não superam 60–70% de acurácia em tarefas similares sem contexto. Modelos VLM zero-shot enfrentam desafios adicionais:

1. **Ambiguidade de gênero em fotos históricas ou de baixa qualidade.**
2. **Sem calibração de limiar:** o modelo não tem acesso à distribuição de similaridade da família específica.
3. **Confusão sistemática entre graus de parentesco:** a diferença visual entre pai-filho e avô-neto é frequentemente insuficiente para distinção confiável.

---

## Interpretação e Comparação com Modelos Treinados

O baseline zero-shot estabelece o ponto de partida antes de qualquer supervisão. Os modelos treinados neste trabalho superam este baseline ao aprender:

- **Embeddings específicos de parentesco** via perda contrastiva supervisionada.
- **Invariância a variações de iluminação, pose e envelhecimento** via data augmentation.
- **Calibração de similaridade** ajustada à distribuição do FIW.

A diferença de desempenho entre o VLM zero-shot e os modelos treinados quantifica o **ganho obtido pelo treinamento supervisionado** no domínio de verificação de parentesco — e motiva a necessidade de abordagens especializadas para esta tarefa.

---

## Artefatos Reprodutíveis

Todos os artefatos da avaliação são versionados para reprodutibilidade:

| Arquivo | Descrição |
|---------|-----------|
| `data/claude_vlm_fiw_150/config.json` | Configuração completa do experimento |
| `data/claude_vlm_fiw_150/manifest.json` | Pares amostrados com caminhos de imagem |
| `data/claude_vlm_fiw_150/predictions.csv` | Predições por par com ground truth |
| `data/claude_vlm_fiw_150/metrics.json` | Métricas agregadas e matriz de confusão |
| `data/claude_vlm_fiw_150/pair_sheets/` | Imagens compostas utilizadas na avaliação |
| `tools/run_claude_vlm_fiw.py` | Script de amostragem e avaliação |

---

*Este documento descreve a metodologia geral do baseline VLM zero-shot. Os resultados numéricos completos encontram-se em `docs/pt/09_visao_geral_pesquisa.md` e em `data/claude_vlm_fiw_150/metrics.json`.*

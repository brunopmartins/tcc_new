# Avaliação crítica do TCC até o Capítulo 06

## 1. Diagnóstico geral do TCC até o Capítulo 06

O TCC apresenta nível técnico acima da média para um trabalho de graduação, especialmente pela maturidade metodológica, pela clareza na delimitação do problema e pela preocupação explícita com limitações éticas e experimentais. A estrutura geral está bem definida: a introdução contextualiza a verificação visual de parentesco, os trabalhos relacionados situam a evolução da área, a fundamentação teórica oferece os conceitos necessários, a metodologia descreve modelos e protocolos, os resultados são interpretados com cautela e a conclusão retoma achados e limitações.

O principal mérito do texto é não se limitar a apresentar resultados. O trabalho tenta construir uma comparação controlada entre famílias de modelos, distingue métricas dependentes e independentes de threshold, explicita que VLMs não são diretamente comparáveis aos supervisionados e trata com cautela a interpretação dos achados experimentais. Esse grau de autoconsciência acadêmica é positivo e reduz a vulnerabilidade diante da banca.

Apesar da qualidade geral, há problemas relevantes de escrita e apresentação. O texto, em vários momentos, usa linguagem muito próxima de relatório técnico interno: "run", "headline de AUC", "full FT", "partial unfreeze", "operating point", "loss", "piso de ruído", "destrava", "ataca". Esses termos são compreensíveis para quem acompanhou o projeto, mas exigem padronização, tradução quando possível e definição na primeira ocorrência. O TCC também apresenta algumas afirmações fortes que exigem citação específica ou tom mais cauteloso, especialmente sobre estado da arte, conflitos armados, VLMs e inexistência de trabalhos equivalentes.

A validação cruzada 5-fold é um eixo metodológico central do trabalho. Por isso, as tabelas e a discussão devem ser lidas e redigidas em termos de média, desvio padrão e variação entre dobras, evitando que a argumentação dependa da melhor execução isolada ou de um único checkpoint.

Parecer sintético: o trabalho é promissor, tecnicamente consistente e defensável, mas requer revisão textual para reduzir informalidades, explicitar melhor a reprodutibilidade, fortalecer referências e alinhar a força das conclusões ao protocolo 5-fold adotado.

## 2. Análise capítulo por capítulo

### Capítulo 1 - Introdução

A introdução é um dos pontos fortes do trabalho. Ela delimita bem o escopo: parentesco consanguíneo por imagens faciais, sem substituir DNA, documentos ou vínculos sociais. O problema de pesquisa, o objetivo geral, os objetivos específicos, a justificativa, as contribuições e as delimitações aparecem de forma clara. Isso prepara adequadamente o leitor.

Há, porém, alguns pontos a ajustar. A contextualização humanitária é relevante, mas usa eventos contemporâneos e números sensíveis. A frase sobre "conflitos armados em curso" e as 19.500 crianças na Ucrânia deve estar muito bem referenciada e datada, porque a banca pode questionar a atualidade e a fonte. Idealmente, use fontes institucionais primárias quando possível, ou deixe claro que se trata de dado reportado por fonte jornalística em data específica.

Também há uma afirmação ampla: "os melhores resultados recentes no FIW ficam próximos de 92% de acurácia". Essa frase precisa indicar exatamente quais trabalhos, qual protocolo, qual métrica e qual split. A própria metodologia do TCC insiste corretamente que accuracy, AUC e TAR@FAR não são sempre comparáveis; portanto, a introdução não deve usar "92% de acurácia" de modo genérico.

Os objetivos estão bons, mas longos e muito técnicos. O objetivo específico sobre o Modelo 12, por exemplo, lista muitos componentes arquitetônicos em uma única frase. Para a introdução, isso pode ser mais legível se o detalhamento completo ficar no Capítulo 4 e o objetivo mencionar a proposta em termos mais sintéticos.

### Capítulo 2 - Trabalhos Relacionados

O capítulo é bem organizado cronologicamente e por famílias de método. A progressão dos métodos clássicos para redes siamesas, metric learning, Transformers, síntese de idade, fairness e VLMs faz sentido. A seção de posicionamento do trabalho é útil porque mostra onde cada modelo se encaixa.

O principal problema é que parte do capítulo soa mais como justificativa interna do projeto do que como revisão crítica da literatura. Em alguns pontos, os autores são citados mais para legitimar uma família de modelos do que para comparar pressupostos, limitações, protocolos e resultados. A banca pode perguntar: em que esses trabalhos diferem metodologicamente entre si? Quais usam split oficial? Quais escolhem threshold em teste? Quais reportam AUC, accuracy ou TAR? Qual é a real comparabilidade dos números?

A seção sobre VLMs precisa de mais cautela. A afirmação de que avaliações sistemáticas de VLMs para parentesco facial "não foram encontradas" é aceitável, mas deve ser formulada como escopo de busca, não como inexistência absoluta. Exemplo de direção: "na revisão realizada para este trabalho, não foram identificados estudos sistemáticos...". Isso evita uma afirmação difícil de provar.

### Capítulo 3 - Fundamentação Teórica

A fundamentação é clara e tecnicamente útil. As equações de atenção, cross-attention, LoRA e perda contrastiva ajudam o leitor a compreender as escolhas metodológicas. A seção de métricas também é pertinente, porque prepara a leitura dos resultados.

O capítulo, no entanto, mistura fundamentação conceitual com descrição de implementação específica. Há várias passagens que dizem "no Modelo 05 deste trabalho", "no Modelo 12", "a versão implementada..." e já antecipam decisões metodológicas. Isso não é necessariamente errado, mas reduz a separação entre teoria e método. Uma versão mais acadêmica deixaria o Capítulo 3 concentrado em conceitos e trabalhos de referência, enquanto detalhes como rank LoRA, número de parâmetros, galeria de 33 mil pares e variantes de execução ficariam prioritariamente no Capítulo 4.

Outro ponto: a seção sobre Supervised Contrastive Loss é honesta ao dizer que a implementação é mais próxima da Contrastive Loss clássica do que da formulação canônica de Khosla. Isso é bom, mas deve ser tratado com cuidado terminológico. Se a perda implementada não é a Supervised Contrastive Loss canônica, a nomenclatura no restante do TCC deve evitar induzir o leitor a pensar que se trata exatamente da formulação original.

### Capítulo 4 - Metodologia

O Capítulo 4 é o núcleo mais forte do trabalho. Ele descreve datasets, amostragem, modelos, treinamento, plasticidade dos backbones, protocolo de avaliação, métricas, VLMs, hardware e reprodutibilidade. A metodologia é detalhada e, em boa parte, replicável.

O capítulo também está bem conectado aos anteriores: ele responde ao problema de pesquisa, operacionaliza os objetivos específicos e justifica por que os modelos foram escolhidos. A conexão com a introdução e com os trabalhos relacionados é satisfatória.

As lacunas principais são de replicabilidade fina. O texto deve explicitar com precisão: versão exata do FIW/RFIW, fonte de download, procedimento de detecção e alinhamento facial, tratamento de falhas de detecção, transformações e normalização por modelo, data augmentation, critérios de exclusão de imagens, composição exata das imagens lado a lado usadas nos VLMs, prompts completos ou referência a apêndice, datas/model versions dos VLMs e forma de parsing das respostas.

Há uma inconsistência importante na seção de VLMs. O texto diz que a saída pode ser uma das onze relações "ou uma indicação de não-parentesco", mas em seguida informa que a amostra é de 750 pares positivos balanceados entre onze classes. Se a avaliação contém apenas pares positivos, a classe "não-parentesco" não deve aparecer como classe de saída avaliada, ou então deve haver explicação sobre como respostas "não-parentesco" foram tratadas. Esse ponto pode gerar questionamento direto da banca.

A seção sobre amostragem negativa é louvável pela transparência ao reconhecer a falha da estratégia `relation_matched`. Porém, qualquer resultado que dependa dessa estratégia deve ser claramente marcado como variação de seed, não como teste de hard negatives. O texto já faz isso, mas convém conferir se essa cautela aparece também nas tabelas e interpretações do Capítulo 5.

### Capítulo 5 - Resultados

O capítulo apresenta resultados com boa organização e interpretação. As tabelas são úteis, e o texto evita a comparação direta indevida entre VLMs e modelos supervisionados. A distinção entre AUC, Average Precision, TAR@FAR e métricas dependentes de threshold é um ponto forte.

O problema central é a força de algumas conclusões diante do desenho experimental adotado. Como o trabalho assume validação cruzada 5-fold, a interpretação deve se apoiar nos valores médios e nos desvios padrão entre dobras, não apenas na melhor execução ou no melhor checkpoint. Expressões como "superou" e "deixa de ser um empate prático" são aceitáveis apenas quando acompanhadas da evidência agregada: "obteve maior AUC médio na validação cruzada 5-fold" ou "apresentou ganho médio de X, com desvio padrão Y".

Há também uma promessa textual que parece não se cumprir integralmente: o capítulo afirma que métricas dependentes de threshold ficam na seção sobre operating points, mas não há uma seção explicitamente intitulada dessa forma. A ausência de uma tabela própria com accuracy, F1, precision e recall globais por modelo enfraquece a consistência entre metodologia e resultados, já que o texto menciona essas métricas como reportadas.

Outro ponto sensível: o B0 tem melhor TAR@FAR=0,001 e TAR@FAR=0,01 absolutos em algumas tabelas. Isso é muito interessante, mas também enfraquece uma leitura simplista de que o Modelo 12 é "melhor" em todos os regimes. O texto já reconhece isso; a recomendação é tornar essa leitura ainda mais explícita no resumo dos resultados.

### Capítulo 6 - Discussão e Conclusão

O Capítulo 6 é maduro e defensável. Ele retoma o problema, apresenta os achados principais, qualifica a contribuição do Modelo 12, enumera limitações metodológicas e discute implicações éticas. A seção de limitações é particularmente forte, porque antecipa críticas reais de banca.

O principal ajuste é reduzir repetição em relação ao Capítulo 5. Alguns parágrafos repetem números e interpretações quase com a mesma função do capítulo anterior. A conclusão deve ser mais sintética e interpretativa: menos recapituladora, mais avaliativa.

A seção ética é adequada, mas comporta linguagem mais precisa sobre risco biométrico, consentimento, populações vulneráveis e impossibilidade de uso decisório isolado. Como o trabalho menciona aplicações forenses e humanitárias, a banca pode cobrar uma postura ética robusta.

## 3. Principais fragilidades encontradas

### Críticas

1. Apresentação dos resultados 5-fold como eixo da análise. Como o texto assume validação cruzada, é indispensável reportar média, desvio padrão e, se possível, intervalo de confiança ou teste pareado entre modelos.

2. Atribuição causal da contribuição arquitetônica formulada com cuidado. Mesmo com 5-fold, o texto deve distinguir o ganho do conjunto do Modelo 12 do efeito isolado de cada componente, como tokens regionais, descongelamento parcial, cabeça auxiliar e forward simétrico.

3. Inconsistência metodológica na avaliação dos VLMs. É necessário esclarecer se há classe "não-parentesco" ou apenas onze classes positivas; se há apenas pares positivos, a saída "não-parentesco" precisa ser tratada explicitamente.

4. Reprodutibilidade incompleta do pré-processamento. Falta explicitar com precisão como as faces foram detectadas, alinhadas, cortadas, normalizadas, aumentadas e descartadas em caso de falha.

5. Afirmações de estado da arte e atualidade precisam de citação específica. Em especial, "92% de acurácia no FIW", "conflitos armados em curso", "não foram encontradas avaliações sistemáticas" e "não foi encontrada combinação equivalente" exigem cuidado.

### Relevantes

1. Mistura excessiva de português e inglês técnico sem padronização. Termos como "run", "full FT", "partial unfreeze", "operating point", "loss", "baseline", "seed" e "headline" devem ser traduzidos ou definidos.

2. Repetição entre capítulos. Introdução, trabalhos relacionados, metodologia, resultados e conclusão retomam várias vezes a lista dos modelos. A repetição ajuda a orientar, mas pode ser enxugada.

3. Capítulo 3 antecipa metodologia em excesso. Parte dos detalhes de implementação poderia migrar para o Capítulo 4.

4. Comparações com literatura são pouco tabulares. Uma tabela de trabalhos relacionados com dataset, protocolo, métrica, resultado e limitação aumentaria muito a qualidade acadêmica.

5. A metodologia dos VLMs precisa ser documentada em apêndice. Prompts, exemplos few-shot, modelo, data, temperatura, critério de resposta e tratamento de recusas ou respostas ambíguas devem ser preservados.

6. O texto precisa diferenciar melhor "melhor AUC observado", "melhor regime de baixa FAR" e "melhor modelo para aplicação". Esses são critérios diferentes.

### Pontuais

1. Substituir expressões informais: "o trabalho atacou", "destrava", "headline de AUC", "piso forte", "deixa de ser um empate prático", "kinship-fortes".

2. Padronizar "threshold" ou "limiar"; "accuracy" ou "acurácia"; "Average Precision" ou "precisão média"; "run" ou "execução".

3. Corrigir pequenos resíduos editoriais no arquivo principal, como "modelo dssse TCC" nos comentários e metadados PDF genéricos.

4. Verificar se o título do arquivo `6_Conclusoes_Preeliminares.tex` não indica versão preliminar em material entregue, embora o título compilado esteja correto.

5. Remover ou manter fora da compilação arquivos de rascunho como `various_things_not_to_add.tex`, que têm trechos claramente não adequados ao texto final.

## 4. Pontos fortes do trabalho

1. Delimitação clara do problema: o texto distingue parentesco consanguíneo, vínculo social, DNA, documentação e análise visual.

2. Objetivos bem alinhados com metodologia e resultados. O que é prometido na introdução aparece operacionalizado no Capítulo 4.

3. Consciência metodológica acima da média. O trabalho discute threshold, TAR@FAR, AUC, diferença entre tarefas binária e multiclasse, validação cruzada e limites de comparação.

4. Transparência sobre falhas e limites. A falha do `relation_matched`, a execução parcial do SAM e a limitação dos VLMs são apresentadas com honestidade.

5. Resultados bem interpretados. As tabelas não são apenas despejadas; há leitura de significado, trade-offs e limites.

6. Discussão ética pertinente. O texto não vende a tecnologia como solução decisória e reforça seu papel como triagem ou apoio.

7. Boa organização macroestrutural. O leitor consegue acompanhar a progressão: problema, literatura, teoria, método, resultados e conclusão.

## 5. Trechos ou tipos de trechos que precisam de reescrita

1. Trechos com afirmações amplas sem fonte específica. Exemplo de tipo: "os melhores resultados recentes no FIW ficam próximos de 92% de acurácia". Reescrever com autores, ano, métrica, protocolo e ressalva de comparabilidade.

2. Trechos datados ou politicamente sensíveis. Exemplo de tipo: "conflitos armados em curso..." Reescrever com data da consulta, fonte e formulação menos dependente do momento.

3. Trechos com linguagem de relatório interno. Exemplo de tipo: "headline de AUC", "piso de ruído", "run", "full FT", "partial unfreeze". Substituir por linguagem acadêmica ou definir termos.

4. Trechos que atribuem causalidade sem ablação suficiente. Exemplo de tipo: "a combinação de tokens regionais... superou". Melhor formular como resultado observado sob o conjunto experimental, distinguindo associação de prova causal.

5. Trechos sobre VLMs. A avaliação é interessante, mas a tarefa é diferente. Reescrever para eliminar qualquer impressão de comparação direta com verificação binária supervisionada.

6. Trechos excessivamente longos na metodologia. O primeiro parágrafo do Capítulo 4 e a descrição do Modelo 12 concentram muitas informações técnicas. Dividir em frases menores melhora legibilidade.

7. Trechos com terminologia inconsistente. Escolher uma forma principal para termos técnicos e manter o padrão até o fim.

## 6. Perguntas prováveis da banca

1. Como foi organizada a validação cruzada 5-fold em FIW? As dobras são disjuntas por família?

2. Qual é a média e o desvio padrão do ganho do Modelo 12 sobre o Modelo 02 nas cinco dobras?

3. Quais ablações permitem separar o efeito dos tokens regionais, do descongelamento parcial, da cabeça auxiliar e do forward simétrico?

4. Como as faces foram detectadas, alinhadas e normalizadas? O mesmo pré-processamento foi usado em todos os modelos?

5. Como foram tratados pares negativos potencialmente ambíguos ou parentes não anotados?

6. O erro encontrado no sampler `relation_matched` afetou quais resultados exatamente?

7. Por que os VLMs foram avaliados em classificação multiclasse, e não na mesma verificação binária dos demais modelos?

8. A avaliação dos VLMs contém classe não-parentesco ou apenas relações positivas?

9. Por que o B0 supera os modelos treinados em TAR@FAR extremamente baixo? O que isso implica para aplicação forense?

10. O modelo foi avaliado por grupo demográfico, idade, gênero aparente ou qualidade da imagem?

11. A diferença entre famílias de treino e teste impede completamente vazamento de identidade ou há imagens duplicadas/semelhantes?

12. Como o trabalho garante que o Modelo 12 não está aprendendo atalhos de dataset, pose, idade ou gênero?

13. Qual seria o uso prático aceitável desse sistema, considerando falsos positivos e privacidade biométrica?

14. O que exatamente diferencia a contribuição autoral de uma combinação incremental de módulos existentes?

## 7. Recomendações de melhoria em ordem de prioridade

### Crítico

1. Apresentar tabelas e discussão no protocolo 5-fold, reportando média, desvio padrão e variação por dobra ao afirmar superioridade do Modelo 12.

2. Esclarecer a metodologia dos VLMs, especialmente a presença ou ausência de classe não-parentesco, tamanho das amostras, prompts e tratamento de respostas ambíguas.

3. Completar a descrição de pré-processamento facial: detecção, alinhamento, cortes, normalização, aumentos, falhas e diferenças entre modelos.

4. Formular a atribuição causal da arquitetura regional com cautela, separando resultado global do modelo e efeito isolado de cada componente.

5. Revisar afirmações de estado da arte e inserir citações específicas para números comparativos.

### Relevante

1. Criar uma tabela de trabalhos relacionados com autor, ano, dataset, protocolo, métrica, resultado e limitação.

2. Padronizar terminologia técnico-acadêmica e reduzir estrangeirismos desnecessários.

3. Separar melhor teoria e metodologia, deixando o Capítulo 3 menos dependente dos nomes dos modelos implementados.

4. Inserir uma tabela clara de métricas globais dependentes de threshold, ou remover a promessa de uma seção de operating points se ela não existir.

5. Tornar a seção ética mais apoiada em princípios: consentimento, contestação, rastreabilidade, auditoria, risco de falso positivo e populações vulneráveis.

6. Indicar no texto quais dados, logs, scripts e checkpoints permitem reprodução de cada resultado reportado.

### Pontual

1. Trocar "atacar o problema" por "abordar o problema".

2. Trocar "run" por "execução" ou "rodada experimental".

3. Trocar "headline de AUC" por "principal resultado em AUC".

4. Trocar "piso de ruído" por "variação empírica observada entre execuções".

5. Trocar "full FT" por "ajuste fino completo" e "partial unfreeze" por "descongelamento parcial".

6. Revisar frases longas da introdução e metodologia, dividindo períodos com mais de três ideias.

7. Corrigir resíduos editoriais no arquivo principal e nos comentários LaTeX antes da compilação final.

## 8. Parecer final como professor avaliador

Meu parecer é favorável com ressalvas acadêmicas relevantes. O trabalho tem tema atual, problema bem delimitado, metodologia ambiciosa, resultados relevantes e uma discussão honesta das limitações. Há densidade técnica suficiente para um TCC forte em Engenharia da Computação.

Entretanto, a versão avaliada demanda lapidação acadêmica. A banca provavelmente não questionará apenas se o modelo obteve bom AUC; ela perguntará se a comparação é justa, se a diferença é robusta, se a contribuição autoral foi isolada, se os VLMs foram avaliados de modo compatível e se o protocolo é plenamente replicável. Esses são os pontos que concentram maior risco de arguição.

Os resultados 5-fold consolidados, o pré-processamento explicitado, a avaliação dos VLMs esclarecida e as afirmações de estado da arte bem referenciadas colocam o TCC em condição sólida para banca. Do ponto de vista textual, o trabalho já tem uma base boa; o principal ajuste é substituir marcas de relatório interno por linguagem acadêmica mais estável e alinhar cada afirmação forte ao nível real de evidência apresentado.

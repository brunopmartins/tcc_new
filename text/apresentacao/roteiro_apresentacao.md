# Roteiro de apresentacao do TCC

Duracao alvo: cerca de 24 minutos. Estrutura: 19 slides. A fala abaixo e um roteiro, nao um texto para leitura literal. O objetivo e manter a defesa fluida, com mais tempo nos resultados e nas ressalvas metodologicas.

## Distribuicao de tempo

| Slide | Tema | Tempo |
|---|---|---:|
| 1 | Titulo | 0:30 |
| 2 | Roteiro da defesa | 0:30 |
| 3 | Problema | 1:20 |
| 4 | Motivacao e escopo | 1:10 |
| 5 | Perguntas de pesquisa | 1:30 |
| 6 | Contribuicoes | 1:00 |
| 7 | Metodologia experimental | 1:40 |
| 8 | Modelos comparados | 1:20 |
| 9 | Hipotese do AdaFace-Regional | 1:20 |
| 10 | Fluxo da arquitetura | 1:50 |
| 11 | Componentes da proposta | 1:30 |
| 12 | Variantes avaliadas | 1:00 |
| 13 | Resultado principal | 2:00 |
| 14 | Leitura estatistica e operacional | 1:40 |
| 15 | Ciclo experimental | 1:30 |
| 16 | Analise por tipo de relacao | 1:20 |
| 17 | VLMs como linha de base externa | 1:20 |
| 18 | Limitacoes e etica | 0:50 |
| 19 | Conclusao | 0:50 |

Tempo total aproximado: 24:10.

## Slide 1 - Titulo (0:30)

Boa tarde. Eu sou Bruno Peixoto Martins e vou apresentar meu trabalho de conclusao de curso: "Verificacao Visual de Parentesco Facial Automatica". O trabalho compara arquiteturas supervisionadas de deep learning e tambem avalia modelos multimodais generalistas como linhas de base externas.

O foco da apresentacao e mostrar o problema, o protocolo experimental, a proposta AdaFace-Regional e os principais resultados no FIW.

## Slide 2 - Roteiro da defesa (0:30)

A apresentacao esta organizada em sete partes. Primeiro eu contextualizo o problema e a motivacao. Depois apresento as perguntas de pesquisa, o protocolo experimental, a arquitetura proposta, os resultados, a avaliacao com VLMs e, por fim, as limitacoes e conclusoes.

Vou concentrar mais tempo na metodologia e nos resultados, porque sao os pontos centrais para avaliar a contribuicao do trabalho.

## Slide 3 - Problema (1:20)

A tarefa estudada e a verificacao visual de parentesco facial. Dado um par de imagens de rosto, o sistema precisa decidir se as pessoas tem parentesco biologico ou nao. No texto, essa tarefa aparece como uma classificacao binaria entre kin e non-kin.

Ela e diferente de reconhecimento facial tradicional. Em reconhecimento facial, a pergunta e se duas imagens representam a mesma identidade. Aqui a pergunta e mais sutil: duas pessoas diferentes podem compartilhar tracos familiares sem serem a mesma pessoa.

O problema e dificil por dois motivos opostos. Parentes podem parecer bastante diferentes, por idade, expressao, pose, iluminacao ou heranca desigual de tracos. Ao mesmo tempo, pessoas sem vinculo biologico podem ser visualmente parecidas. O desafio, portanto, nao e apenas criar uma regra de decisao, mas produzir uma ordenacao confiavel dos pares.

## Slide 4 - Motivacao e escopo (1:10)

A motivacao cientifica vem de uma fragilidade comum na literatura: muitos trabalhos usam protocolos, limiares, bancos e metricas diferentes. Isso dificulta comparar percentuais diretamente. Uma parte importante deste TCC e justamente colocar modelos distintos sob o mesmo protocolo.

Tambem ha uma motivacao recente: modelos multimodais generalistas se tornaram populares e levantam a pergunta de ate que ponto eles conseguiriam resolver essa tarefa por prompt, sem treino especifico no FIW.

Do ponto de vista aplicado, a verificacao de parentesco por face pode apoiar triagem em investigacoes, reunificacao familiar e busca humanitaria. Mas o escopo etico e restrito. O sistema nao substitui DNA, documento, investigacao social nem avaliacao especializada. Ele e, no maximo, uma ferramenta auxiliar de triagem.

## Slide 5 - Perguntas de pesquisa (1:30)

O trabalho tem uma pergunta principal e uma pergunta exploratoria.

A pergunta principal e: em que medida uma arquitetura supervisionada especializada, combinando backbone facial, descongelamento parcial e tokens regionais anatomicos, apresenta ganho sobre uma linha de base sem treinamento e outras arquiteturas supervisionadas no FIW?

A pergunta exploratoria e sobre VLMs: qual e o desempenho de modelos multimodais generalistas em verificacao binaria zero-shot quando eles respondem a mesma pergunta kin/non-kin?

O objetivo geral segue essas duas perguntas. Primeiro, comparar cinco abordagens supervisionadas sob um protocolo unico. Segundo, mapear o desempenho de dois VLMs como linhas de base externas, sem trata-los como equivalentes aos modelos treinados.

## Slide 6 - Contribuicoes (1:00)

As contribuicoes principais sao quatro.

A primeira e o protocolo comparativo uniforme. Todos os modelos supervisionados sao avaliados no FIW com validacao cruzada por familia, escolha de limiar em validacao e metricas comuns.

A segunda e a proposta AdaFace-Regional. Ela combina um backbone facial especializado, tokens regionais anatomicos, atencao cruzada regional, gate, cabeca auxiliar de relacao e passagem simetrica.

A terceira e a avaliacao binaria zero-shot de Codex e Claude Sonnet.

A quarta e a analise por tipo de relacao, especialmente porque relacoes de avos e netos sao historicamente mais dificeis no FIW.

## Slide 7 - Metodologia experimental (1:40)

O dataset principal e o Families in the Wild, na configuracao do Track-I do RFIW. Ele possui 761 familias e onze relacoes positivas, incluindo pais e filhos, maes e filhos, irmaos e quatro relacoes de avos e netos.

O protocolo principal usa validacao cruzada de cinco dobras separadas por familia. Em cada rodada, uma dobra fica para validacao e as demais para treino. O conjunto de teste oficial fica preservado. O limiar e escolhido somente na validacao e depois aplicado no teste sem reotimizacao.

Isso e importante porque evita escolher limiar olhando para o teste. A metrica principal e ROC AUC, porque ela independe de um limiar especifico. Tambem sao reportadas Average Precision e TAR@FAR em 0,1, 0,01 e 0,001, porque a tarefa tem semelhanca com verificacao biometrica, onde a falsa aceitacao precisa ser controlada.

## Slide 8 - Modelos comparados (1:20)

Os cinco modelos supervisionados cobrem familias diferentes.

O AdaFace-Cosseno e a linha de base sem treinamento. Ele usa embeddings de um AdaFace IR-101 congelado e similaridade de cosseno.

O ViT-FaCoR representa a familia baseada em Vision Transformer com atencao cruzada entre faces.

O DINOv2-LoRA testa uma representacao auto-supervisionada ampla, adaptada com LoRA e atencao cruzada diferencial.

O Modelo com Recuperacao usa uma galeria de pares positivos do treino como memoria externa.

Por fim, o AdaFace-Regional e a proposta autoral, com AdaFace parcialmente descongelado, regioes anatomicas, atencao regional, gate e passagem simetrica.

## Slide 9 - Hipotese do AdaFace-Regional (1:20)

A hipotese do AdaFace-Regional e intermediaria. O AdaFace-Cosseno mostra que um extrator facial forte ja contem sinal util para parentesco, mas comparar apenas embeddings globais limita a adaptacao ao FIW.

Por outro lado, modelos com patches densos tem muita liberdade e podem explorar partes pouco informativas, como fundo, cabelo ou artefatos de iluminacao.

O AdaFace-Regional tenta preservar o sinal facial especializado e reorganiza-lo para a tarefa. A proposta compara regioes anatomicas explicitas e ainda considera que a tarefa e simetrica: se A e parente de B, B tambem e parente de A.

## Slide 10 - Fluxo da arquitetura (1:50)

O fluxo comeca com duas faces alinhadas. Cada face passa pelo AdaFace IR-101. O modelo extrai um embedding global e tambem embeddings de cinco regioes: face, olhos, nariz, boca e mandibula.

Esses tokens regionais das duas faces entram em uma atencao cruzada regional. Como sao cinco regioes de cada lado, a matriz de interacao e pequena, 5 por 5. Isso e bem mais restrito do que comparar centenas de patches de um ViT.

Depois, o modelo calcula similaridades por regiao e pesos de gate. A cabeca final combina informacoes globais, diferencas, produtos elemento a elemento, similaridades regionais e pesos do gate. A saida e um logit binario kin/non-kin, com uma cabeca auxiliar que prediz o tipo de relacao apenas nos pares positivos.

## Slide 11 - Componentes da proposta (1:30)

Aqui estao os componentes mais importantes.

O backbone AdaFace e parcialmente descongelado: os estagios iniciais ficam preservados e o ultimo estagio recebe gradiente. Isso tenta equilibrar adaptacao ao FIW e preservacao da representacao de identidade.

Os tokens regionais inserem um viés estrutural: o modelo e orientado a comparar partes faciais interpretaveis.

O gate usa sigmoide por regiao. Isso permite que mais de uma regiao seja relevante ao mesmo tempo.

A passagem simetrica processa o par nas duas ordens. Durante a avaliacao, o logit final e a media de z(A,B) e z(B,A). Isso reduz dependencia da posicao das imagens no par.

## Slide 12 - Variantes avaliadas (1:00)

As variantes formam uma sequencia de desenvolvimento. A Base usa descongelamento parcial, regioes, atencao e gate.

A Simetrica adiciona a avaliacao nas duas ordens. A Comparativa remove embeddings globais crus da fusao final, mantendo apenas termos comparativos. A variante com Negativos Dificeis altera a amostragem, incluindo 30% de negativos formados por familias diferentes, mas preservando os mesmos papeis familiares do par positivo.

Um cuidado importante: isso nao e uma ablacao estrita de todos os componentes. A conclusao mais segura e sobre o conjunto de componentes associado ao ganho.

## Slide 13 - Resultado principal (2:00)

Este e o resultado central do trabalho. Sob validacao cruzada de cinco dobras, o AdaFace-Regional com Negativos Dificeis atinge AUC de 0,876 mais ou menos 0,003. Ele fica acima do ViT-FaCoR, que atinge 0,846, do DINOv2-LoRA, com 0,814, da linha de base AdaFace-Cosseno, com 0,799, e do Modelo com Recuperacao, com 0,778.

As tres variantes finalistas do AdaFace-Regional ficam muito proximas em AUC: 0,873, 0,874 e 0,876. Entao eu nao interpreto isso como uma superioridade forte entre elas. A leitura principal e que a familia AdaFace-Regional ficou acima das alternativas no AUC global.

Ao mesmo tempo, a tabela mostra uma ressalva importante. Em baixa falsa aceitacao, o AdaFace-Cosseno continua muito competitivo. Em TAR@FAR=0,01, ele fica em 0,218, enquanto a configuracao com Negativos Dificeis fica em 0,207. Portanto, AUC global maior nao elimina a necessidade de escolher o ponto de operacao de acordo com o custo dos erros.

## Slide 14 - Leitura estatistica e operacional (1:40)

A comparacao mais importante e contra o ViT-FaCoR, porque ele e a referencia supervisionada anterior mais forte na comparacao. A diferenca media em AUC entre o AdaFace-Regional com Negativos Dificeis e o ViT-FaCoR e de aproximadamente 0,030.

No teste t pareado sobre as cinco dobras, o resultado foi t(4)=10,264 com p=0,0005. Isso reforca a diferenca observada. Ja o Wilcoxon bilateral ficou em p=0,0625. Esse valor precisa ser interpretado com cuidado, porque com apenas cinco dobras o menor p-valor bilateral exato possivel ja e 0,0625 quando todas as diferencas tem o mesmo sinal.

Por isso, eu trato os testes como evidencia complementar. A leitura principal continua sendo media, desvio por dobra e comportamento no ponto operacional.

## Slide 15 - Ciclo experimental (1:30)

O ciclo de desenvolvimento ajuda a entender de onde veio o ganho.

Com AdaFace congelado, o modelo ficava em AUC de 0,746. Quando o ultimo estagio foi descongelado, subiu para 0,856. Esse foi o primeiro salto importante.

Depois, a passagem simetrica levou o AUC para 0,879 no ciclo de desenvolvimento e reduziu bastante o gap validacao-teste. Isso sugere que parte do erro anterior vinha de dependencia da ordem das imagens.

Por fim, os negativos dificeis reais melhoraram o regime de rejeicao de nao-parentes, chegando a 0,213 em TAR@FAR=0,01 no ciclo de desenvolvimento. A leitura e que eles tornam o treino mais exigente para a cabeca de decisao.

## Slide 16 - Analise por tipo de relacao (1:20)

A analise por relacao mostra que o resultado nao e uniforme.

Pais e filhos ficam geralmente entre 85% e 92% de acuracia. Irmaos ficam perto ou acima de 90%. O ponto mais relevante sao as relacoes de avos e netos. Essas relacoes eram o principal gargalo nas execucoes iniciais e, com a configuracao Simetrica, sobem para a faixa de 80% a 84%.

Mas ha um trade-off. A Simetrica favorece mais as classes positivas, enquanto Comparativa e Negativos Dificeis melhoram a classe non-kin, que sobe para 69,7%. Isso se conecta ao ponto operacional: rejeitar melhor nao-parentes pode custar desempenho em algumas relacoes positivas.

## Slide 17 - VLMs como linha de base externa (1:20)

Os VLMs foram avaliados em protocolo separado, como linha de base externa zero-shot. O Claude Sonnet atingiu AUC de 0,789 e o Codex VLM ficou em 0,624. O Claude chega perto da linha de base AdaFace-Cosseno, mas ainda fica abaixo. O Codex tem um problema forte de especificidade: ele aceita muitos non-kin como parentes.

Tambem foi feita uma comparacao controlada no mesmo manifesto de 6.000 pares usado pelo Claude. Nesse recorte, o M12 R012 atinge AUC de 0,888 contra 0,789 do Claude, e TAR@FAR=0,01 de 0,259 contra 0,042. Isso reforca que o VLM nao substitui um modelo supervisionado especializado. Mas esse resultado nao substitui a validacao cruzada principal; ele vale apenas para essa comparacao pareada no manifesto.

## Slide 18 - Limitacoes e etica (0:50)

As limitacoes principais sao: cinco dobras e uma semente principal, ausencia de transferencia para outros bancos, ausencia de auditoria de sobreposicao WebFace4M-FIW, falta de uma ablacao estrita e escopo mais restrito para VLMs.

Eticamente, o sistema so deve ser visto como triagem. Ele nao e prova de parentesco e nao deve ser usado sem decisao humana, DNA, documentos e auditoria de vieses demograficos.

## Slide 19 - Conclusao (0:50)

A conclusao principal e que, dentro do protocolo adotado, o AdaFace-Regional apresenta ganho em AUC global sobre a linha de base sem treinamento e sobre as arquiteturas supervisionadas avaliadas.

O melhor resultado principal foi 0,876 mais ou menos 0,003 de AUC, cerca de 0,030 acima do ViT-FaCoR. A proposta tambem melhora a recuperacao das relacoes de avos e netos nas variantes voltadas a classes positivas.

Ao mesmo tempo, o trabalho mostra que AUC maior nao resolve sozinho o ponto operacional. O AdaFace-Cosseno continua forte em baixa falsa aceitacao e a escolha entre variantes depende do custo de falso positivo e falso negativo.

Por fim, os VLMs ajudam a situar a tarefa, mas ainda nao dao evidencia de substituir modelos supervisionados especializados. Com isso, eu encerro a apresentacao e fico a disposicao para perguntas.

## Perguntas provaveis da banca

### Por que usar AUC como metrica principal?

Porque os modelos produzem escores em escalas diferentes e o AUC independe de um limiar especifico. Como o limiar pode mudar muito o resultado em verificacao, AUC permite comparar a ordenacao global. Mesmo assim, o trabalho tambem reporta TAR@FAR para pontos operacionais relevantes.

### Por que o AdaFace-Cosseno vence em alguns regimes de baixa falsa aceitacao?

Porque o AdaFace foi treinado para reconhecimento facial em larga escala e ja produz embeddings muito discriminativos. Em limiares conservadores, esse sinal de identidade separa parte dos pares. A contribuicao do AdaFace-Regional e melhorar a ordenacao global e a adaptacao ao FIW, nao vencer o AdaFace-Cosseno em todos os pontos possiveis.

### A comparacao com VLMs e justa?

Ela e informativa, mas nao equivalente. Os VLMs nao sao treinados no FIW, nao tem validacao cruzada e nao ajustam limiar por validacao. Por isso eles aparecem como linhas de base externas zero-shot, nao como competidores supervisionados diretos.

### O que os negativos dificeis realmente mostram?

Eles mostram que treinar com negativos semanticamente mais proximos, mas ainda de familias diferentes, ajuda a deslocar o modelo para maior rejeicao de nao-parentes. A conclusao deve ser limitada: a configuracao melhora o ponto operacional, mas nao prova isoladamente a contribuicao de cada modulo da arquitetura.

### O modelo poderia ser usado em aplicacao real?

Nao como decisor unico. O uso adequado seria apenas como triagem assistida, com decisao humana qualificada, base legal, consentimento quando aplicavel, registros de uso, direito de contestacao e validacao demografica antes de qualquer aplicacao real.

## Slide 1 — Capa — 0:30

> Bom dia.
>Eu sou Bruno Peixoto Martins e vou apresentar meu Trabalho de Conclusão de Curso, "Verificação Visual de Parentesco Facial Automática" que foi orientado pelo professor Ismael Santana.
>O trabalho investiga uma tarefa de visão computacional em que o sistema recebe duas imagens faciais e precisa estimar se existe parentesco biológico entre as pessoas.
>A apresentação está organizada como um estudo comparativo.
>Eu avalio modelos supervisionados de deep learning, uma proposta arquitetônica autoral chamada AdaFace-Regional e, de forma complementar, modelos multimodais generalistas usados em regime zero-shot.
>

## Slide 2 — Roteiro — 0:30

> A apresentação segue as seguinte etapas.
>Primeiro, eu defino o problema de verificação visual de parentesco, explico as diferenças em relação ao reconhecimento facial comum e mostro exemplos concretos do FIW.
>Em seguida, apresento as perguntas de pesquisa, os objetivos e as contribuições do trabalho.
>Depois entro na metodologia experimental, com o dataset, as dobras, a amostragem e as métricas usadas para comparação.
>A partir daí, detalho a proposta AdaFace-Regional, que é a contribuição arquitetônica principal, discuto os resultados no FIW, primeiro para os modelos supervisionados e depois para os VLMs como linhas de base externas.
>No final, fecho com limitações, implicações éticas e a conclusão direta sobre o que os resultados sustentam.

## Slide 3 — Problema — 1:10

> A tarefa estudada é uma verificação binária.
>O sistema recebe duas imagens de rosto e deve produzir um escore indicando se o par deve ser classificado como parentes ou non-parentes.
>O ponto central é que isso não é reconhecimento de identidade.
>Em reconhecimento facial, o objetivo típico é verificar se duas imagens mostram a mesma pessoa.
>Aqui, as pessoas são diferentes, e o que se busca é um sinal de semelhança familiar.
>Então o modelo precisa fazer um score prático, mesmo quando há diferença de idade, pose, iluminação ou expressão.
>No protocolo do trabalho, esse escore contínuo é convertido em decisão usando um limiar escolhido somente na validação e depois aplicado em teste.

## Slide 4 — Por que a tarefa é difícil — 1:15

> A dificuldade da tarefa vem, princípalmente, de duas direções opostas.
>Por um lado, duas pessoas sem parentesco podem ter rostos parecidos por coincidência, por características populacionais comuns ou por atributos visuais superficiais, como o famoso caso do Ed Sheeran e do príncipe Harry.
>Por outro lado, parentes reais podem ser visualmente diferentes.
>Isso acontece por diferença de idade, expressão, iluminação, pose, qualidade da imagem, diferença geracional e também por herança desigual de traços, como quando irmãos lembram mais lados diferentes da família.
>Essa ambiguidade torna o problema mais fino que o reconhecimento facial comum, porque não basta separar identidades.
>É preciso capturar um sinal fraco e indireto de parentesco.
>Além disso, em aplicações forenses ou humanitárias, um falso positivo pode gerar investigações, exames e deslocamentos desnecessários, aumentando os custos financeiros e o uso de recursos institucionais.
>Para as pessoas envolvidas, também pode criar falsas expectativas, ansiedade e sofrimento emocional ao indicar incorretamente um possível vínculo familiar.
>Por isso, a avaliação não pode depender apenas de acurácia.
>Eu uso AUC, Precisão média e TAR em taxas fixas de falsa aceitação.
>

## Slide 5 — Exemplos de pares no FIW — 0:30

> Aqui eu mostro dois exemplos retirados do próprio FIW.
>À esquerda está um par positivo, formado por pessoas da mesma família.
>À direita está um par negativo, formado por pessoas de famílias distintas.
>Esses exemplos  ajudam a mostrar que a decisão não é baseada em identidade, mas em um sinal visual de parentesco, que pode ser sutil e ambíguo.

## Slide 6 — Perguntas de pesquisa e objetivo — 1:05

> A pergunta principal do trabalho é se uma arquitetura supervisionada especializada, combinando AdaFace, descongelamento parcial e regiões anatômicas, apresenta ganho em relação a uma linha de base com treinamento e a arquiteturas supervisionadas alternativas.
>Essa pergunta é importante porque muitos resultados da literatura não são diretamente comparáveis, já que mudam dataset, partição, métrica e forma de escolher limiar.
>A pergunta secundária é exploratória e olha para os modelos multimodais generalistas como claude sonnet e gpt 5.5.
>A ideia é entender onde VLMs entram quando são usados em zero-shot, sem treino específico no FIW.
>O objetivo geral, então, foi comparar múltiplas abordagens sob o mesmo protocolo experimental no Families in the Wild e medir o ganho efetivo da proposta AdaFace-Regional.

## Slide 7 — Protocolo experimental — 1:45

> O benchmark principal é o Families in the Wild, usando a formulação do Track-I do RFIW, do smile lab.
>Esse banco é adequado porque contém famílias reais, múltiplos tipos de relação e pares positivos definidos a partir de vínculos familiares.
>O protocolo usa cinco dobras separadas por família para treino e validação, mantendo o teste oficial fixo.
>Essa separação por família é essencial para evitar vazamento.
>Se membros da mesma família aparecessem ao mesmo tempo em subconjuntos usados para ajustar e selecionar o modelo, o resultado poderia medir memorização familiar em vez de generalização.
>Em cada dobra, o limiar de decisão é escolhido apenas na validação, maximizando F1, e depois é aplicado no teste sem reotimização.
>A métrica principal é ROC AUC, porque ela avalia a ordenação dos escores e não depende de um limiar específico.
>As métricas com limiar continuam sendo reportadas, mas são lidas como pontos operacionais, não como único critério de escolha.

## Slide 8 — Amostragem e pré-processamento — 1:00

> A amostragem também foi padronizada, porque ela afeta diretamente a comparação.
>Os pares negativos são sempre formados por pessoas de famílias diferentes.
>Isso reduz a chance de marcar como não parentes duas pessoas que poderiam ter algum parentesco não anotado dentro da mesma família.
>Durante o treinamento, usei dois pares negativos para cada par positivo, o que aumenta a pressão para o modelo aprender a rejeitar falsos parentes.
>Na validação e no teste, a proporção é de um negativo para cada positivo, com listas fixas dentro de cada dobra.
>Assim, todos os modelos são avaliados sobre o mesmo conjunto de pares.
>No pré-processamento, as faces são alinhadas para 224 por 224 pixels.
>No AdaFace-Regional, além da face alinhada completa, cada imagem gera cinco recortes regionais de 112 por 112, correspondentes a regiões anatômicas usadas pela arquitetura.

## Slide 9 — Modelos comparados — 1:20

> A comparação principal reúne cinco famílias de modelo.
>O AdaFace-Cosseno é a linha de base mínima.
>Ele usa um extrator facial forte, já treinado para reconhecimento de identidade.
>O ViT-FaCoR representa a família de Transformadores Visuais com atenção cruzada entre faces, aproximando a comparação de patches das duas imagens.
>O DINOv2-LoRA testa uma hipótese diferente, usando um backbone auto-supervisionado amplo e adaptando parte do modelo com LoRA e atenção diferencial.
>O modelo com recuperação acrescenta uma galeria de pares positivos como suporte externo para a decisão.
>Por fim, o AdaFace-Regional é a proposta autoral deste trabalho.
>Os VLMs não aparecem nessa tabela principal porque eles não passam pelo mesmo processo de treino, validação cruzada e escolha de limiar.
>Eles são avaliados depois, como linhas de base externas zero-shot.
>Essa separação evita misturar uma comparação supervisionada controlada com uma avaliação exploratória por prompt.

## Slide 10 — Hipótese do AdaFace-Regional — 1:00

> A hipótese do AdaFace-Regional é intermediária entre dois extremos.
>De um lado, o AdaFace-Cosseno mostra que um backbone facial especializado já contém um sinal útil para parentesco, porque embeddings treinados para identidade carregam informação sobre estrutura facial.
>De outro lado, comparar apenas dois embeddings globais por cosseno limita o modelo, porque não há adaptação supervisionada ao FIW e não há mecanismo para decidir quais regiões do rosto são mais informativas para aquele par.
>A proposta tenta reorganizar esse sinal facial para a tarefa de parentesco.
>Para isso, ela adapta apenas parte do backbone, preservando boa parte do pré-treino, transforma regiões anatômicas em tokens explícitos, usa atenção cruzada entre as regiões das duas faces e reduz a dependência da ordem do par por meio de uma passagem simétrica.

## Slide 11 — AdaFace-Regional: fluxo — 1:50

> O fluxo do AdaFace-Regional começa com duas faces alinhadas, que eu chamo aqui de A e B.
>Cada uma passa pelo AdaFace IR-101 com descongelamento parcial do último estágio.
>O modelo extrai um embedding global da face inteira e também embeddings regionais.
>Essas regiões correspondem à face , olhos, nariz, boca e mandíbula.
>Em vez de comparar centenas de patches, como em um Transformador visual denso, a arquitetura compara um conjunto pequeno de tokens anatômicos.
>Os tokens regionais de A e B passam por atenção cruzada bidirecional, formando uma matriz de interação de 5 por 5.
>Depois, um gate sigmoide estima pesos para as regiões, permitindo que mais de uma parte da face contribua para a decisão.
>A cabeça final combina embeddings globais, diferenças, produtos elemento a elemento, similaridades regionais e pesos do gate.
>Na versão simétrica, o par é avaliado como A-B e B-A, e os logits são agregados antes do sigmoide.

## Slide 12 — Componentes da proposta — 1:15

> Cada componente da proposta responde a uma limitação específica.
>O descongelamento parcial permite adaptar o AdaFace à tarefa de parentesco sem reescrever todos os pesos do backbone e sem depender apenas de uma cabeça rasa.
>As regiões anatômicas introduzem um prior estrutural simples, porque o modelo passa a comparar partes interpretáveis da face, e não qualquer região da imagem com a mesma liberdade.
>A atenção cruzada deixa uma face influenciar a representação da outra durante a comparação.
>O gate sigmoide permite pesar várias regiões ao mesmo tempo, em vez de escolher uma única parte do rosto.
>A cabeça auxiliar de relação usa os pares positivos para regularizar a representação, tentando preservar informação sobre tipos de vínculo.
>Por fim, a passagem simétrica incorpora uma propriedade natural da tarefa: Se A é parente de B, então B também é parente de A, e a decisão ideal não deveria depender da ordem das imagens.

## Slide 13 — Treinamento e avaliação — 0:45

> Aqui eu quero destacar principalmente como mantive a comparação controlada entre modelos.
>No treinamento, usei uma configuração estável, com AdamW, lote 16, precisão mista e parada antecipada, para não continuar ajustando o modelo depois que a validação deixasse de melhorar.
>Como as arquiteturas produzem escores de formas diferentes, eu não comparo diretamente o valor bruto de saída: nos modelos de embedding uso o cosseno reescalado, e nos modelos com cabeça binária uso o sigmoide do logit.
>Em cada dobra, o limiar é escolhido apenas na validação e depois aplicado ao teste sem nenhum novo ajuste.
>Por fim, os testes t pareado  entram apenas como evidência complementar aos resultados das cinco dobras.

## Slide 14 — Resultado principal em FIW — 2:00

> Esta é a tabela principal de resultados no FIW.
>O resultado mais importante é que o AdaFace-Regional com negativos difíceis alcança o maior AUC médio, com 0,876 e desvio de 0,003.
>As três variantes finalistas do AdaFace-Regional ficam próximas em AUC, o que sugere que o conjunto de componentes da proposta é robusto, mas que pequenas escolhas de fusão e amostragem deslocam o ponto de operação.
>A conclusão é que ela melhora a ordenação global dos pares, medida por AUC, dentro do protocolo adotado.
>Por isso, a leitura da tabela precisa combinar duas dimensões.
>O AUC responde se o ranking geral melhorou, enquanto o TAR@FAR responde quanto o modelo recupera quando a falsa aceitação é limitada.

## Slide 15 — O que o resultado indica — 1:15

> A comparação mais relevante é contra o ViT-FaCoR, porque ele é a referência supervisionada mais forte entre as arquiteturas alternativas avaliadas.
>A diferença média é de aproximadamente 0,030 em AUC, e essa diferença é maior que a variação média observada entre dobras.
>O teste t pareado resulta em p igual a 0,0005, indicando evidência estatística forte nesse recorte.
>A leitura segura é que, dentro do protocolo adotado, há evidência forte de ganho em ordenação global. Ao mesmo tempo, a escolha operacional continua dependendo do custo relativo dos erros, porque AUC alto não decide sozinho qual falsa aceitação é aceitável.

## Slide 16 — Ciclo de desenvolvimento — 1:05

> O ciclo de desenvolvimento ajuda a entender de onde veio a configuração final.
>Ele não deve ser lido como uma ablação perfeita, porque as versões foram construídas em sequência, mas ele mostra quais intervenções mais mudaram o comportamento do modelo.
>Com o AdaFace congelado, o AUC era 0,746.
>Ao liberar parcialmente o backbone, o AUC subiu para 0,856, que é o primeiro salto relevante.
>Isso sugere que o pré-treino facial é útil, mas precisa de adaptação ao domínio de parentesco.
>A passagem simétrica foi outro salto importante, chegando a 0,879 no ciclo de desenvolvimento e reduzindo o gap entre validação e teste.
>A fusão comparativa e os negativos difíceis reais ajudaram principalmente no ponto de operação, em especial na rejeição de não-parentes.

## Slide 17 — Análise por tipo de relação — 1:20

> A análise por tipo de relação mostra que o ganho não é perfeitamente uniforme.
>Relações entre pais e filhos ficam em um patamar alto, geralmente entre 85 e 92 por cento de acurácia, dependendo da configuração.
>Irmãos também aparecem como um grupo mais estável, próximo ou acima de 90 por cento em várias execuções.
>O grupo mais sensível no caso foram o de avós e netos, por duas razões.
>Primeiro, há menos pares no teste, o que aumenta a instabilidade estatística.
>Segundo, a diferença geracional torna a semelhança facial menos direta. A idade avançada gera muitas mudanças na estrutura e textura facial de uma pessoa.
>A configuração simétrica melhora várias classes positivas, inclusive relações de avós e netos.
>Já as versões comparativa e com negativos difíceis tendem a melhorar a rejeição de parentes.

## Slide 18 — Linhas de base com VLMs — 1:25

> Os VLMs foram avaliados como linhas de base externas, não como parte da validação cruzada principal.
>Cada modelo recebeu 6.000 pares balanceados, com 3.000 positivos e 3.000 negativos, em regime zero-shot.
>O Claude alcança AUC de 0,789, próximo do AdaFace-Cosseno, mas ainda abaixo da proposta supervisionada.
>O Codex fica em 0,624 e apresenta baixa especificidade, classificando muitos pares nao parentes como parentes.
>O ponto mais crítico aparece em TAR@FAR igual a 0,01 onde os VLMs apresentão maior rigidez do modelo vista pelo limiar baixo .
>Para uma comparação mais controlada, o AdaFace-Regional  também foi executado no mesmo manifesto usado pelas VLMs.
>Nesse subconjunto, ele chega a 0,888 de AUC e 0,259 em TAR@0,01.
>Isso indica que os VLMs capturam algum sinal visual de parentesco, mas a confiança produzida por eles não ordena bem os pares quando a falsa aceitação precisa ser baixa.
>Em outras palavras, o problema não é apenas responder kin ou non-kin.
>O problema também é calibrar a confiança de modo útil para uma curva de verificação.

## Slide 19 — Contribuições — 0:45

> As contribuições do trabalho podem ser resumidas em quatro pontos.
>A primeira é um protocolo comparativo uniforme para cinco abordagens de verificação binária no FIW, com separação por família, limiar escolhido em validação e métricas comparáveis.
>A segunda é a proposta AdaFace-Regional, que combina backbone facial especializado, descongelamento parcial, tokens anatômicos, atenção cruzada regional, gate, cabeça auxiliar e passagem simétrica.
>A terceira é a análise por tipo de relação, que mostra que o desempenho não é homogêneo e que relações de avós e netos continuam sendo um recorte importante.
>A quarta é a avaliação binária dos VLMs zero-shot, que ajuda a situar o limite da inferência multimodal genérica nessa tarefa.
>Em conjunto, essas contribuições procuram separar o que é ganho arquitetônico, o que é efeito de protocolo e o que é apenas desempenho exploratório de modelos generalistas.

## Slide 20 — Limitações e implicações éticas — 1:20

> As limitações são importantes para evitar uma interpretação excessiva dos resultados:
>A avaliação principal está concentrada no FIW, então não é possível afirmar automaticamente que o mesmo desempenho se transfere para outros bancos, outras populações ou imagens coletadas em condições diferentes.
>Também não há uma ablação estrita separando todos os módulos da proposta.
>Por isso, a conclusão é sobre o conjunto arquitetônico avaliado, e não sobre a contribuição isolada de cada componente.
>Além disso, não foi feita auditoria demográfica nem análise cross-dataset.
>Do ponto de vista ético, o modelo deve ser entendido apenas como triagem ou apoio à decisão humana qualificada.
>Ele não substitui DNA, documentação, investigação especializada ou consentimento adequado.
>Falsos positivos e falsos negativos podem ter custo alto, especialmente em contextos forenses e humanitários. A consequência prática é que qualquer uso real exigiria governança, rastreabilidade da decisão, análise de viés e possibilidade de contestação.

## Slide 21 — Conclusão — 1:00

> Concluindo, o AdaFace-Regional com negativos difíceis obteve o melhor AUC no protocolo principal, com 0,876 e desvio de 0,003.
>O ganho sobre o ViT-FaCoR é de aproximadamente 0,030 em AUC, o que sustenta a contribuição da proposta dentro do protocolo adotado.
>A leitura mais importante é que combinar um backbone facial especializado com adaptação parcial, regiões anatômicas e passagem simétrica melhora a ordenação global dos pares no FIW.
>Ao mesmo tempo, os resultados mostram que o ponto operacional continua decisivo.
>Os VLMs zero-shot ficaram abaixo dos modelos supervisionados especializados, principalmente em TAR@FAR baixo, indicando que inferência multimodal genérica ainda não substitui treinamento específico nessa tarefa.
>Como próximos passos, eu destacaria múltiplas sementes, ablações estritas, auditoria demográfica e calibração de limiar para uso operacional.
>Portanto, a resposta final é positiva, mas delimitada.
>A proposta melhora o ranking global no FIW, porém não transforma verificação visual de parentesco em evidência conclusiva de vínculo biológico.

## Slide 22 — Obrigado / perguntas — 0:10

> Obrigado pela atenção.
>Fico à disposição para perguntas e comentários da banca.

## Tempo total

Somando os tempos acima: aproximadamente 24 minutos e 25 segundos.

# Perguntas prováveis da banca e respostas

Use este arquivo como material de preparação. Cada pergunta tem uma resposta curta, para uso direto na defesa, e uma resposta expandida, para estudo.

## Índice rápido por tema

- **Problema e escopo**: P01 a P05
- **Dataset e protocolo**: P06 a P11
- **Métricas e estatística**: P12 a P17
- **AdaFace-Regional**: P18 a P27
- **Resultados e interpretação**: P28 a P36
- **VLMs**: P37 a P42
- **Limitações, ética e trabalhos futuros**: P43 a P50

---

## Problema e Escopo

### P01 — Qual é exatamente o problema resolvido pelo trabalho?

**Resposta curta:**
O trabalho resolve verificação visual binária de parentesco facial. Dado um par de faces, o modelo produz um escore para decidir se há parentesco biológico (`kin`) ou não (`non-kin`).

**Resposta expandida:**
A tarefa não é identificar quem são as pessoas, nem classificar a relação exata entre elas como pai-filho ou avô-neto. A comparação principal é binária. O modelo recebe duas faces e deve ordenar pares positivos acima de pares negativos. A decisão final depende de um limiar escolhido na validação e aplicado no teste.

---

### P02 — Por que isso não é apenas reconhecimento facial comum?

**Resposta curta:**
Porque reconhecimento facial compara identidade, enquanto parentesco compara semelhança familiar entre pessoas diferentes.

**Resposta expandida:**
Em reconhecimento facial, duas imagens positivas normalmente mostram a mesma pessoa. Neste trabalho, um par positivo mostra duas pessoas diferentes que têm vínculo biológico. O sinal visual é mais fraco e indireto, pois parentes podem ser diferentes e não-parentes podem parecer semelhantes. Por isso, um bom backbone de identidade ajuda, mas não resolve sozinho toda a tarefa.

---

### P03 — O trabalho detecta qualquer tipo de parentesco?

**Resposta curta:**
Não. O escopo é parentesco consanguíneo inferido por imagem facial.

**Resposta expandida:**
O trabalho exclui vínculos sociais sem base genética, como adoção ou convivência, porque o sinal explorado pelo modelo é a semelhança facial herdada. Também não usa DNA, documentos ou registros oficiais. A saída do modelo é apenas um indício visual, não uma prova de parentesco.

---

### P04 — Por que a tarefa é difícil?

**Resposta curta:**
Porque há ambiguidade nos dois sentidos. Parentes podem ser visualmente diferentes, e não-parentes podem ser parecidos.

**Resposta expandida:**
A dificuldade vem de idade, pose, iluminação, expressão, qualidade da imagem e diferença geracional. Além disso, irmãos podem herdar traços diferentes de cada lado da família. No sentido oposto, pessoas sem vínculo biológico podem compartilhar aparência semelhante por coincidência. Isso torna a tarefa mais sensível que uma classificação facial comum.

---

### P05 — Qual é a principal pergunta de pesquisa?

**Resposta curta:**
Se uma arquitetura supervisionada especializada, baseada em AdaFace, descongelamento parcial e regiões anatômicas, melhora a verificação binária no FIW em relação às alternativas avaliadas.

**Resposta expandida:**
A pergunta principal é se o AdaFace-Regional apresenta ganho sobre uma linha de base sem treinamento e sobre arquiteturas supervisionadas alternativas sob o mesmo protocolo. A pergunta secundária é exploratória e avalia como VLMs generalistas se comportam em zero-shot na mesma formulação binária.

---

## Dataset e Protocolo

### P06 — Por que o FIW foi escolhido como benchmark principal?

**Resposta curta:**
Porque o Families in the Wild é o maior banco público de parentesco facial e possui múltiplas famílias e tipos de relação.

**Resposta expandida:**
O FIW é mais realista e variado que bancos pequenos como KinFaceW-I. Ele contém 761 famílias e onze relações positivas no Track-I do RFIW. Isso permite avaliar a tarefa em condições mais heterogêneas de idade, pose, iluminação e qualidade de imagem.

---

### P07 — Por que KinFaceW-I não entrou na comparação principal?

**Resposta curta:**
Porque ele foi usado como referência histórica e prototipagem, mas o FIW é mais amplo e adequado para a comparação final.

**Resposta expandida:**
KinFaceW-I tem cerca de 1.066 pares e apenas quatro relações, com imagens mais controladas. Ele é útil para testes rápidos e contextualização histórica, mas a comparação principal precisa de um banco maior, com mais relações e maior variabilidade. Por isso, o resultado principal fica restrito ao FIW.

---

### P08 — Como o protocolo evita vazamento entre treino, validação e teste?

**Resposta curta:**
As dobras são separadas por família, e o teste oficial é preservado.

**Resposta expandida:**
Membros da mesma família não aparecem simultaneamente em subconjuntos usados para ajustar e selecionar o modelo. Isso evita que o modelo memorize características familiares presentes tanto em treino quanto em validação. Além disso, o limiar é escolhido apenas na validação e aplicado no teste sem reotimização.

---

### P09 — Por que usar validação cruzada de cinco dobras?

**Resposta curta:**
Para reduzir a dependência de uma única partição de treino e validação e estimar variação entre dobras.

**Resposta expandida:**
Cada dobra usa uma divisão diferente de famílias para treino e validação, mantendo o teste oficial fixo. Isso fornece cinco medidas por modelo, permitindo média, desvio padrão e uma comparação pareada. A limitação é que cinco observações ainda são poucas para uma inferência estatística forte.

---

### P10 — Como os pares negativos foram construídos?

**Resposta curta:**
Os pares negativos são formados por pessoas de famílias diferentes.

**Resposta expandida:**
Essa escolha reduz a chance de rotular como `non-kin` duas pessoas que poderiam ter algum parentesco não anotado dentro da mesma família. No treino, foram usados dois negativos para cada positivo. Na validação e no teste, a proporção é de um negativo para cada positivo, com listas fixas dentro de cada dobra.

---

### P11 — O que são negativos difíceis?

**Resposta curta:**
São pares negativos de famílias diferentes que preservam os mesmos papéis familiares do par positivo de referência.

**Resposta expandida:**
Por exemplo, para um par positivo pai-filho, um negativo difícil pode ser formado por um pai e um filho de famílias diferentes. Isso torna o par negativo semanticamente mais próximo do positivo sem violar a separação por família. A configuração final usa 30% de negativos difíceis reais no treinamento.

---

## Métricas e Estatística

### P12 — Por que ROC AUC é a métrica principal?

**Resposta curta:**
Porque avalia a ordenação dos escores sem depender de um limiar específico.

**Resposta expandida:**
Os modelos produzem escores em escalas diferentes. Alguns usam cosseno reescalado, outros usam sigmoide de logits. O AUC permite comparar se o modelo tende a atribuir escores maiores a pares positivos do que a negativos, independentemente do ponto de decisão escolhido.

---

### P13 — Por que não usar apenas acurácia?

**Resposta curta:**
Porque acurácia depende do limiar e pode esconder erros relevantes em regimes de baixa falsa aceitação.

**Resposta expandida:**
Em aplicações forenses ou humanitárias, falso positivo pode ter custo alto. Um modelo pode ter boa acurácia em um limiar, mas ser ruim quando exigimos baixa taxa de falsa aceitação. Por isso, o trabalho reporta AUC, Average Precision e TAR@FAR, além das métricas com limiar.

---

### P14 — O que significa TAR@FAR?

**Resposta curta:**
É a taxa de aceitação verdadeira quando a taxa de falsa aceitação é fixada em um valor específico.

**Resposta expandida:**
FAR mede a proporção de pares negativos aceitos indevidamente como parentes. TAR mede a proporção de pares positivos aceitos corretamente. Em TAR@FAR=0,01, por exemplo, o limiar é ajustado para permitir 1% de falsa aceitação e mede-se quantos positivos ainda são recuperados nesse regime.

---

### P15 — Como o limiar de decisão foi escolhido?

**Resposta curta:**
O limiar foi escolhido na validação de cada dobra, maximizando F1, e aplicado no teste sem reotimização.

**Resposta expandida:**
Essa regra evita escolher o limiar olhando o teste. Como cada modelo pode produzir escores em escalas diferentes, cada um passa pela mesma busca de limiar dentro da própria dobra. As métricas independentes de limiar, como AUC e AP, complementam essa avaliação.

---

### P16 — O teste t com cinco dobras é confiável?

**Resposta curta:**
Ele oferece evidência complementar, mas não deve sustentar sozinho uma conclusão forte.

**Resposta expandida:**
No teste t pareado entre AdaFace-Regional com Negativos Difíceis e ViT-FaCoR, o p-valor foi 0,0005. Entretanto, há apenas cinco diferenças pareadas, e a normalidade dessas diferenças é difícil de verificar. Por isso, o resultado deve ser interpretado junto com a diferença média de AUC, o desvio padrão e a consistência da direção do ganho entre as dobras. A conclusão permanece restrita ao protocolo adotado.

---

### P17 — O resultado é estatisticamente suficiente?

**Resposta curta:**
Ele sustenta ganho em AUC dentro do protocolo adotado, mas não elimina a necessidade de repetir com múltiplas sementes e outros bancos.

**Resposta expandida:**
A evidência é forte para o recorte avaliado, especialmente contra o ViT-FaCoR em AUC. Porém, cinco dobras ainda são poucas para uma conclusão universal. A tese correta é delimitada: dentro do protocolo FIW adotado, o AdaFace-Regional apresenta ganho em AUC global sobre as alternativas avaliadas.

---

## AdaFace-Regional

### P18 — Qual é a contribuição arquitetônica principal?

**Resposta curta:**
O AdaFace-Regional combina backbone facial especializado, tokens anatômicos, atenção cruzada regional, gate, cabeça auxiliar e passagem simétrica.

**Resposta expandida:**
A contribuição não é apenas trocar o backbone. A proposta reorganiza o sinal facial do AdaFace para a tarefa de parentesco. Ela extrai embeddings globais e regionais, compara regiões entre faces, pondera regiões com gate e reduz dependência da ordem do par com inferência simétrica.

---

### P19 — Por que usar AdaFace como backbone?

**Resposta curta:**
Porque ele já é um extrator facial forte, treinado para reconhecimento de identidade, e seus embeddings carregam sinal útil de estrutura facial.

**Resposta expandida:**
O AdaFace-Cosseno, mesmo sem treino específico, teve desempenho competitivo. Isso indica que o pré-treino facial contém informação relevante. O AdaFace-Regional parte desse sinal e tenta adaptá-lo à tarefa de parentesco sem descartar completamente o pré-treino.

---

### P20 — Por que não descongelar o AdaFace inteiro?

**Resposta curta:**
Para adaptar o modelo ao parentesco sem destruir representações faciais já aprendidas.

**Resposta expandida:**
O conjunto de parentesco é muito menor que bases de reconhecimento facial usadas no pré-treino. Descongelar tudo aumentaria o risco de sobreajuste e perda das representações de identidade. Por isso, os estágios iniciais ficam congelados e o ajuste se concentra no último estágio e na camada de saída.

---

### P21 — Por que usar regiões anatômicas fixas?

**Resposta curta:**
Para orientar a comparação para partes interpretáveis da face, em vez de deixar o modelo comparar qualquer patch da imagem.

**Resposta expandida:**
Modelos com muitos patches podem usar regiões pouco informativas, como fundo, cabelo ou artefatos de iluminação. As regiões anatômicas impõem um prior simples: face global, olhos, nariz, boca e mandíbula. Isso reduz a complexidade da atenção e aproxima a arquitetura da forma visual como o problema é descrito.

---

### P22 — Qual é a limitação das regiões fixas?

**Resposta curta:**
Elas dependem da qualidade do alinhamento facial.

**Resposta expandida:**
Se a face for mal detectada, estiver em pose extrema ou se o recorte estiver deslocado, as regiões deixam de corresponder às partes anatômicas esperadas. Essa é uma limitação clara da proposta. Um trabalho futuro poderia usar landmarks mais detalhados ou regiões aprendidas dinamicamente.

---

### P23 — Para que serve a atenção cruzada regional?

**Resposta curta:**
Ela permite comparar regiões de uma face com regiões da outra.

**Resposta expandida:**
Em vez de comparar só embeddings globais, a atenção cruzada estima interações entre tokens regionais das duas imagens. Como são cinco regiões por face, a comparação é pequena e interpretável, com uma matriz 5x5. Isso reduz a liberdade em relação a atenção densa sobre centenas de patches.

---

### P24 — Para que serve o gate sigmoide?

**Resposta curta:**
Para permitir que o modelo pese diferentes regiões do rosto na decisão.

**Resposta expandida:**
O gate não força a escolha de uma única região. Como é sigmoide, mais de uma região pode receber peso alto ao mesmo tempo. Isso é adequado para parentesco, pois o sinal pode aparecer em combinações de traços, não em uma única parte da face.

---

### P25 — Por que usar passagem simétrica?

**Resposta curta:**
Porque parentesco é simétrico. Se A é parente de B, B é parente de A.

**Resposta expandida:**
Arquiteturas que recebem pares concatenados ou usam atenção direcional podem aprender atalhos ligados à posição da imagem no par. A passagem simétrica avalia o par nas duas ordens, A-B e B-A, e agrega os logits. Isso reduz dependência da ordem e foi uma das intervenções com maior impacto no ciclo experimental.

---

### P26 — A cabeça auxiliar de relação não mistura a tarefa binária com multiclasse?

**Resposta curta:**
Ela é auxiliar e atua como regularização nos pares positivos; a decisão principal continua binária.

**Resposta expandida:**
A cabeça de relação tenta preservar informação sobre o tipo de vínculo nos pares positivos. Ela não substitui a cabeça kin/non-kin. A avaliação principal continua binária, e a cabeça auxiliar é usada apenas como sinal adicional de treinamento.

---

### P27 — A proposta é uma ablação completa dos componentes?

**Resposta curta:**
Não. As variantes formam um ciclo de desenvolvimento, não uma ablação estrita.

**Resposta expandida:**
O trabalho mostra que o conjunto de componentes está associado ao ganho, especialmente descongelamento parcial, simetrização e negativos difíceis. Porém, não isola rigorosamente cada módulo em todas as combinações possíveis. Isso é uma limitação reconhecida e um caminho claro para trabalhos futuros.

---

## Resultados e Interpretação

### P28 — Qual é o principal resultado quantitativo?

**Resposta curta:**
O AdaFace-Regional com Negativos Difíceis obteve AUC de 0,876 ± 0,003, o melhor valor da comparação principal.

**Resposta expandida:**
Esse resultado fica acima do ViT-FaCoR, com 0,846 ± 0,004; do DINOv2-LoRA, com 0,814 ± 0,007; do AdaFace-Cosseno, com 0,799; e do Modelo com Recuperação, com 0,778 ± 0,008. A conclusão principal é ganho em ordenação global no protocolo FIW adotado.

---

### P29 — Se o AdaFace-Cosseno é melhor em TAR@FAR=0,01, por que a proposta é melhor?

**Resposta curta:**
Porque a proposta é melhor em AUC global, mas o AdaFace-Cosseno permanece competitivo em baixa falsa aceitação.

**Resposta expandida:**
Essas métricas medem aspectos diferentes. AUC mede a ordenação geral dos pares em todos os limiares. TAR@FAR=0,01 mede um ponto operacional específico e conservador. O resultado mostra que o AdaFace-Regional melhora o ranking global, mas não domina todos os pontos de operação. Essa ressalva está explicitamente reconhecida.

---

### P30 — Como interpretar o ganho de 0,030 sobre o ViT-FaCoR?

**Resposta curta:**
É um ganho relevante em AUC dentro do protocolo, maior que a variação média entre dobras.

**Resposta expandida:**
O ViT-FaCoR é a referência supervisionada mais forte entre as alternativas. A diferença média de aproximadamente 0,030 em AUC é consistente entre as dobras e apoiada pelo teste t pareado. Ainda assim, a interpretação deve ficar restrita ao protocolo usado.

---

### P31 — Por que o DINOv2-LoRA não superou ViT-FaCoR?

**Resposta curta:**
Porque a representação auto-supervisionada ampla não foi suficiente para substituir uma arquitetura ajustada para comparação entre faces.

**Resposta expandida:**
O DINOv2-LoRA melhora em AUC quando há descongelamento completo, mas não alcança ViT-FaCoR. Isso sugere que o pré-treino geral do DINOv2, mesmo adaptado, não captura tão diretamente o sinal facial especializado necessário para parentesco. O resultado também mostra que liberar mais parâmetros não melhora todos os pontos operacionais.

---

### P32 — Por que o Modelo com Recuperação ficou abaixo?

**Resposta curta:**
Porque recuperar pares positivos parecidos não garantiu suportes suficientemente informativos para melhorar o ranking global.

**Resposta expandida:**
A recuperação depende da qualidade da assinatura do par, da galeria e da capacidade de recuperar exemplos úteis. No experimento, essa estratégia ficou abaixo da linha de base AdaFace-Cosseno. Isso indica que memória externa de positivos, por si só, não resolveu a dificuldade de separar parentes e não-parentes no FIW.

---

### P33 — O ganho está nas regiões ou no AdaFace?

**Resposta curta:**
O resultado seguro é que o ganho está associado ao conjunto: AdaFace adaptado, regiões, atenção, gate, cabeça auxiliar e simetria.

**Resposta expandida:**
O AdaFace sozinho já é forte, como mostra o AdaFace-Cosseno. A proposta melhora AUC ao adaptar esse sinal e reorganizar a comparação. Como não há ablação estrita de todos os componentes, não é correto atribuir todo o ganho a um módulo isolado.

---

### P34 — Por que as relações de avós e netos são importantes?

**Resposta curta:**
Porque são classes menores, mais instáveis e com maior diferença geracional.

**Resposta expandida:**
No FIW, relações de avós e netos têm menos pares no teste e envolvem diferença de idade maior. Isso torna a semelhança facial menos direta. O AdaFace-Regional, especialmente a versão Simétrica, melhora essas classes em relação às execuções iniciais, mas elas continuam sendo um recorte que exige cautela.

---

### P35 — A configuração com Negativos Difíceis é sempre a melhor?

**Resposta curta:**
Não. Ela é a escolha final por AUC e por alguns regimes conservadores, mas há trade-off com classes positivas.

**Resposta expandida:**
As variantes Simétrica, Comparativa e Negativos Difíceis têm AUC muito próxima. A Simétrica é melhor em várias classes positivas, enquanto Comparativa e Negativos Difíceis melhoram non-kin e alguns pontos de baixa falsa aceitação. A melhor escolha depende do custo do falso positivo e do falso negativo.

---

### P36 — O resultado pode ser considerado estado da arte?

**Resposta curta:**
Eu evitaria afirmar isso, porque protocolos da literatura variam muito.

**Resposta expandida:**
O trabalho foi desenhado para comparação controlada interna entre modelos sob o mesmo protocolo. Como outros artigos podem usar partições, limiares, métricas e versões diferentes do FIW, comparar percentuais isolados pode ser inadequado. A afirmação segura é que, no protocolo adotado, a proposta supera as alternativas avaliadas.

---

## VLMs

### P37 — Por que avaliar VLMs?

**Resposta curta:**
Porque eles são uma linha de base externa interessante para medir o que um modelo multimodal generalista consegue inferir sem treino específico.

**Resposta expandida:**
Modelos como Claude e Codex conseguem responder perguntas sobre imagens por linguagem natural. A avaliação testa se essa capacidade é suficiente para parentesco facial em zero-shot. O resultado mostra que há algum sinal, mas não o bastante para substituir modelos supervisionados especializados.

---

### P38 — Por que os VLMs não entram na comparação principal?

**Resposta curta:**
Porque eles não seguem o mesmo protocolo supervisionado de treino, validação cruzada e seleção de limiar.

**Resposta expandida:**
Os VLMs são avaliados sem treino no FIW, sem validação própria e sem ajuste de pesos. Eles usam uma amostra balanceada de 6.000 pares, não o mesmo protocolo completo de cinco dobras. Por isso, aparecem como linhas de base externas, não como competidores equivalentes.

---

### P39 — Como a confiança dos VLMs vira escore?

**Resposta curta:**
A decisão kin/non-kin e a confiança são mapeadas para um escore contínuo entre 0 e 1.

**Resposta expandida:**
Se o VLM responde kin com confiança `c`, o escore fica acima de 0,5. Se responde non-kin, fica abaixo de 0,5. Isso permite calcular AUC, Average Precision e TAR@FAR. A limitação é que a confiança do VLM não é uma probabilidade calibrada por validação.

---

### P40 — O que os resultados dos VLMs mostram?

**Resposta curta:**
Eles capturam algum sinal visual, mas erram muito em non-kin e têm baixa TAR@FAR em regime conservador.

**Resposta expandida:**
Claude teve AUC de 0,789, próximo do AdaFace-Cosseno, mas TAR@FAR=0,01 de apenas 0,042. Codex teve AUC de 0,624 e especificidade baixa. Isso indica dificuldade em rejeitar pares não aparentados com semelhança visual fortuita.

---

### P41 — Por que comparar o AdaFace-Regional R012 no manifesto do Claude?

**Resposta curta:**
Para controlar o conjunto de pares na comparação com o VLM mais forte.

**Resposta expandida:**
Como Claude foi avaliado em um manifesto específico de 6.000 pares, executar o AdaFace-Regional nesse mesmo manifesto permite uma comparação direta naquele subconjunto. O resultado favorece o modelo supervisionado, mas não substitui a validação cruzada principal nem o teste oficial completo.

---

### P42 — Os VLMs podem substituir modelos supervisionados no futuro?

**Resposta curta:**
Possivelmente podem melhorar, mas os resultados atuais não sustentam substituição.

**Resposta expandida:**
O principal problema é a calibração e a ordenação fina dos escores. Mesmo quando o VLM acerta muitos positivos, ele não controla bem falsa aceitação. Para uso em verificação biométrica, isso é crítico. Modelos futuros poderiam melhorar com fine-tuning, calibração ou prompts mais estruturados, mas isso já mudaria o protocolo zero-shot.

---

## Limitações, Ética e Trabalhos Futuros

### P43 — Qual é a maior limitação metodológica do trabalho?

**Resposta curta:**
A ausência de ablação estrita e de múltiplas sementes por dobra.

**Resposta expandida:**
O trabalho mostra ganho do conjunto arquitetônico, mas não mede isoladamente todos os componentes. Além disso, cinco dobras reduzem dependência de uma única partição, mas múltiplas sementes por dobra dariam uma estimativa melhor de variabilidade.

---

### P44 — O modelo generaliza para fora do FIW?

**Resposta curta:**
Isso ainda não foi demonstrado.

**Resposta expandida:**
O resultado principal é restrito ao FIW. Não foram feitos testes cross-dataset, em bases demograficamente diferentes ou em imagens operacionais reais. Um próximo passo importante seria avaliar transferência para outros bancos e condições de captura.

---

### P45 — Há risco de sobreposição entre WebFace4M e FIW?

**Resposta curta:**
É um risco metodológico não auditado.

**Resposta expandida:**
O AdaFace foi pré-treinado no WebFace4M, e o trabalho não fez uma auditoria completa de sobreposição com identidades ou imagens do FIW. Não há evidência local de vazamento, mas a possibilidade precisa ser reconhecida. Uma auditoria de identidade/imagem seria necessária para fortalecer a conclusão.

---

### P46 — O trabalho avalia viés demográfico?

**Resposta curta:**
Não. Essa é uma limitação importante.

**Resposta expandida:**
O trabalho avalia desempenho global e por tipo de relação, mas não por grupo demográfico. Como modelos faciais podem ter diferenças de desempenho por idade, gênero, raça ou qualidade de imagem, uma aplicação real exigiria auditoria demográfica antes de qualquer uso operacional.

---

### P47 — Esse sistema poderia ser usado em contexto forense?

**Resposta curta:**
Só como triagem ou apoio, nunca como prova isolada.

**Resposta expandida:**
O modelo pode ajudar a reduzir espaço de busca ou priorizar casos, mas a decisão final deve depender de DNA, documentação, investigação especializada e avaliação humana qualificada. Falsos positivos e falsos negativos podem causar danos graves, especialmente em populações vulneráveis.

---

### P48 — Qual erro é mais grave: falso positivo ou falso negativo?

**Resposta curta:**
Depende da aplicação, mas em contexto forense e humanitário o falso positivo tende a ser especialmente sensível.

**Resposta expandida:**
Um falso positivo pode associar pessoas sem vínculo biológico, afetando investigações, reunificação familiar ou decisões institucionais. Um falso negativo também é ruim, pois pode deixar de sugerir um vínculo real. Por isso, o trabalho reporta métricas em diferentes FARs e não defende um único ponto operacional universal.

---

### P49 — Quais seriam os próximos passos técnicos?

**Resposta curta:**
Múltiplas sementes, ablações estritas, testes cross-dataset, auditoria demográfica e calibração de limiar.

**Resposta expandida:**
Também seria útil testar regiões aprendidas por landmarks mais ricos, avaliar robustez a desalinhamento, comparar mais backbones faciais, calibrar probabilidades e estudar estratégias de negativos difíceis mais sistemáticas. Para VLMs, seria interessante avaliar prompts, calibração e fine-tuning, deixando claro quando o regime deixa de ser zero-shot.

---

### P50 — Qual é a conclusão em uma frase?

**Resposta curta:**
Dentro do protocolo adotado no FIW, o AdaFace-Regional melhora a ordenação global dos pares em relação às alternativas avaliadas, mas não deve ser interpretado como prova automática de parentesco.

**Resposta expandida:**
O trabalho sustenta ganho em AUC global para a proposta, mostra limites dos VLMs zero-shot e explicita que o ponto operacional continua decisivo. A contribuição é metodológica e arquitetônica, mas a aplicação real exige validação externa, auditoria de viés, governança e confirmação por evidências independentes.

# Notas — decisões e desvios do planejamento

Registro das escolhas que o planejamento não fixa e dos desvios dele. Os números dos resultados estão nos
arquivos citados, não aqui.

## Fase 1

- O baseline do EDA roda só no conjunto de desenvolvimento (80%), para nenhum modelo ver o hold-out antes
  do fim.

## Fase 2

1. **Busca do MLP com `RandomizedSearchCV(n_iter=40, random_state=42)`** em vez de `GridSearchCV`
   (troca prevista no planejamento para buscas acima de ~15 min). Medição prévia, com 8 processos, de 20
   configurações da grade (`alpha=1e-3`, `learning_rate_init=1e-3`, ajuste em 960 vítimas): 3,3 s por
   ajuste em média na regressão e 7,7 s na classificação, com configurações `tanh` sem `early_stopping`
   batendo o teto de 3000 iterações. A grade completa teria 600 ajustes na regressão e 1200 na
   classificação (sem peso + balanceado), projetando cerca de 4 + 19 min. Como as três buscas do MLP usam
   a mesma semente e a mesma grade, sorteiam as mesmas 40 configurações. O tempo real está em `custo.csv`.
   A Random Forest manteve a grade completa.
2. **Métrica de seleção na classificação: F1-macro** (o planejamento não fixa). Com a classe 4 em 2% das
   vítimas, a acurácia seria dominada pela classe 2. Na regressão, RMSE.
3. **Pesos do MLP de classificação.** Na busca, `compute_sample_weight("balanced", y)` é calculado no
   desenvolvimento inteiro e fatiado por dobra pelo próprio `RandomizedSearchCV`, como o planejamento
   descreve. Como as dobras são estratificadas, cada parte de treino tem a mesma proporção de classes do
   desenvolvimento (a menos de uma vítima), então os pesos fatiados equivalem aos da própria dobra. Na
   validação cruzada repetida e nos modelos finais, os pesos são calculados só sobre a parte de treino.
   A variante (sem peso × balanceado) escolhida é a de maior F1-macro na busca; os dois valores estão em
   `melhores_params.json`.
4. **Precision, recall e F1 com `zero_division=0`:** classe nunca prevista numa dobra conta como 0.
5. **Desvio-padrão com `ddof=0`**, a mesma convenção do `GridSearchCV` e do EDA.
6. **Distância ao corte** (`erros_vs_distancia_corte.csv`): medida até os cortes 25/50/75, que
   reproduzem 100% das classes reais (EDA), e não até os limiares estimados em cada dobra.
7. **`rmse_por_classe.csv`:** na CV, o RMSE é calculado sobre as predições fora da dobra das 5 repetições
   juntas (cada vítima do desenvolvimento aparece 5 vezes); a coluna `desvio_entre_repeticoes` é o desvio
   do RMSE calculado repetição a repetição. No teste, uma predição por vítima.
8. **Custo:** busca com 8 processos (limite de memória do WSL, 4 GB); treino final e latência com 1
   processo. Latência = mediana de 20 repetições; "lote" = predição das 300 vítimas do teste dividida por
   300; "unitária" = uma vítima por chamada. O tempo de busca do MLP de classificação soma as duas
   variantes de peso.
9. **Arquivo extra `importancia_permutacao.csv`:** os valores da fig4, para que os números citados no
   artigo estejam em `resultados/`.
10. **Predições (`predicoes_cv.csv`, `predicoes_teste.csv`):** `g_pred_RF` e `g_pred_MLP` são os ĝ; as
    colunas das 4 variantes são as classes previstas. Na CV, `limiar_1_2`, `limiar_2_3` e `limiar_3_4`
    são os limiares estimados na parte de treino daquela dobra; no teste, os limiares são os do
    desenvolvimento (`limiares.json`).
11. **Extrapolação (`extrapolacao.csv`):** o "treino" de cada predição é a parte de treino da dobra (CV) ou
    o desenvolvimento inteiro (teste). Só conta como fora da faixa a predição que passa do mínimo ou do
    máximo por mais de 1e-6, a resolução dos dados (6 casas): sem essa tolerância, a média das folhas da RF
    sobre vítimas com o mesmo g mínimo saía 7e-15 abaixo dele por arredondamento de ponto flutuante.
12. **Limiar conservador (`limiar_conservador.csv`, seção 12 do `tabelas.md`):** δ é somado aos três
    limiares de cada dobra, nas predições fora da dobra das variantes em 2 estágios. Métricas por dobra e
    depois média ± dp nas 25 dobras; contagens somadas. O código confere que δ = 0 reproduz as classes em
    2 estágios gravadas (erro se não reproduzir). As métricas principais continuam com δ = 0.

## Fase 3

13. **Estratégia do teste cego: 2 estágios**, o padrão do planejamento. Na validação cruzada, a variante
    direta só fica à frente na RF, e por menos de um desvio-padrão em F1-macro (`tabelas.md`, seção 2); no
    MLP o 2 estágios é melhor. A direta não é "claramente melhor", então o padrão fica.
14. **Coerência gravid ↔ classe:** o `prever.py` arredonda ĝ a 4 casas antes de classificar, então a classe
    gravada é sempre a faixa do gravid gravado. Sem isso, um ĝ a menos de 0,00005 de um limiar poderia
    sair com uma classe incoerente com o valor impresso.
15. **Leitura da entrada:** linhas vazias são ignoradas; só a primeira linha não vazia pode ser cabeçalho
    (nenhum campo numérico), e ela é mostrada na saída; qualquer outra linha precisa ter 4 números finitos,
    senão o programa para indicando a linha. Os ids saem exatamente como vieram. Valores fora dos
    intervalos do enunciado e ids repetidos geram aviso, sem abortar; a predição é feita mesmo assim.
    (A regra do cabeçalho de uma linha foi substituída pelo item 18.)
16. **Teste de fumaça em módulo próprio (`src/teste_fumaca.py`)**, fora da estrutura prevista no planejamento, para o
    `prever.py` ficar só com a predição. A entrada de exemplo tem cabeçalho e espaço após a vírgula, como o
    exemplo do enunciado.
17. **Limitação de ĝ no teste cego (ajuste opcional, depois da Fase 3).** O `prever.py` limita ĝ com
    `np.clip` à faixa de g dos 1500 rotulados (os dados do treino final, lidos com `ler_rotulados()`), na RF
    e na MLP, antes de arredondar e classificar. Motivo: a MLP extrapola abaixo do menor g observado
    (`extrapolacao.csv`) e, com entradas fora dos intervalos do enunciado, chegou a prever gravidade
    negativa; a RF nunca sai da faixa do treino. Conta como "limitada" a previsão cujo valor com 4 casas
    mudou (a comparação exata contaria as diferenças de ~1e-14 da RF). Os limites ficam fora dos limiares
    do 2 estágios, então a classe nunca muda. O efeito está em `limitacao_saida.csv` (`src/limitacao.py`),
    calculado a partir das predições gravadas na Fase 2, sem re-treinar: na CV, com os limites da parte de
    treino de cada dobra (as outras dobras da mesma repetição), média ± dp nas 25 dobras; no teste, com os
    limites do desenvolvimento. O script confere que as métricas sem limitação reproduzem
    `metricas_cv_resumo.csv` e `metricas_teste.csv` e que as classes em 2 estágios recalculadas são as
    gravadas. As métricas da Fase 2 (`tabelas.md` e demais CSVs) continuam sem limitação. Consequência
    visível: no exemplo do hold-out, as previsões da MLP abaixo do menor g passaram a sair no próprio
    mínimo (`teste_cego/predicoes_mlp.txt`), todas ainda na classe 1.
18. **Cabeçalho de várias linhas (ajuste opcional, depois da Fase 3).** Qualquer número de linhas sem campo
    numérico antes da primeira vítima é tratado como cabeçalho e mostrado na saída, como o cabeçalho de
    duas linhas do enunciado (`i si3 si4 si5` / `qPA pulso resp`). Texto depois da primeira vítima continua
    sendo erro. O teste de fumaça ganhou o caso do exemplo do enunciado (`teste_cego/enunciado/`) e passou
    a conferir, nos dois casos, que todo gravid fica dentro da faixa dos rotulados.

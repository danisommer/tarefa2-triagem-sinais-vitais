# Tabelas para o artigo

Gerado por `python -m src.experimentos`; não editar à mão. CV = validação cruzada repetida (5 repetições × 5 dobras estratificadas) no desenvolvimento (1200 vítimas): média ± desvio-padrão (ddof=0) nas 25 dobras. Teste = hold-out estratificado (300 vítimas), usado uma única vez, com os modelos treinados no desenvolvimento.

## 1. Regressão

| Modelo | RMSE CV | MAE CV | R² CV | RMSE teste | MAE teste | R² teste |
|---|---|---|---|---|---|---|
| RF | 2,8567 ± 0,3482 | 1,8051 ± 0,1918 | 0,9702 ± 0,0069 | 2,3818 | 1,4728 | 0,9786 |
| MLP | 2,1570 ± 0,1805 | 1,5356 ± 0,0938 | 0,9831 ± 0,0028 | 2,0990 | 1,5679 | 0,9834 |

## 2. Classificação — 4 variantes

| Variante | Acurácia CV | F1-macro CV | Precision-macro CV | Recall-macro CV | Acurácia teste | F1-macro teste | Precision-macro teste | Recall-macro teste |
|---|---|---|---|---|---|---|---|---|
| RF direta | 0,9218 ± 0,0162 | 0,8887 ± 0,0424 | 0,9297 ± 0,0356 | 0,8626 ± 0,0500 | 0,9333 | 0,8998 | 0,9617 | 0,8576 |
| RF 2 estágios | 0,9207 ± 0,0181 | 0,8687 ± 0,0479 | 0,9478 ± 0,0137 | 0,8304 ± 0,0572 | 0,9433 | 0,9318 | 0,9612 | 0,9073 |
| MLP direta | 0,9138 ± 0,0185 | 0,8791 ± 0,0473 | 0,9130 ± 0,0531 | 0,8587 ± 0,0530 | 0,8867 | 0,8466 | 0,8459 | 0,8522 |
| MLP 2 estágios | 0,9275 ± 0,0139 | 0,9189 ± 0,0245 | 0,9256 ± 0,0207 | 0,9183 ± 0,0357 | 0,9100 | 0,9016 | 0,9145 | 0,8967 |

## 3. Classificação por classe

| Variante | Classe | Precision CV | Recall CV | F1 CV | Precision teste | Recall teste | F1 teste |
|---|---|---|---|---|---|---|---|
| RF direta | 1 | 0,9662 ± 0,0279 | 0,9039 ± 0,0433 | 0,9334 ± 0,0293 | 0,9778 | 0,9362 | 0,9565 |
| RF direta | 2 | 0,9259 ± 0,0198 | 0,9478 ± 0,0140 | 0,9366 ± 0,0127 | 0,9111 | 0,9880 | 0,9480 |
| RF direta | 3 | 0,8907 ± 0,0316 | 0,8947 ± 0,0378 | 0,8921 ± 0,0258 | 0,9577 | 0,8395 | 0,8947 |
| RF direta | 4 | 0,9360 ± 0,1253 | 0,7040 ± 0,1755 | 0,7928 ± 0,1423 | 1,0000 | 0,6667 | 0,8000 |
| RF 2 estágios | 1 | 0,9976 ± 0,0082 | 0,8623 ± 0,0565 | 0,9240 ± 0,0320 | 0,9767 | 0,8936 | 0,9333 |
| RF 2 estágios | 2 | 0,9237 ± 0,0209 | 0,9510 ± 0,0181 | 0,9370 ± 0,0149 | 0,9412 | 0,9639 | 0,9524 |
| RF 2 estágios | 3 | 0,8766 ± 0,0340 | 0,9164 ± 0,0389 | 0,8955 ± 0,0282 | 0,9268 | 0,9383 | 0,9325 |
| RF 2 estágios | 4 | 0,9933 ± 0,0327 | 0,5920 ± 0,2038 | 0,7182 ± 0,1683 | 1,0000 | 0,8333 | 0,9091 |
| MLP direta | 1 | 0,9163 ± 0,0480 | 0,8227 ± 0,0671 | 0,8653 ± 0,0473 | 0,8444 | 0,8085 | 0,8261 |
| MLP direta | 2 | 0,9159 ± 0,0198 | 0,9441 ± 0,0214 | 0,9296 ± 0,0148 | 0,8814 | 0,9398 | 0,9096 |
| MLP direta | 3 | 0,9129 ± 0,0288 | 0,9158 ± 0,0353 | 0,9137 ± 0,0208 | 0,9437 | 0,8272 | 0,8816 |
| MLP direta | 4 | 0,9068 ± 0,1808 | 0,7520 ± 0,1857 | 0,8077 ± 0,1564 | 0,7143 | 0,8333 | 0,7692 |
| MLP 2 estágios | 1 | 0,8628 ± 0,0489 | 0,8793 ± 0,0546 | 0,8690 ± 0,0306 | 0,7719 | 0,9362 | 0,8462 |
| MLP 2 estágios | 2 | 0,9395 ± 0,0190 | 0,9349 ± 0,0211 | 0,9369 ± 0,0131 | 0,9375 | 0,9036 | 0,9202 |
| MLP 2 estágios | 3 | 0,9441 ± 0,0282 | 0,9411 ± 0,0268 | 0,9421 ± 0,0152 | 0,9487 | 0,9136 | 0,9308 |
| MLP 2 estágios | 4 | 0,9560 ± 0,0787 | 0,9180 ± 0,1406 | 0,9276 ± 0,0874 | 1,0000 | 0,8333 | 0,9091 |

## 4. Triagem

Subtriagem = classe prevista > real (vítima tratada como mais estável do que está); sobretriagem = prevista < real. Taxa = fração das vítimas; na CV, média ± dp nas 25 dobras e, entre parênteses, a soma das contagens nas 25 dobras.

| Variante | Recall classe 1 CV | Subtriagem CV | Sobretriagem CV | Recall classe 1 teste | Subtriagem teste | Sobretriagem teste |
|---|---|---|---|---|---|---|
| RF direta | 0,9039 ± 0,0433 | 0,0400 ± 0,0113 (240 de 6000) | 0,0382 ± 0,0138 (229 de 6000) | 0,9362 | 0,0133 (4 de 300) | 0,0533 (16 de 300) |
| RF 2 estágios | 0,8623 ± 0,0565 | 0,0485 ± 0,0149 (291 de 6000) | 0,0308 ± 0,0116 (185 de 6000) | 0,8936 | 0,0333 (10 de 300) | 0,0233 (7 de 300) |
| MLP direta | 0,8227 ± 0,0671 | 0,0487 ± 0,0132 (292 de 6000) | 0,0375 ± 0,0126 (225 de 6000) | 0,8085 | 0,0467 (14 de 300) | 0,0667 (20 de 300) |
| MLP 2 estágios | 0,8793 ± 0,0546 | 0,0335 ± 0,0117 (201 de 6000) | 0,0390 ± 0,0157 (234 de 6000) | 0,9362 | 0,0200 (6 de 300) | 0,0700 (21 de 300) |

## 5. Matrizes de confusão no teste

Linhas = classe real; colunas = classe prevista.

**RF direta**

| Real × prevista | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 44 | 3 | 0 | 0 |
| 2 | 1 | 164 | 1 | 0 |
| 3 | 0 | 13 | 68 | 0 |
| 4 | 0 | 0 | 2 | 4 |

**RF 2 estágios**

| Real × prevista | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 42 | 5 | 0 | 0 |
| 2 | 1 | 160 | 5 | 0 |
| 3 | 0 | 5 | 76 | 0 |
| 4 | 0 | 0 | 1 | 5 |

**MLP direta**

| Real × prevista | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 38 | 9 | 0 | 0 |
| 2 | 7 | 156 | 3 | 0 |
| 3 | 0 | 12 | 67 | 2 |
| 4 | 0 | 0 | 1 | 5 |

**MLP 2 estágios**

| Real × prevista | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 44 | 3 | 0 | 0 |
| 2 | 13 | 150 | 3 | 0 |
| 3 | 0 | 7 | 74 | 0 |
| 4 | 0 | 0 | 1 | 5 |

## 6. Limiares do classificador em 2 estágios

Desenvolvimento → 2 estágios no teste; 100% rotulados → modelos finais (teste cego); dobras → 2 estágios na CV.

| Corte | Hipótese | Desenvolvimento (ponto médio) | Desenvolvimento (árvore prof. 2) | 100% rotulados | Dobras CV: média ± dp | Dobras CV: mín–máx |
|---|---|---|---|---|---|---|
| 1→2 | 25 | 25,0017 | 25,0017 | 25,0017 | 25,0005 ± 0,0076 | 24,9846–25,0155 |
| 2→3 | 50 | 50,0137 | 50,0137 | 50,0137 | 50,0136 ± 0,0043 | 50,0010–50,0212 |
| 3→4 | 75 | 75,2561 | 75,2561 | 74,9100 | 75,3359 ± 0,2536 | 74,8529–75,9906 |

## 7. Hiperparâmetros escolhidos

| Modelo | Busca | Hiperparâmetros | Score de seleção (CV da busca) |
|---|---|---|---|
| RF_reg | GridSearchCV (54 candidatos) | `max_depth=20, max_features=3, min_samples_leaf=1, n_estimators=500` | rmse 2,7843 ± 0,4784 |
| RF_clf | GridSearchCV (162 candidatos) | `class_weight='balanced_subsample', max_depth=10, max_features=2, min_samples_leaf=1, n_estimators=500` | f1_macro 0,8983 ± 0,0268 |
| MLP_reg | RandomizedSearchCV (40 candidatos) | `activation='relu', alpha=0.001, early_stopping=False, hidden_layer_sizes=(128, 64), learning_rate_init=0.001` | rmse 2,1780 ± 0,1564 |
| MLP_clf | RandomizedSearchCV (40 candidatos) | `activation='tanh', alpha=0.01, early_stopping=False, hidden_layer_sizes=(32,), learning_rate_init=0.001` | f1_macro 0,8985 ± 0,0217 |

MLP de classificação, melhor F1-macro na busca por variante de peso: sem peso = 0,8985; balanceado = 0,8875. Escolhido: peso nenhum.

## 8. Custo

Busca com 8 processos; treino final e latência com 1 processo. MLP_clf soma as buscas sem peso e balanceada. O custo do 2 estágios é o do regressor.

| Modelo | Busca | Candidatos | Ajustes na busca | Busca (s) | Treino final (s) | Latência em lote (ms/vítima) | Latência unitária (ms) | .joblib (KB) |
|---|---|---|---|---|---|---|---|---|
| RF_reg | GridSearchCV | 54 | 270 | 49,1 | 1,301 | 0,1866 | 30,409 | 60213,0 |
| RF_clf | GridSearchCV | 162 | 810 | 177,0 | 1,512 | 0,1225 | 25,294 | 10412,2 |
| MLP_reg | RandomizedSearchCV | 40 | 200 | 96,8 | 1,669 | 0,0037 | 1,241 | 289,6 |
| MLP_clf | RandomizedSearchCV | 80 | 400 | 502,5 | 6,018 | 0,0044 | 1,061 | 58,3 |

Tempo total da execução: 915 s (15,2 min).

## 9. RMSE por classe real

CV: RMSE sobre as predições fora da dobra das 5 repetições juntas (entre parênteses, o desvio do RMSE entre as repetições).

| Regressor | Classe | n CV (predições) | RMSE CV | n teste | RMSE teste |
|---|---|---|---|---|---|
| RF | todas | 6000 | 2,8778 (0,0551) | 300 | 2,3818 |
| RF | 1 | 935 | 3,2954 (0,1901) | 47 | 3,0802 |
| RF | 2 | 3330 | 2,3357 (0,0658) | 166 | 1,9224 |
| RF | 3 | 1615 | 3,0341 (0,0890) | 81 | 2,6768 |
| RF | 4 | 120 | 7,3620 (0,8649) | 6 | 3,2189 |
| MLP | todas | 6000 | 2,1645 (0,0437) | 300 | 2,0990 |
| MLP | 1 | 935 | 2,8854 (0,0827) | 47 | 2,4344 |
| MLP | 2 | 3330 | 2,0026 (0,0325) | 166 | 2,1644 |
| MLP | 3 | 1615 | 1,9413 (0,0831) | 81 | 1,7158 |
| MLP | 4 | 120 | 2,7159 (0,1579) | 6 | 2,1239 |

## 10. Taxa de erro × distância ao corte mais próximo

Distância entre o g real e o corte (25, 50 ou 75) mais próximo; predições fora da dobra das 5 repetições agregadas. Célula = taxa de erro (número de erros).

| Distância ao corte | n (predições) | RF direta | RF 2 estágios | MLP direta | MLP 2 estágios |
|---|---|---|---|---|---|
| 0–0,5 | 450 | 0,2511 (113) | 0,3200 (144) | 0,4533 (204) | 0,4889 (220) |
| 0,5–1 | 230 | 0,3000 (69) | 0,4000 (92) | 0,5000 (115) | 0,3565 (82) |
| 1–2,5 | 875 | 0,1257 (110) | 0,0960 (84) | 0,0823 (72) | 0,0857 (75) |
| 2,5–5 | 675 | 0,1185 (80) | 0,1141 (77) | 0,0993 (67) | 0,0637 (43) |
| > 5 | 3770 | 0,0257 (97) | 0,0210 (79) | 0,0156 (59) | 0,0040 (15) |

## 11. Importância por permutação no teste

Aumento do RMSE ao embaralhar o atributo (30 repetições; média ± dp).

| Atributo | RF | MLP |
|---|---|---|
| `qPA` | 9,9932 ± 0,3711 | 10,6268 ± 0,3927 |
| `pulso` | 10,4744 ± 0,4552 | 10,9373 ± 0,4417 |
| `resp` | 11,9158 ± 0,4945 | 12,1960 ± 0,5246 |

## 12. Limiar conservador (variantes em 2 estágios)

Limiares de cada dobra somados a δ: a vítima vai para a classe mais grave quando ĝ < limiar + δ. Predições fora da dobra das 5 repetições; média ± dp nas 25 dobras e, entre parênteses, a soma das contagens. δ = 0 reproduz as seções 2 a 4; as métricas principais usam δ = 0.

| Variante | δ | Subtriagem | Sobretriagem | Recall classe 1 | Acurácia | F1-macro |
|---|---|---|---|---|---|---|
| RF 2 estágios | 0,0 | 0,0485 ± 0,0149 (291 de 6000) | 0,0308 ± 0,0116 (185 de 6000) | 0,8623 ± 0,0565 | 0,9207 ± 0,0181 | 0,8687 ± 0,0479 |
| RF 2 estágios | 0,5 | 0,0352 ± 0,0107 (211 de 6000) | 0,0575 ± 0,0173 (345 de 6000) | 0,9083 ± 0,0454 | 0,9073 ± 0,0206 | 0,8526 ± 0,0505 |
| RF 2 estágios | 1,0 | 0,0275 ± 0,0089 (165 de 6000) | 0,0652 ± 0,0177 (391 de 6000) | 0,9295 ± 0,0424 | 0,9073 ± 0,0200 | 0,8438 ± 0,0467 |
| RF 2 estágios | 1,5 | 0,0232 ± 0,0093 (139 de 6000) | 0,0718 ± 0,0188 (431 de 6000) | 0,9360 ± 0,0411 | 0,9050 ± 0,0190 | 0,8357 ± 0,0446 |
| RF 2 estágios | 2,0 | 0,0177 ± 0,0093 (106 de 6000) | 0,0978 ± 0,0215 (587 de 6000) | 0,9467 ± 0,0411 | 0,8845 ± 0,0217 | 0,8128 ± 0,0458 |
| RF 2 estágios | 2,5 | 0,0150 ± 0,0082 (90 de 6000) | 0,1297 ± 0,0192 (778 de 6000) | 0,9552 ± 0,0374 | 0,8553 ± 0,0199 | 0,7800 ± 0,0457 |
| RF 2 estágios | 3,0 | 0,0118 ± 0,0065 (71 de 6000) | 0,1417 ± 0,0180 (850 de 6000) | 0,9637 ± 0,0291 | 0,8465 ± 0,0185 | 0,7603 ± 0,0503 |
| MLP 2 estágios | 0,0 | 0,0335 ± 0,0117 (201 de 6000) | 0,0390 ± 0,0157 (234 de 6000) | 0,8793 ± 0,0546 | 0,9275 ± 0,0139 | 0,9189 ± 0,0245 |
| MLP 2 estágios | 0,5 | 0,0227 ± 0,0080 (136 de 6000) | 0,0522 ± 0,0194 (313 de 6000) | 0,9241 ± 0,0344 | 0,9252 ± 0,0185 | 0,9219 ± 0,0267 |
| MLP 2 estágios | 1,0 | 0,0182 ± 0,0056 (109 de 6000) | 0,0692 ± 0,0211 (415 de 6000) | 0,9402 ± 0,0242 | 0,9127 ± 0,0205 | 0,9136 ± 0,0277 |
| MLP 2 estágios | 1,5 | 0,0140 ± 0,0054 (84 de 6000) | 0,0873 ± 0,0217 (524 de 6000) | 0,9519 ± 0,0263 | 0,8987 ± 0,0210 | 0,8991 ± 0,0270 |
| MLP 2 estágios | 2,0 | 0,0102 ± 0,0062 (61 de 6000) | 0,1058 ± 0,0199 (635 de 6000) | 0,9647 ± 0,0249 | 0,8840 ± 0,0202 | 0,8821 ± 0,0260 |
| MLP 2 estágios | 2,5 | 0,0073 ± 0,0050 (44 de 6000) | 0,1220 ± 0,0210 (732 de 6000) | 0,9765 ± 0,0231 | 0,8707 ± 0,0209 | 0,8653 ± 0,0322 |
| MLP 2 estágios | 3,0 | 0,0057 ± 0,0041 (34 de 6000) | 0,1400 ± 0,0244 (840 de 6000) | 0,9818 ± 0,0181 | 0,8543 ± 0,0239 | 0,8408 ± 0,0450 |

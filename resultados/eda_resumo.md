# Fase 1 — Análise exploratória

Gerado por `python -m src.eda`; não editar à mão. Análise sobre o arquivo 02; o 01 entra só na seção 2.

## Conferência dos fatos esperados

**23 de 23 fatos conferem.**

| # | Fato | Esperado | Obtido | Confere |
|---|---|---|---|---|
| 1 | 02: 1500 linhas, sem cabeçalho | 1500, sem | 1500, sem | sim |
| 2 | 02: 8 colunas separadas por `,` | sim | sim | sim |
| 3 | 01: 1500 linhas, sem cabeçalho | 1500, sem | 1500, sem | sim |
| 4 | 01: 7 colunas (sem `classe`) separadas por `, ` | sim | sim | sim |
| 5 | 01 × 02: mesmas vítimas e entradas idênticas | sim | sim | sim |
| 6 | 01 × 02: diferença máxima em `gravid` | ≤ 1.3e-05 | 1.30e-05 | sim |
| 7 | Sem valores faltantes (02 e 01) | 0 e 0 | 0 e 0 | sim |
| 8 | Contagem por classe | 1 = 234 · 2 = 832 · 3 = 404 · 4 = 30 | 1 = 234 · 2 = 832 · 3 = 404 · 4 = 30 | sim |
| 9 | Faixa de `qPA` | [-8,733; 8,733] | [-8,733; 8,733] | sim |
| 10 | Classe 1: faixa de g | [13,2227; 24,9784] | [13,2227; 24,9784] | sim |
| 11 | Classe 2: faixa de g | [25,0249; 49,9977] | [25,0249; 49,9977] | sim |
| 12 | Classe 3: faixa de g | [50,0296; 74,6574] | [50,0296; 74,6574] | sim |
| 13 | Classe 4: faixa de g | [75,1626; 87,0000] | [75,1626; 87,0000] | sim |
| 14 | Faixas de g das classes sem sobreposição | sim | sim | sim |
| 15 | Cortes 25 / 50 / 75 reproduzem a classe | 1500 de 1500 | 1500 de 1500 | sim |
| 16 | Maiores platôs de g | 40,0000 (90) · 13,2227 (41) · 25,1705 (26) · 27,1226 (23) | 40,0000 (90) · 13,2227 (41) · 25,1705 (26) · 27,1226 (23) | sim |
| 17 | 13,2227 é o mínimo de g | 13,2227 | 13,2227 | sim |
| 18 | Maiores platôs de qPA | 8,7333 (197) · 0,0000 (193) · 4,6667 (79) · -4,3333 (61) | 8,7333 (197) · 0,0000 (193) · 4,6667 (79) · -4,3333 (61) | sim |
| 19 | R² de qPA ~ (pSist, pDiast) com RF | 0,997 | 0,997 | sim |
| 20 | Baseline RF regressão: R² ≥ 0,96 | ≥ 0,96 | 0,9709 | sim |
| 21 | Baseline MLP (64, 32) regressão: R² ≥ 0,96 | ≥ 0,96 | 0,9789 | sim |
| 22 | Baseline RF classificação: acurácia ≥ 0,89 | ≥ 0,89 | 0,9108 | sim |
| 23 | Baseline MLP (64, 32) classificação: acurácia ≥ 0,89 | ≥ 0,89 | 0,9083 | sim |

Critérios: valores com 3 ou 4 casas conferem se forem iguais após o arredondamento; Δgravid conferem se o máximo arredondar para até 1,3e-05 (< 1.35e-05); R² do qPA ±0,002; baseline confere se ficar acima dos pisos de sanidade (R² ≥ 0,96 e acurácia ≥ 0,89).

## 1. Formato dos arquivos

| Arquivo | Linhas | Linhas vazias | Colunas | Separador | Separador em todas as linhas | Cabeçalho |
|---|---|---|---|---|---|---|
| 02_treino_sinais_vitais_com_label.txt | 1500 | 0 | 8 | `,` | sim | não |
| 01_treino_sinais_vitais_sem_label.txt | 1500 | 0 | 7 | `, ` | sim | não |

Colunas do 02: `id,pSist,pDiast,qPA,pulso,resp,gravid,classe`. A ordem é confirmada pelas faixas da seção 5, que batem com os intervalos do enunciado.

## 2. Consistência entre o 01 e o 02

- Mesmos ids (1 a 1500), na mesma ordem: sim.
- `id, pSist, pDiast, qPA, pulso, resp` idênticos nos dois arquivos: sim.
- `gravid`: 933 de 1500 linhas diferem; diferença absoluta máxima = 1.30e-05, média = 1.20e-06.
- Conclusão: o 01 é o 02 sem a coluna `classe`, com `gravid` arredondado de outra forma. Só o 02 é usado daqui em diante.

## 3. Valores faltantes

02: 0 · 01: 0.

## 4. Distribuição das classes

| Classe | Nome | Vítimas | % |
|---|---|---|---|
| 1 | crítico | 234 | 15,6 |
| 2 | instável | 832 | 55,5 |
| 3 | potencialmente estável | 404 | 26,9 |
| 4 | estável | 30 | 2,0 |

Razão entre a maior e a menor classe: 27,7. A classe 4 tem só 30 vítimas.

## 5. Atributos

| Atributo | Uso | Mín | Máx | Média | Desvio | Intervalo do enunciado | Fora do intervalo |
|---|---|---|---|---|---|---|---|
| `pSist` | proibido | 0,4355 | 21,9975 | 14,8883 | 4,7907 | [5; 22] | 54 |
| `pDiast` | proibido | 0,0078 | 14,9924 | 7,5001 | 4,2482 | [0; 15] | 0 |
| `qPA` | entrada | -8,7329 | 8,7333 | 2,4150 | 5,1280 | [-10; 10] | 0 |
| `pulso` | entrada | 0,0462 | 199,9546 | 101,1962 | 57,8727 | [0; 200] | 0 |
| `resp` | entrada | 0,0072 | 21,9596 | 10,8546 | 6,2772 | [0; 22] | 0 |
| `gravid` | alvo | 13,2227 | 87,0000 | 41,4274 | 16,5905 | — | — |

Correlação de cada entrada com `gravid`:

| Entrada | Spearman | Pearson |
|---|---|---|
| `qPA` | -0,317 | -0,287 |
| `pulso` | -0,097 | -0,092 |
| `resp` | 0,399 | 0,390 |

## 6. Gravidade por classe

Escala invertida: g baixo = estado mais grave.

| Classe | Nome | Vítimas | g mín | g máx | g médio |
|---|---|---|---|---|---|
| 1 | crítico | 234 | 13,2227 | 24,9784 | 16,9431 |
| 2 | instável | 832 | 25,0249 | 49,9977 | 37,7131 |
| 3 | potencialmente estável | 404 | 50,0296 | 74,6574 | 60,2567 |
| 4 | estável | 30 | 75,1626 | 87,0000 | 81,8473 |

Cortes 25 / 50 / 75 (g < 25 → 1; 25 ≤ g < 50 → 2; 50 ≤ g < 75 → 3; g ≥ 75 → 4): 1500 de 1500 acertos. Faixas sem sobreposição: sim.

Limiares estimados nos 1500 (só descritivo; na Fase 2 eles saem só das partes de treino):

| Corte | Hipótese | Ponto médio | Árvore prof. 2 sobre g | Folga entre as classes |
|---|---|---|---|---|
| 1→2 | 25 | 25,0017 | 25,0017 | 0,0465 |
| 2→3 | 50 | 50,0137 | 50,0137 | 0,0319 |
| 3→4 | 75 | 74,9100 | 74,9100 | 0,5052 |

Árvore e pontos médios coincidem (tolerância 1e-4): sim. A folga é a distância entre o maior g de uma classe e o menor g da seguinte: um erro de regressão menor que ela, perto do corte, já basta para mudar a classe.

Figura: `figuras/fig1_gravidade_por_classe.png`.

## 7. Platôs (valores repetidos)

`gravid`: 1301 valores distintos em 1500 vítimas. `qPA`: 936.

| `gravid` | Vítimas | `qPA` | Vítimas |
|---|---|---|---|
| 40,0000 | 90 | 8,7333 | 197 |
| 13,2227 | 41 | 0,0000 | 193 |
| 25,1705 | 26 | 4,6667 | 79 |
| 27,1226 | 23 | -4,3333 | 61 |
| 68,3333 | 7 | 4,6674 | 3 |
| 67,9891 | 4 | -4,4176 | 3 |

O platô 25,1705 fica a 0,1705 do corte 25.
Hipótese (não verificada): valores gerados por sistema fuzzy com defuzzificação por centroide.

## 8. `qPA` como função de (`pSist`, `pDiast`)

Random Forest padrão, `KFold(5, shuffle=True, random_state=42)`, nos 1500: R² = 0,9970 ± 0,0041.
Análise só descritiva: mostra que `qPA` resume as duas pressões. `pSist` e `pDiast` não entram em nenhum modelo.

## 9. Baseline sem ajuste (sanidade)

Só no conjunto de desenvolvimento (1200 vítimas, 80% estratificado com `SEED = 42`, a mesma divisão da Fase 2), para que nenhum modelo veja o hold-out antes do fim. `StratifiedKFold(5)` pela classe, também na regressão. RF com hiperparâmetros padrão do scikit-learn; MLP com os parâmetros fixos (`adam`, `max_iter=3000`, `n_iter_no_change=30`).

| Regressor | RMSE | R² |
|---|---|---|
| RF | 2,8251 ± 0,4577 | 0,9709 ± 0,0081 |
| MLP (64, 32) | 2,4182 ± 0,1941 | 0,9789 ± 0,0025 |
| MLP (64, 32) + early_stopping | 2,3360 ± 0,2352 | 0,9802 ± 0,0032 |

| Classificador (direto) | Acurácia |
|---|---|
| RF | 0,9108 ± 0,0164 |
| MLP (64, 32) | 0,9083 ± 0,0160 |

`early_stopping=True` no MLP (64, 32): RMSE 2,34 com × 2,42 sem. Só informativo: ele entra na grade da Fase 2.

## 10. g médio por faixa de cada entrada

Só descritivo, nos 1500. Faixas fechadas à direita; a primeira de `pulso` e de `resp` inclui também o limite inferior. g médio geral = 41,4274. Figura: `figuras/fig5_g_por_faixa.png`.

`pulso`:

| Faixa | n | g médio |
|---|---|---|
| [0; 25] | 183 | 30,7591 |
| (25; 50] | 187 | 38,7337 |
| (50; 75] | 174 | 52,2299 |
| (75; 100] | 197 | 56,6877 |
| (100; 125] | 188 | 47,4055 |
| (125; 150] | 183 | 37,2435 |
| (150; 175] | 188 | 34,6137 |
| (175; 200] | 200 | 33,8915 |

`resp`:

| Faixa | n | g médio |
|---|---|---|
| [0; 4] | 279 | 32,7268 |
| (4; 8] | 278 | 32,6155 |
| (8; 12] | 271 | 36,4367 |
| (12; 16] | 271 | 57,3196 |
| (16; 20] | 284 | 49,5108 |
| (20; 22] | 117 | 38,2408 |

`qPA`:

| Faixa | n | g médio |
|---|---|---|
| (-9; -6] | 61 | 37,1941 |
| (-6; -3] | 293 | 42,5824 |
| (-3; -0,5] | 25 | 56,7052 |
| (-0,5; 0,5] | 216 | 56,7095 |
| (0,5; 3] | 84 | 52,4068 |
| (3; 6] | 423 | 40,2272 |
| (6; 9] | 398 | 30,9308 |

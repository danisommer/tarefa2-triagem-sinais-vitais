# Tarefa 2 — Random Forest × MLP na triagem por sinais vitais

Trabalho da disciplina de Sistemas Inteligentes (UTFPR). A partir de três sinais vitais da vítima — `qPA`
(qualidade da pressão arterial), `pulso` e `resp` (frequência respiratória) — o código:

- reconstitui a gravidade `g` (**regressão**);
- classifica a vítima em 4 classes (**classificação**): 1 = crítico, 2 = instável, 3 = potencialmente
  estável, 4 = estável;

com duas técnicas do scikit-learn: **Random Forest** e **rede neural MLP**. Como pede o enunciado, `pSist`
e `pDiast` não são usados como entrada. Todos os números, tabelas e figuras do artigo são gerados por este
código e ficam em `resultados/`.

Metodologia em resumo: divisão estratificada em 80% desenvolvimento e 20% teste (hold-out, ids em
`resultados/ids_teste.csv`); hiperparâmetros escolhidos por validação cruzada só no desenvolvimento;
robustez medida com validação cruzada repetida (5 × 5 dobras); o hold-out é usado uma única vez, no fim.
A classificação é feita de duas formas: direta (o classificador prevê a classe) e em 2 estágios (o
regressor prevê a gravidade e a classe é a faixa dela).

## Estrutura

```
dados/              dados rotulados fornecidos (o arquivo 02 é o usado; o 01 só entra numa checagem do EDA)
src/
  comum.py          caminhos, semente, leitura dos dados, limiares e criação dos modelos
  eda.py            análise exploratória
  experimentos.py   busca de hiperparâmetros, validação cruzada, teste, tabelas, figuras e modelos finais
  limitacao.py      efeito de limitar a gravidade prevista à faixa vista no treino
  prever.py         teste cego
  teste_fumaca.py   conferência automática do teste cego
modelos/            modelos finais (.joblib, gerados pelo src.experimentos) e limiares.json
resultados/         tabelas, CSVs, figuras, notas e predições
```

## Ambiente

Python 3.12 (os resultados foram gerados com 3.12.3; versões exatas das bibliotecas em `requirements.txt`
e `resultados/ambiente.json`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Execução

Sempre como módulo, a partir da raiz do repositório, nesta ordem:

```bash
python -m src.eda            # análise exploratória (~30 s)
python -m src.experimentos   # busca, validação cruzada repetida, teste e modelos finais (~15 min)
python -m src.limitacao      # efeito de limitar a gravidade prevista (lê as predições do passo anterior)
python -m src.teste_fumaca   # opcional: confere o teste cego no hold-out e no exemplo do enunciado
```

A semente é fixa (42), então rodar de novo reproduz os mesmos resultados. A exceção são os tempos medidos
em `resultados/custo.csv` (e na seção 8 de `resultados/tabelas.md`), que variam a cada execução.

**Os modelos treinados (`modelos/*.joblib`) não estão no repositório** (o da Random Forest de regressão tem
cerca de 60 MB). Antes do teste cego, rode `python -m src.experimentos` uma vez para gerá-los;
`modelos/limiares.json` já está versionado.

## Teste cego

```bash
python -m src.prever --entrada ARQUIVO [--modelo rf|mlp|ambos] [--saida-dir resultados/teste_cego]
```

- **Entrada:** uma vítima por linha, `i, qPA, pulso, resp`, separados por vírgula (espaços depois da
  vírgula são aceitos). Linhas de cabeçalho sem números antes da primeira vítima (como
  `i si3 si4 si5` / `qPA pulso resp`, do enunciado) e linhas vazias são ignoradas. Valores fora de
  qPA [−10, 10], pulso [0, 200] ou resp [0, 22] geram aviso, sem interromper.
- **Saída:** `predicoes_rf.txt` e `predicoes_mlp.txt` em `resultados/teste_cego/` (ou na pasta indicada em
  `--saida-dir`), sem cabeçalho, uma linha por vítima, na ordem da entrada, no formato `i,gravid,classe`,
  com a gravidade em 4 casas decimais. Por padrão, `--modelo ambos`.
- **Como a classe é obtida (2 estágios):** o regressor treinado com os 1500 dados rotulados prevê a
  gravidade, que é limitada à faixa de g observada nesses dados e arredondada a 4 casas; a classe é a
  faixa desse valor segundo os limiares de `modelos/limiares.json` (≈ 25, 50 e 75). Assim, gravidade e
  classe de cada linha são sempre coerentes.

Exemplos prontos: `resultados/teste_cego/exemplo_entrada.txt` (vítimas do hold-out) e
`resultados/teste_cego/enunciado/exemplo_entrada.txt` (o exemplo do enunciado), cada um com as predições
correspondentes ao lado.

## Onde estão os resultados

- `resultados/tabelas.md`: tabelas prontas para o artigo (regressão; as 4 variantes de classificação,
  RF e MLP, direta e em 2 estágios; métricas por classe; triagem; matrizes de confusão; limiares;
  hiperparâmetros; custo; análises complementares).
- `resultados/*.csv` e `resultados/*.json`: os mesmos números com precisão completa, entre eles
  `metricas_cv_dobras.csv` (por dobra), `metricas_cv_resumo.csv` (média ± desvio), `metricas_teste.csv`,
  `predicoes_cv.csv`, `predicoes_teste.csv`, `melhores_params.json` e `custo.csv`.
- `resultados/figuras/`: figuras do artigo (PNG, 300 dpi).
- `resultados/eda_resumo.md`: análise exploratória dos dados.
- `resultados/notas.md`: decisões de metodologia e seus motivos.

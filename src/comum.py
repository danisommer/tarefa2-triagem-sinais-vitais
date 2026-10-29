"""Caminhos, semente, leitura dos dados, limiares e fábricas de modelos compartilhados pelas fases."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42

RAIZ = Path(__file__).resolve().parent.parent
DADOS = RAIZ / "dados"
RESULTADOS = RAIZ / "resultados"
FIGURAS = RESULTADOS / "figuras"
MODELOS = RAIZ / "modelos"

ARQ_COM_LABEL = DADOS / "02_treino_sinais_vitais_com_label.txt"
ARQ_SEM_LABEL = DADOS / "01_treino_sinais_vitais_sem_label.txt"

COLUNAS_COM_LABEL = ["id", "pSist", "pDiast", "qPA", "pulso", "resp", "gravid", "classe"]
COLUNAS_SEM_LABEL = COLUNAS_COM_LABEL[:-1]
TIPOS = {"id": "int64", "classe": "int64"} | {c: "float64" for c in COLUNAS_COM_LABEL[1:-1]}

ATRIBUTOS = ["qPA", "pulso", "resp"]
ALVO_REG = "gravid"
ALVO_CLF = "classe"

CLASSES = [1, 2, 3, 4]
NOMES_CLASSES = {1: "crítico", 2: "instável", 3: "potencialmente estável", 4: "estável"}
LIMIARES_HIPOTESE = [25.0, 50.0, 75.0]
FAIXAS_ENUNCIADO = {
    "pSist": (5.0, 22.0),
    "pDiast": (0.0, 15.0),
    "qPA": (-10.0, 10.0),
    "pulso": (0.0, 200.0),
    "resp": (0.0, 22.0),
}

FRACAO_TESTE = 0.2
CASAS_GRAVID = 4
N_JOBS = 8
MLP_FIXOS = {"solver": "adam", "max_iter": 3000, "n_iter_no_change": 30, "random_state": SEED}

GRADE_RF = {
    "n_estimators": [200, 500],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [1, 2, 3],
}
GRADE_RF_CLF = GRADE_RF | {"class_weight": [None, "balanced", "balanced_subsample"]}
GRADE_MLP = {
    "hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (64, 64, 32)],
    "activation": ["relu", "tanh"],
    "alpha": [1e-4, 1e-3, 1e-2],
    "learning_rate_init": [1e-3, 3e-3],
    "early_stopping": [True, False],
}

CORES_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
RAMPA_AZUL = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SUPERFICIE = "#ffffff"
TINTA = "#0b0b0b"
TINTA_SECUNDARIA = "#52514e"
COR_GRADE = "#e1e0d9"
COR_EIXO = "#c3c2b7"
ESTILO_FIGURA = {
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "text.color": TINTA,
    "axes.labelcolor": TINTA_SECUNDARIA,
    "xtick.color": TINTA_SECUNDARIA,
    "ytick.color": TINTA_SECUNDARIA,
    "axes.edgecolor": COR_EIXO,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "ytick.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": COR_GRADE,
    "grid.linewidth": 0.5,
}


def ler_txt(caminho, colunas):
    """Lê um .txt sem cabeçalho separado por vírgula (com ou sem espaço); levanta erro se algum campo não for numérico."""
    return pd.read_csv(
        caminho,
        header=None,
        names=colunas,
        sep=",",
        skipinitialspace=True,
        dtype={c: TIPOS[c] for c in colunas},
    )


def ler_rotulados():
    """Lê o arquivo 02 (com classe), a única fonte de dados de treino e teste."""
    return ler_txt(ARQ_COM_LABEL, COLUNAS_COM_LABEL)


def dividir_hold_out(df):
    """Divide em desenvolvimento (80%) e teste (20%), estratificado por classe; retorna (dev, teste)."""
    return train_test_split(df, test_size=FRACAO_TESTE, stratify=df[ALVO_CLF], random_state=SEED)


def estimar_limiares(g, classe):
    """Ponto médio entre o maior g da classe k e o menor g da classe k+1; levanta ValueError se faltar classe ou houver sobreposição."""
    g, classe = np.asarray(g, dtype=float), np.asarray(classe)
    ausentes = [k for k in CLASSES if not np.any(classe == k)]
    if ausentes:
        raise ValueError(f"classes ausentes na estimação dos limiares: {ausentes}")
    maximos = [g[classe == k].max() for k in CLASSES[:-1]]
    minimos = [g[classe == k].min() for k in CLASSES[1:]]
    sobrepostas = [k for k, (ma, mi) in zip(CLASSES, zip(maximos, minimos)) if ma >= mi]
    if sobrepostas:
        raise ValueError(f"faixas de g sobrepostas entre as classes k e k+1 para k = {sobrepostas}")
    return [float((ma + mi) / 2) for ma, mi in zip(maximos, minimos)]


def limiares_arvore(g, classe):
    """Limiares escolhidos por uma árvore de profundidade 2 treinada só sobre g, em ordem crescente."""
    arvore = DecisionTreeClassifier(max_depth=2, random_state=SEED)
    arvore.fit(np.asarray(g, dtype=float).reshape(-1, 1), classe)
    no = arvore.tree_
    return sorted(float(t) for t, f in zip(no.threshold, no.feature) if f >= 0)


def faixa_g(d):
    """Menor e maior g de um conjunto rotulado."""
    return float(d[ALVO_REG].min()), float(d[ALVO_REG].max())


def limitar_g(g_pred, minimo, maximo):
    """ĝ limitado a [mínimo, máximo] (limites únicos ou um por previsão) e a máscara das previsões cujo valor com 4 casas mudou."""
    g_pred = np.asarray(g_pred, dtype=float)
    limitado = np.clip(g_pred, minimo, maximo)
    return limitado, np.round(limitado, CASAS_GRAVID) != np.round(g_pred, CASAS_GRAVID)


def classe_por_limiares(g, limiares):
    """Classe 1..4 pela faixa de g: g < l1 → 1, l1 ≤ g < l2 → 2, l2 ≤ g < l3 → 3, g ≥ l3 → 4."""
    return np.digitize(np.asarray(g, dtype=float), limiares) + 1


def criar_rf_reg(**params):
    """Random Forest de regressão (CART, erro quadrático)."""
    return RandomForestRegressor(criterion="squared_error", random_state=SEED, **params)


def criar_rf_clf(**params):
    """Random Forest de classificação com ganho de informação (entropia)."""
    return RandomForestClassifier(criterion="entropy", random_state=SEED, **params)


def criar_mlp_reg(**params):
    """MLP de regressão com entradas e alvo padronizados dentro do pipeline."""
    rede = Pipeline([("sc", StandardScaler()), ("mlp", MLPRegressor(**MLP_FIXOS, **params))])
    return TransformedTargetRegressor(regressor=rede, transformer=StandardScaler())


def criar_mlp_clf(**params):
    """MLP de classificação (softmax) com entradas padronizadas dentro do pipeline."""
    return Pipeline([("sc", StandardScaler()), ("mlp", MLPClassifier(**MLP_FIXOS, **params))])


def fmt(x, casas=4):
    """Número com vírgula decimal, para texto e tabelas em português."""
    return f"{x:.{casas}f}".replace(".", ",")


def tabela_md(cabecalho, linhas):
    """Linhas de uma tabela Markdown."""
    return [
        "| " + " | ".join(cabecalho) + " |",
        "|" + "|".join("---" for _ in cabecalho) + "|",
        *("| " + " | ".join(str(c) for c in linha) + " |" for linha in linhas),
    ]

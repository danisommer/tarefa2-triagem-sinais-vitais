"""Fase 1 — análise exploratória: confere os fatos esperados e gera fig1, eda_resumo.md e ambiente.json.

Termina com código 1 se algum fato não conferir (os arquivos de saída são gravados mesmo assim).
"""

import json
import platform
import sys
from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate

from src.comum import (
    ALVO_CLF,
    ALVO_REG,
    ARQ_COM_LABEL,
    ARQ_SEM_LABEL,
    ATRIBUTOS,
    CLASSES,
    COLUNAS_COM_LABEL,
    COLUNAS_SEM_LABEL,
    CORES_SERIES,
    ESTILO_FIGURA,
    FAIXAS_ENUNCIADO,
    FIGURAS,
    LIMIARES_HIPOTESE,
    NOMES_CLASSES,
    RESULTADOS,
    SEED,
    SUPERFICIE,
    TINTA_SECUNDARIA,
    classe_por_limiares,
    criar_mlp_clf,
    criar_mlp_reg,
    criar_rf_clf,
    criar_rf_reg,
    dividir_hold_out,
    estimar_limiares,
    fmt,
    ler_rotulados,
    ler_txt,
    limiares_arvore,
    tabela_md,
)

BIBLIOTECAS = ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib", "scipy"]

FATO_N_LINHAS = 1500
FATO_CLASSES = {1: 234, 2: 832, 3: 404, 4: 30}
FATO_FAIXAS_G = {1: (13.2227, 24.9784), 2: (25.0249, 49.9977), 3: (50.0296, 74.6574), 4: (75.1626, 87.0)}
FATO_PLATOS_G = [(40.0, 90), (13.2227, 41), (25.1705, 26), (27.1226, 23)]
FATO_PLATOS_QPA = [(8.7333, 197), (0.0, 193), (4.6667, 79), (-4.3333, 61)]
FATO_FAIXA_QPA = (-8.733, 8.733)
FATO_DIF_GRAVID = 1.3e-5
FATO_R2_QPA = 0.997
PISO_R2 = 0.96
PISO_ACURACIA = 0.89

TOL_4_CASAS = 5e-5
TOL_3_CASAS = 5e-4
LIMITE_DIF_GRAVID = 1.35e-5
TOL_R2_QPA = 0.002

CRITERIOS = (
    "Critérios: valores com 3 ou 4 casas conferem se forem iguais após o arredondamento; "
    f"Δgravid conferem se o máximo arredondar para até 1,3e-05 (< {LIMITE_DIF_GRAVID:.2e}); "
    f"R² do qPA ±{fmt(TOL_R2_QPA, 3)}; baseline confere se ficar acima dos pisos de sanidade "
    f"(R² ≥ {fmt(PISO_R2, 2)} e acurácia ≥ {fmt(PISO_ACURACIA, 2)})."
)

CORES_CLASSES = dict(zip(CLASSES, CORES_SERIES))

FAIXAS_G_MEDIO = {
    "pulso": ([0, 25, 50, 75, 100, 125, 150, 175, 200], True),
    "resp": ([0, 4, 8, 12, 16, 20, 22], True),
    "qPA": ([-9, -6, -3, -0.5, 0.5, 3, 6, 9], False),
}


def checar(fato, esperado, obtido, ok):
    """Registro de uma checagem de fato esperado."""
    return {"fato": fato, "esperado": esperado, "obtido": obtido, "ok": bool(ok)}


def sim_nao(valor):
    return "sim" if valor else "não"


def media_dp(valores, casas=4):
    return f"{fmt(np.mean(valores), casas)} ± {fmt(np.std(valores), casas)}"


def separador_consistente(linha, separador, n):
    """A linha tem exatamente n separadores e nenhum outro espaço."""
    return linha.count(",") == n and linha.count(separador) == n and " " not in linha.replace(separador, "")


def resumo_arquivo(caminho, separador, n_separadores):
    """Contagens do arquivo bruto: linhas, linhas vazias, separador consistente e ausência de cabeçalho."""
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    return {
        "linhas": len(linhas),
        "vazias": sum(not linha.strip() for linha in linhas),
        "separador": all(separador_consistente(linha, separador, n_separadores) for linha in linhas),
        "sem_cabecalho": linhas[0].split(",")[0].strip().isdigit(),
    }


def analisar_formato():
    r02 = resumo_arquivo(ARQ_COM_LABEL, ",", len(COLUNAS_COM_LABEL) - 1)
    r01 = resumo_arquivo(ARQ_SEM_LABEL, ", ", len(COLUNAS_SEM_LABEL) - 1)
    corpo = [
        "## 1. Formato dos arquivos",
        "",
        *tabela_md(
            ["Arquivo", "Linhas", "Linhas vazias", "Colunas", "Separador", "Separador em todas as linhas", "Cabeçalho"],
            [
                [ARQ_COM_LABEL.name, r02["linhas"], r02["vazias"], len(COLUNAS_COM_LABEL), "`,`", sim_nao(r02["separador"]), sim_nao(not r02["sem_cabecalho"])],
                [ARQ_SEM_LABEL.name, r01["linhas"], r01["vazias"], len(COLUNAS_SEM_LABEL), "`, `", sim_nao(r01["separador"]), sim_nao(not r01["sem_cabecalho"])],
            ],
        ),
        "",
        f"Colunas do 02: `{','.join(COLUNAS_COM_LABEL)}`. A ordem é confirmada pelas faixas da seção 5, que batem com os intervalos do enunciado.",
    ]
    checagens = [
        checar("02: 1500 linhas, sem cabeçalho", "1500, sem", f"{r02['linhas']}, {'sem' if r02['sem_cabecalho'] else 'com'}", r02["linhas"] == FATO_N_LINHAS and r02["sem_cabecalho"]),
        checar("02: 8 colunas separadas por `,`", "sim", sim_nao(r02["separador"]), r02["separador"]),
        checar("01: 1500 linhas, sem cabeçalho", "1500, sem", f"{r01['linhas']}, {'sem' if r01['sem_cabecalho'] else 'com'}", r01["linhas"] == FATO_N_LINHAS and r01["sem_cabecalho"]),
        checar("01: 7 colunas (sem `classe`) separadas por `, `", "sim", sim_nao(r01["separador"]), r01["separador"]),
    ]
    return corpo, checagens


def analisar_consistencia(df, df01):
    brutas = COLUNAS_SEM_LABEL[:-1]
    ids_sequenciais = df["id"].tolist() == list(range(1, len(df) + 1))
    mesmas_vitimas = ids_sequenciais and df["id"].equals(df01["id"])
    entradas_iguais = df[brutas].equals(df01[brutas])
    dif = (df[ALVO_REG] - df01[ALVO_REG]).abs()
    corpo = [
        "## 2. Consistência entre o 01 e o 02",
        "",
        f"- Mesmos ids (1 a {len(df)}), na mesma ordem: {sim_nao(mesmas_vitimas)}.",
        f"- `{', '.join(brutas)}` idênticos nos dois arquivos: {sim_nao(entradas_iguais)}.",
        f"- `gravid`: {int((dif > 0).sum())} de {len(df)} linhas diferem; diferença absoluta máxima = {dif.max():.2e}, média = {dif.mean():.2e}.",
        "- Conclusão: o 01 é o 02 sem a coluna `classe`, com `gravid` arredondado de outra forma. Só o 02 é usado daqui em diante.",
    ]
    checagens = [
        checar("01 × 02: mesmas vítimas e entradas idênticas", "sim", sim_nao(mesmas_vitimas and entradas_iguais), mesmas_vitimas and entradas_iguais),
        checar("01 × 02: diferença máxima em `gravid`", f"≤ {FATO_DIF_GRAVID:.1e}", f"{dif.max():.2e}", dif.max() < LIMITE_DIF_GRAVID),
    ]
    return corpo, checagens


def analisar_faltantes(df, df01):
    n02, n01 = int(df.isna().sum().sum()), int(df01.isna().sum().sum())
    corpo = ["## 3. Valores faltantes", "", f"02: {n02} · 01: {n01}."]
    return corpo, [checar("Sem valores faltantes (02 e 01)", "0 e 0", f"{n02} e {n01}", n02 == 0 and n01 == 0)]


def analisar_classes(df):
    contagem = df[ALVO_CLF].value_counts().reindex(CLASSES, fill_value=0)
    corpo = [
        "## 4. Distribuição das classes",
        "",
        *tabela_md(
            ["Classe", "Nome", "Vítimas", "%"],
            [[k, NOMES_CLASSES[k], contagem[k], fmt(100 * contagem[k] / len(df), 1)] for k in CLASSES],
        ),
        "",
        f"Razão entre a maior e a menor classe: {fmt(contagem.max() / contagem.min(), 1)}. A classe 4 tem só {contagem[4]} vítimas.",
    ]
    obtido = " · ".join(f"{k} = {contagem[k]}" for k in CLASSES)
    esperado = " · ".join(f"{k} = {n}" for k, n in FATO_CLASSES.items())
    ok = contagem.to_dict() == FATO_CLASSES and contagem.sum() == len(df)
    return corpo, [checar("Contagem por classe", esperado, obtido, ok)]


def analisar_atributos(df):
    colunas = ["pSist", "pDiast", *ATRIBUTOS, ALVO_REG]
    uso = {"pSist": "proibido", "pDiast": "proibido", ALVO_REG: "alvo"} | {a: "entrada" for a in ATRIBUTOS}
    linhas = [
        [
            f"`{c}`",
            uso[c],
            fmt(df[c].min()),
            fmt(df[c].max()),
            fmt(df[c].mean()),
            fmt(df[c].std()),
            *intervalo_enunciado(df[c], FAIXAS_ENUNCIADO.get(c)),
        ]
        for c in colunas
    ]
    spearman = df[ATRIBUTOS].corrwith(df[ALVO_REG], method="spearman")
    pearson = df[ATRIBUTOS].corrwith(df[ALVO_REG], method="pearson")
    qpa_min, qpa_max = df["qPA"].min(), df["qPA"].max()
    corpo = [
        "## 5. Atributos",
        "",
        *tabela_md(["Atributo", "Uso", "Mín", "Máx", "Média", "Desvio", "Intervalo do enunciado", "Fora do intervalo"], linhas),
        "",
        "Correlação de cada entrada com `gravid`:",
        "",
        *tabela_md(["Entrada", "Spearman", "Pearson"], [[f"`{a}`", fmt(spearman[a], 3), fmt(pearson[a], 3)] for a in ATRIBUTOS]),
    ]
    ok = abs(qpa_min - FATO_FAIXA_QPA[0]) < TOL_3_CASAS and abs(qpa_max - FATO_FAIXA_QPA[1]) < TOL_3_CASAS
    esperado = f"[{fmt(FATO_FAIXA_QPA[0], 3)}; {fmt(FATO_FAIXA_QPA[1], 3)}]"
    return corpo, [checar("Faixa de `qPA`", esperado, f"[{fmt(qpa_min, 3)}; {fmt(qpa_max, 3)}]", ok)]


def intervalo_enunciado(serie, faixa):
    """Intervalo do enunciado e quantos valores ficam fora dele; '—' quando o enunciado não define."""
    if faixa is None:
        return ["—", "—"]
    return [f"[{fmt(faixa[0], 0)}; {fmt(faixa[1], 0)}]", int((~serie.between(*faixa)).sum())]


def analisar_gravidade(df):
    g, classe = df[ALVO_REG], df[ALVO_CLF]
    por_classe = df.groupby(ALVO_CLF)[ALVO_REG].agg(["count", "min", "max", "mean"])
    sem_sobreposicao = all(por_classe.loc[k, "max"] < por_classe.loc[k + 1, "min"] for k in CLASSES[:-1])
    acertos = int((classe_por_limiares(g, LIMIARES_HIPOTESE) == classe).sum())
    medios = estimar_limiares(g, classe)
    arvore = limiares_arvore(g, classe)
    arvore_confere = len(arvore) == len(medios) and np.allclose(arvore, medios, atol=1e-4)
    folgas = [por_classe.loc[k + 1, "min"] - por_classe.loc[k, "max"] for k in CLASSES[:-1]]
    corpo = [
        "## 6. Gravidade por classe",
        "",
        "Escala invertida: g baixo = estado mais grave.",
        "",
        *tabela_md(
            ["Classe", "Nome", "Vítimas", "g mín", "g máx", "g médio"],
            [[k, NOMES_CLASSES[k], int(por_classe.loc[k, "count"]), fmt(por_classe.loc[k, "min"]), fmt(por_classe.loc[k, "max"]), fmt(por_classe.loc[k, "mean"])] for k in CLASSES],
        ),
        "",
        f"Cortes 25 / 50 / 75 (g < 25 → 1; 25 ≤ g < 50 → 2; 50 ≤ g < 75 → 3; g ≥ 75 → 4): {acertos} de {len(df)} acertos. Faixas sem sobreposição: {sim_nao(sem_sobreposicao)}.",
        "",
        "Limiares estimados nos 1500 (só descritivo; na Fase 2 eles saem só das partes de treino):",
        "",
        *tabela_md(
            ["Corte", "Hipótese", "Ponto médio", "Árvore prof. 2 sobre g", "Folga entre as classes"],
            [[f"{k}→{k + 1}", fmt(h, 0), fmt(m), fmt(a), fmt(f)] for k, h, m, a, f in zip(CLASSES, LIMIARES_HIPOTESE, medios, arvore, folgas)],
        ),
        "",
        f"Árvore e pontos médios coincidem (tolerância 1e-4): {sim_nao(arvore_confere)}. "
        "A folga é a distância entre o maior g de uma classe e o menor g da seguinte: um erro de regressão "
        "menor que ela, perto do corte, já basta para mudar a classe.",
        "",
        "Figura: `figuras/fig1_gravidade_por_classe.png`.",
    ]
    checagens = [
        checar(
            f"Classe {k}: faixa de g",
            f"[{fmt(a)}; {fmt(b)}]",
            f"[{fmt(por_classe.loc[k, 'min'])}; {fmt(por_classe.loc[k, 'max'])}]",
            abs(por_classe.loc[k, "min"] - a) < TOL_4_CASAS and abs(por_classe.loc[k, "max"] - b) < TOL_4_CASAS,
        )
        for k, (a, b) in FATO_FAIXAS_G.items()
    ]
    checagens += [
        checar("Faixas de g das classes sem sobreposição", "sim", sim_nao(sem_sobreposicao), sem_sobreposicao),
        checar("Cortes 25 / 50 / 75 reproduzem a classe", f"{len(df)} de {len(df)}", f"{acertos} de {len(df)}", acertos == len(df)),
    ]
    return corpo, checagens


def maiores_platos(serie, n):
    """Os n valores mais repetidos (arredondados a 4 casas) com suas contagens."""
    contagem = serie.round(4).value_counts().head(n)
    return list(zip(contagem.index.tolist(), contagem.tolist()))


def descrever_platos(platos):
    return " · ".join(f"{fmt(v)} ({n})" for v, n in platos)


def platos_conferem(obtidos, esperados):
    return len(obtidos) == len(esperados) and all(
        abs(vo - ve) < TOL_4_CASAS and no == ne for (vo, no), (ve, ne) in zip(obtidos, esperados)
    )


def analisar_platos(df):
    g, qpa = df[ALVO_REG], df["qPA"]
    platos_g, platos_qpa = maiores_platos(g, 6), maiores_platos(qpa, 6)
    corpo = [
        "## 7. Platôs (valores repetidos)",
        "",
        f"`gravid`: {g.round(4).nunique()} valores distintos em {len(df)} vítimas. `qPA`: {qpa.round(4).nunique()}.",
        "",
        *tabela_md(["`gravid`", "Vítimas", "`qPA`", "Vítimas"], [[fmt(vg), ng, fmt(vq), nq] for (vg, ng), (vq, nq) in zip(platos_g, platos_qpa)]),
        "",
        f"O platô 25,1705 fica a {fmt(25.1705 - LIMIARES_HIPOTESE[0], 4)} do corte 25.",
        "Hipótese (não verificada): valores gerados por sistema fuzzy com defuzzificação por centroide.",
    ]
    n_fatos = len(FATO_PLATOS_G)
    checagens = [
        checar("Maiores platôs de g", descrever_platos(FATO_PLATOS_G), descrever_platos(platos_g[:n_fatos]), platos_conferem(platos_g[:n_fatos], FATO_PLATOS_G)),
        checar("13,2227 é o mínimo de g", fmt(FATO_PLATOS_G[1][0]), fmt(g.min()), abs(g.min() - FATO_PLATOS_G[1][0]) < TOL_4_CASAS),
        checar("Maiores platôs de qPA", descrever_platos(FATO_PLATOS_QPA), descrever_platos(platos_qpa[:n_fatos]), platos_conferem(platos_qpa[:n_fatos], FATO_PLATOS_QPA)),
    ]
    return corpo, checagens


def analisar_qpa(df):
    r2 = cross_val_score(
        criar_rf_reg(),
        df[["pSist", "pDiast"]],
        df["qPA"],
        cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="r2",
        n_jobs=-1,
    )
    corpo = [
        "## 8. `qPA` como função de (`pSist`, `pDiast`)",
        "",
        f"Random Forest padrão, `KFold(5, shuffle=True, random_state={SEED})`, nos 1500: R² = {media_dp(r2)}.",
        "Análise só descritiva: mostra que `qPA` resume as duas pressões. `pSist` e `pDiast` não entram em nenhum modelo.",
    ]
    ok = abs(np.mean(r2) - FATO_R2_QPA) <= TOL_R2_QPA
    return corpo, [checar("R² de qPA ~ (pSist, pDiast) com RF", fmt(FATO_R2_QPA, 3), fmt(np.mean(r2), 3), ok)]


def avaliar_regressores(modelos, X, g, dobras):
    """RMSE e R² por dobra de cada regressor."""
    resultados = {}
    for nome, modelo in modelos.items():
        r = cross_validate(modelo, X, g, cv=dobras, scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"}, n_jobs=-1)
        resultados[nome] = {"rmse": -r["test_rmse"], "r2": r["test_r2"]}
    return resultados


def avaliar_classificadores(modelos, X, y, dobras):
    """Acurácia por dobra de cada classificador."""
    return {nome: cross_validate(m, X, y, cv=dobras, scoring="accuracy", n_jobs=-1)["test_score"] for nome, m in modelos.items()}


def analisar_baseline(dev):
    X, g, y = dev[ATRIBUTOS], dev[ALVO_REG], dev[ALVO_CLF]
    dobras = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
    reg = avaliar_regressores(
        {
            "RF": criar_rf_reg(),
            "MLP (64, 32)": criar_mlp_reg(hidden_layer_sizes=(64, 32)),
            "MLP (64, 32) + early_stopping": criar_mlp_reg(hidden_layer_sizes=(64, 32), early_stopping=True),
        },
        X, g, dobras,
    )
    clf = avaliar_classificadores({"RF": criar_rf_clf(), "MLP (64, 32)": criar_mlp_clf(hidden_layer_sizes=(64, 32))}, X, y, dobras)
    rmse_sem, rmse_com = np.mean(reg["MLP (64, 32)"]["rmse"]), np.mean(reg["MLP (64, 32) + early_stopping"]["rmse"])
    corpo = [
        "## 9. Baseline sem ajuste (sanidade)",
        "",
        f"Só no conjunto de desenvolvimento ({len(dev)} vítimas, 80% estratificado com `SEED = {SEED}`, a mesma divisão da Fase 2), "
        "para que nenhum modelo veja o hold-out antes do fim. `StratifiedKFold(5)` pela classe, também na regressão. "
        "RF com hiperparâmetros padrão do scikit-learn; MLP com os parâmetros fixos (`adam`, `max_iter=3000`, `n_iter_no_change=30`).",
        "",
        *tabela_md(["Regressor", "RMSE", "R²"], [[nome, media_dp(r["rmse"]), media_dp(r["r2"])] for nome, r in reg.items()]),
        "",
        *tabela_md(["Classificador (direto)", "Acurácia"], [[nome, media_dp(a)] for nome, a in clf.items()]),
        "",
        f"`early_stopping=True` no MLP (64, 32): RMSE {fmt(rmse_com, 2)} com × {fmt(rmse_sem, 2)} sem. "
        "Só informativo: ele entra na grade da Fase 2.",
    ]
    checagens = [
        checar(
            f"Baseline {nome} regressão: R² ≥ {fmt(PISO_R2, 2)}",
            f"≥ {fmt(PISO_R2, 2)}",
            fmt(np.mean(reg[nome]["r2"]), 4),
            np.mean(reg[nome]["r2"]) >= PISO_R2,
        )
        for nome in ("RF", "MLP (64, 32)")
    ]
    checagens += [
        checar(
            f"Baseline {nome} classificação: acurácia ≥ {fmt(PISO_ACURACIA, 2)}",
            f"≥ {fmt(PISO_ACURACIA, 2)}",
            fmt(np.mean(a), 4),
            np.mean(a) >= PISO_ACURACIA,
        )
        for nome, a in clf.items()
    ]
    return corpo, checagens


def num_curto(x):
    """Número sem zeros à direita e com vírgula decimal (25 → '25'; -0.5 → '-0,5')."""
    return f"{x:g}".replace(".", ",")


def rotulos_faixas(bordas, inclui_menor):
    """Rótulos das faixas do pd.cut, fechadas à direita; com include_lowest, a primeira fecha também à esquerda."""
    aberturas = ["[" if inclui_menor else "("] + ["("] * (len(bordas) - 2)
    return [f"{a}{num_curto(e)}; {num_curto(d)}]" for a, e, d in zip(aberturas, bordas, bordas[1:])]


def g_por_faixa(df, atributo, bordas, inclui_menor):
    """n e g médio por faixa do atributo; levanta ValueError se algum valor ficar fora das faixas."""
    faixa = pd.cut(df[atributo], bordas, include_lowest=inclui_menor, labels=rotulos_faixas(bordas, inclui_menor))
    if faixa.isna().any():
        raise ValueError(f"{int(faixa.isna().sum())} valores de {atributo} fora das faixas {bordas}")
    return df.groupby(faixa, observed=False)[ALVO_REG].agg(n="size", g_medio="mean")


def analisar_g_por_faixa(df):
    corpo = [
        "## 10. g médio por faixa de cada entrada",
        "",
        f"Só descritivo, nos {len(df)}. Faixas fechadas à direita; a primeira de `pulso` e de `resp` inclui também o limite "
        f"inferior. g médio geral = {fmt(df[ALVO_REG].mean())}. Figura: `figuras/fig5_g_por_faixa.png`.",
    ]
    for atributo, (bordas, inclui_menor) in FAIXAS_G_MEDIO.items():
        resumo = g_por_faixa(df, atributo, bordas, inclui_menor)
        linhas = [[faixa, int(r["n"]), fmt(r["g_medio"])] for faixa, r in resumo.iterrows()]
        corpo += ["", f"`{atributo}`:", "", *tabela_md(["Faixa", "n", "g médio"], linhas)]
    return corpo, []


def escrever_resumo(secoes, checagens, caminho):
    conferem = sum(c["ok"] for c in checagens)
    linhas = [
        "# Fase 1 — Análise exploratória",
        "",
        "Gerado por `python -m src.eda`; não editar à mão. Análise sobre o arquivo 02; o 01 entra só na seção 2.",
        "",
        "## Conferência dos fatos esperados",
        "",
        f"**{conferem} de {len(checagens)} fatos conferem.**",
        "",
        *tabela_md(
            ["#", "Fato", "Esperado", "Obtido", "Confere"],
            [[i, c["fato"], c["esperado"], c["obtido"], "sim" if c["ok"] else "**NÃO**"] for i, c in enumerate(checagens, 1)],
        ),
        "",
        CRITERIOS,
    ]
    for corpo, _ in secoes:
        linhas += ["", *corpo]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def figura_gravidade(df, caminho):
    """Histograma de g por classe (faixas de largura 1) com os cortes 25/50/75 e rótulos diretos."""
    g, classe = df[ALVO_REG], df[ALVO_CLF]
    bordas = np.arange(np.floor(g.min()), np.ceil(g.max()) + 1)
    pico = np.histogram(g, bordas)[0].max()
    limites = [g.min(), *LIMIARES_HIPOTESE, g.max()]
    with plt.rc_context(ESTILO_FIGURA):
        fig, ax = plt.subplots(figsize=(3.4, 2.5), layout="constrained")
        ax.hist(
            [g[classe == k] for k in CLASSES],
            bins=bordas,
            stacked=True,
            color=[CORES_CLASSES[k] for k in CLASSES],
            edgecolor=SUPERFICIE,
            linewidth=0.4,
            label=[f"{k} – {NOMES_CLASSES[k]} (n = {(classe == k).sum()})" for k in CLASSES],
        )
        for corte in LIMIARES_HIPOTESE:
            ax.axvline(corte, color=TINTA_SECUNDARIA, linestyle=(0, (3, 2)), linewidth=0.8, zorder=3)
        for k, (a, b) in zip(CLASSES, zip(limites, limites[1:])):
            ax.text((a + b) / 2, pico * 1.27, str(k), ha="center", va="top", fontweight="bold")
        ax.set(xlim=(10, 90), ylim=(0, pico * 1.3), xticks=[10, 25, 50, 75, 90])
        ax.set_xlabel("Gravidade (g)")
        ax.set_ylabel("Nº de vítimas")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, handlelength=1, columnspacing=1)
        fig.savefig(caminho, dpi=300, facecolor=SUPERFICIE)
        plt.close(fig)


def figura_g_por_faixa(df, caminho):
    """g médio por faixa de cada entrada (degraus com o ponto no centro da faixa), com a média geral tracejada."""
    media_geral = df[ALVO_REG].mean()
    estilo = ESTILO_FIGURA | {"axes.spines.left": True, "ytick.left": True}
    with plt.rc_context(estilo):
        fig, eixos = plt.subplots(1, len(FAIXAS_G_MEDIO), figsize=(3.4, 1.8), sharey=True, layout="constrained")
        for ax, (atributo, (bordas, inclui_menor)) in zip(eixos, FAIXAS_G_MEDIO.items()):
            medias = g_por_faixa(df, atributo, bordas, inclui_menor)["g_medio"].to_numpy()
            centros = (np.array(bordas[:-1]) + np.array(bordas[1:])) / 2
            ax.axhline(media_geral, color=TINTA_SECUNDARIA, linestyle=(0, (3, 2)), linewidth=0.6)
            ax.stairs(medias, bordas, baseline=None, color=CORES_SERIES[0], linewidth=1.2)
            ax.scatter(centros, medias, s=10, color=CORES_SERIES[0], edgecolors=SUPERFICIE, linewidths=0.4, zorder=3)
            ax.set_xticks([bordas[0], (bordas[0] + bordas[-1]) / 2, bordas[-1]])
            ax.set_xlabel(atributo)
        eixos[0].set_ylabel("g médio")
        eixos[1].text(
            FAIXAS_G_MEDIO["resp"][0][0] + 0.5, media_geral + 0.5, "média\ngeral",
            ha="left", va="bottom", fontsize=6, color=TINTA_SECUNDARIA,
        )
        fig.savefig(caminho, dpi=300, facecolor=SUPERFICIE)
        plt.close(fig)


def salvar_ambiente(caminho):
    info = {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "bibliotecas": {b: version(b) for b in BIBLIOTECAS},
    }
    caminho.write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    FIGURAS.mkdir(parents=True, exist_ok=True)
    df = ler_rotulados()
    df01 = ler_txt(ARQ_SEM_LABEL, COLUNAS_SEM_LABEL)
    dev, _ = dividir_hold_out(df)
    secoes = [
        analisar_formato(),
        analisar_consistencia(df, df01),
        analisar_faltantes(df, df01),
        analisar_classes(df),
        analisar_atributos(df),
        analisar_gravidade(df),
        analisar_platos(df),
        analisar_qpa(df),
        analisar_baseline(dev),
        analisar_g_por_faixa(df),
    ]
    checagens = [c for _, cs in secoes for c in cs]
    escrever_resumo(secoes, checagens, RESULTADOS / "eda_resumo.md")
    figura_gravidade(df, FIGURAS / "fig1_gravidade_por_classe.png")
    figura_g_por_faixa(df, FIGURAS / "fig5_g_por_faixa.png")
    salvar_ambiente(RESULTADOS / "ambiente.json")
    for c in checagens:
        print(f"[{'ok' if c['ok'] else 'NÃO'}] {c['fato']}: esperado {c['esperado']}, obtido {c['obtido']}")
    falhas = [c["fato"] for c in checagens if not c["ok"]]
    if falhas:
        sys.exit(f"\n{len(falhas)} fato(s) esperado(s) não conferem: {'; '.join(falhas)}")
    print(f"\nTodos os {len(checagens)} fatos conferem.")


if __name__ == "__main__":
    main()

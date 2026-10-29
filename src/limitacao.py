"""Efeito de limitar ĝ à faixa de g do treino: RMSE, MAE e R² sem e com a limitação, na CV e no teste.

Só lê as predições gravadas pela Fase 2 (não re-treina nada) e grava resultados/limitacao_saida.csv.
Levanta RuntimeError se as classes em 2 estágios ou as métricas sem limitação não reproduzirem as da Fase 2.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from src.comum import RESULTADOS, classe_por_limiares, dividir_hold_out, faixa_g, ler_rotulados, limitar_g
from src.experimentos import COLUNAS_LIMIARES, REGRESSORES

METRICAS = {"rmse": root_mean_squared_error, "mae": mean_absolute_error, "r2": r2_score}
COLUNAS_METRICAS = [f"{m}_{versao}" for m in METRICAS for versao in ("sem", "com")]
TOL_REPRODUCAO = 1e-9


def com_limites_cv(pred):
    """Acrescenta a cada predição da CV o menor e o maior g da parte de treino da sua dobra (as outras dobras da mesma repetição)."""
    limites = [
        {
            "repeticao": r,
            "dobra": d,
            "g_min_treino": pred.loc[(pred["repeticao"] == r) & (pred["dobra"] != d), "g"].min(),
            "g_max_treino": pred.loc[(pred["repeticao"] == r) & (pred["dobra"] != d), "g"].max(),
        }
        for r, d in pred[["repeticao", "dobra"]].drop_duplicates().itertuples(index=False)
    ]
    return pred.merge(pd.DataFrame(limites), on=["repeticao", "dobra"], how="left", validate="many_to_one")


def avaliar(parte, nome):
    """Métricas sem e com limitação, previsões alteradas e classes em 2 estágios alteradas numa dobra (ou no teste)."""
    g, g_pred = parte["g"].to_numpy(), parte[f"g_pred_{nome}"].to_numpy()
    limitado, alteradas = limitar_g(g_pred, parte["g_min_treino"].to_numpy(), parte["g_max_treino"].to_numpy())
    limiares = parte[COLUNAS_LIMIARES].iloc[0].to_numpy()
    classes_sem, classes_com = classe_por_limiares(g_pred, limiares), classe_por_limiares(limitado, limiares)
    if not np.array_equal(classes_sem, parte[f"{nome} 2 estágios"].to_numpy()):
        raise RuntimeError(f"as classes de {nome} 2 estágios recalculadas não reproduzem as gravadas")
    return {
        **{f"{m}_sem": f(g, g_pred) for m, f in METRICAS.items()},
        **{f"{m}_com": f(g, limitado) for m, f in METRICAS.items()},
        "previsoes_alteradas": int(alteradas.sum()),
        "classes_2_estagios_alteradas": int((classes_com != classes_sem).sum()),
    }


def resumir(conjunto, partes):
    """Uma linha por regressor: métricas médias (com desvio ddof=0 quando há várias dobras) e contagens somadas."""
    linhas = []
    for nome in REGRESSORES:
        por_parte = pd.DataFrame([avaliar(p, nome) for p in partes])
        linha = {
            "conjunto": conjunto,
            "regressor": nome,
            "n_predicoes": sum(len(p) for p in partes),
            "previsoes_alteradas": int(por_parte["previsoes_alteradas"].sum()),
            "classes_2_estagios_alteradas": int(por_parte["classes_2_estagios_alteradas"].sum()),
        }
        for coluna in COLUNAS_METRICAS:
            linha[coluna] = por_parte[coluna].mean()
            linha[f"{coluna}_desvio"] = por_parte[coluna].std(ddof=0) if len(partes) > 1 else np.nan
        linhas.append(linha)
    return linhas


def conferir_fase2(saida):
    """Levanta RuntimeError se as métricas sem limitação não forem as de metricas_cv_resumo.csv e metricas_teste.csv."""
    cv = pd.read_csv(RESULTADOS / "metricas_cv_resumo.csv").set_index(["variante", "metrica"])
    teste = pd.read_csv(RESULTADOS / "metricas_teste.csv").set_index(["variante", "metrica"])["valor"]
    referencia = {("cv", n, m, ""): cv.loc[(n, m), "media"] for n in REGRESSORES for m in METRICAS}
    referencia |= {("cv", n, m, "_desvio"): cv.loc[(n, m), "desvio"] for n in REGRESSORES for m in METRICAS}
    referencia |= {("teste", n, m, ""): teste.loc[(n, m)] for n in REGRESSORES for m in METRICAS}
    obtido = saida.set_index(["conjunto", "regressor"])
    divergentes = [
        f"{c} {n} {m}_sem{s}"
        for (c, n, m, s), valor in referencia.items()
        if abs(obtido.loc[(c, n), f"{m}_sem{s}"] - valor) > TOL_REPRODUCAO
    ]
    if divergentes:
        raise RuntimeError(f"métricas sem limitação diferentes das da Fase 2: {divergentes}")


def main():
    dev, _ = dividir_hold_out(ler_rotulados())
    minimo_dev, maximo_dev = faixa_g(dev)
    limiares_dev = json.loads((RESULTADOS / "limiares.json").read_text(encoding="utf-8"))["desenvolvimento"]["ponto_medio"]
    cv = com_limites_cv(pd.read_csv(RESULTADOS / "predicoes_cv.csv"))
    teste = pd.read_csv(RESULTADOS / "predicoes_teste.csv").assign(
        g_min_treino=minimo_dev, g_max_treino=maximo_dev, **dict(zip(COLUNAS_LIMIARES, limiares_dev))
    )
    partes_cv = [dobra for _, dobra in cv.groupby(["repeticao", "dobra"])]
    saida = pd.DataFrame([*resumir("cv", partes_cv), *resumir("teste", [teste])])
    conferir_fase2(saida)
    colunas = [
        "conjunto", "regressor", "n_predicoes", "previsoes_alteradas", "classes_2_estagios_alteradas",
        *(c for coluna in COLUNAS_METRICAS for c in (coluna, f"{coluna}_desvio")),
    ]
    saida[colunas].to_csv(RESULTADOS / "limitacao_saida.csv", index=False)
    print(saida[colunas].to_string(index=False))
    print("Métricas sem limitação conferem com as da Fase 2; classes em 2 estágios alteradas: "
          f"{int(saida['classes_2_estagios_alteradas'].sum())}.")


if __name__ == "__main__":
    main()

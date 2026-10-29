"""Fase 2 — busca de hiperparâmetros, validação cruzada repetida, teste no hold-out, figuras 2 a 4, custos, tabelas e modelos finais."""

import json
import time

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib.colors import LinearSegmentedColormap
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

from src.comum import (
    ALVO_CLF,
    ALVO_REG,
    ATRIBUTOS,
    CLASSES,
    COR_EIXO,
    CORES_SERIES,
    ESTILO_FIGURA,
    FIGURAS,
    GRADE_MLP,
    GRADE_RF,
    GRADE_RF_CLF,
    LIMIARES_HIPOTESE,
    MLP_FIXOS,
    MODELOS,
    N_JOBS,
    RAMPA_AZUL,
    RESULTADOS,
    SEED,
    SUPERFICIE,
    TINTA,
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
    limiares_arvore,
    tabela_md,
)

N_DOBRAS = 5
N_REPETICOES = 5
N_ITER_MLP = 40
N_REPETICOES_PERMUTACAO = 30
N_REPETICOES_LATENCIA = 20

REGRESSORES = ["RF", "MLP"]
VARIANTES = ["RF direta", "RF 2 estágios", "MLP direta", "MLP 2 estágios"]
MODELOS_FINAIS = ["RF_reg", "RF_clf", "MLP_reg", "MLP_clf"]
VARIANTES_PESO_MLP = ["MLP_clf sem peso", "MLP_clf balanceado"]

SCORING = {"reg": "neg_root_mean_squared_error", "clf": "f1_macro"}
METRICA_SELECAO = {"reg": ("rmse", -1), "clf": ("f1_macro", 1)}

FAIXAS_DISTANCIA = [0.0, 0.5, 1.0, 2.5, 5.0, np.inf]
ROTULOS_DISTANCIA = ["0–0,5", "0,5–1", "1–2,5", "2,5–5", "> 5"]

COLUNAS_LIMIARES = ["limiar_1_2", "limiar_2_3", "limiar_3_4"]
COLUNAS_PREDICAO = ["id", "g", "classe", *(f"g_pred_{n}" for n in REGRESSORES), *VARIANTES]
DELTAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
VARIANTES_2_ESTAGIOS = {f"{n} 2 estágios": n for n in REGRESSORES}
METRICAS_CONSERVADOR = ["subtriagem_taxa", "sobretriagem_taxa", "recall_1", "acuracia", "f1_macro"]
TOL_EXTRAPOLACAO = 1e-6

MAPA_AZUL = LinearSegmentedColormap.from_list("azul", RAMPA_AZUL)
FRACAO_TEXTO_CLARO = 0.6


def prefixar(grade, prefixo):
    return {prefixo + k: v for k, v in grade.items()}


def especificacoes():
    """Cada busca: estimador base, grade, tarefa, se é aleatória e se o fit usa peso balanceado."""
    return {
        "RF_reg": {"estimador": criar_rf_reg(), "grade": GRADE_RF, "tarefa": "reg", "aleatoria": False, "peso": False},
        "RF_clf": {"estimador": criar_rf_clf(), "grade": GRADE_RF_CLF, "tarefa": "clf", "aleatoria": False, "peso": False},
        "MLP_reg": {
            "estimador": criar_mlp_reg(),
            "grade": prefixar(GRADE_MLP, "regressor__mlp__"),
            "tarefa": "reg",
            "aleatoria": True,
            "peso": False,
        },
        "MLP_clf sem peso": {
            "estimador": criar_mlp_clf(),
            "grade": prefixar(GRADE_MLP, "mlp__"),
            "tarefa": "clf",
            "aleatoria": True,
            "peso": False,
        },
        "MLP_clf balanceado": {
            "estimador": criar_mlp_clf(),
            "grade": prefixar(GRADE_MLP, "mlp__"),
            "tarefa": "clf",
            "aleatoria": True,
            "peso": True,
        },
    }


def alvo(tarefa, g, y):
    return {"reg": g, "clf": y}[tarefa]


def ajustar(estimador, X, y, peso):
    """Ajusta o estimador; com peso, passa ao passo `mlp` o sample_weight balanceado calculado sobre y."""
    if peso:
        return estimador.fit(X, y, mlp__sample_weight=compute_sample_weight("balanced", y))
    return estimador.fit(X, y)


def criar_busca(espec, dobras):
    opcoes = {"scoring": SCORING[espec["tarefa"]], "cv": dobras, "n_jobs": N_JOBS, "error_score": "raise"}
    if espec["aleatoria"]:
        return RandomizedSearchCV(espec["estimador"], espec["grade"], n_iter=N_ITER_MLP, random_state=SEED, **opcoes)
    return GridSearchCV(espec["estimador"], espec["grade"], **opcoes)


def buscar(especs, X, g, y, dobras):
    """Roda as buscas no desenvolvimento (com refit nele) e mede o tempo de cada uma."""
    buscas, tempos = {}, {}
    for nome, espec in especs.items():
        print(f"Busca {nome}...", flush=True)
        busca = criar_busca(espec, dobras)
        inicio = time.perf_counter()
        ajustar(busca, X, alvo(espec["tarefa"], g, y), espec["peso"])
        tempos[nome] = time.perf_counter() - inicio
        buscas[nome] = busca
        print(f"  {tempos[nome]:.0f} s", flush=True)
    return buscas, tempos


def escolher_modelos(especs, buscas):
    """Modelo final de cada tarefa; no MLP de classificação, a variante de peso com maior F1-macro na busca."""
    origem = {nome: nome for nome in MODELOS_FINAIS[:3]}
    origem["MLP_clf"] = max(VARIANTES_PESO_MLP, key=lambda n: buscas[n].best_score_)
    modelos = {
        nome: {"estimador": buscas[o].best_estimator_, "tarefa": especs[o]["tarefa"], "peso": especs[o]["peso"]}
        for nome, o in origem.items()
    }
    return modelos, origem


def prever(ajustados, d, limiares, g_treino):
    """Id, g e classe reais, ĝ dos dois regressores, classes das 4 variantes e a faixa de g do treino que gerou a predição."""
    X = d[ATRIBUTOS]
    tabela = pd.DataFrame({"id": d["id"].to_numpy(), "g": d[ALVO_REG].to_numpy(), "classe": d[ALVO_CLF].to_numpy()})
    for nome in REGRESSORES:
        g_pred = ajustados[f"{nome}_reg"].predict(X)
        tabela[f"g_pred_{nome}"] = g_pred
        tabela[f"{nome} direta"] = ajustados[f"{nome}_clf"].predict(X)
        tabela[f"{nome} 2 estágios"] = classe_por_limiares(g_pred, limiares)
    return tabela.assign(g_min_treino=g_treino.min(), g_max_treino=g_treino.max())


def avaliar_dobra(modelos, dev, treino, valid):
    """Reajusta os 4 modelos na parte de treino da dobra e prediz a validação com limiares estimados só no treino."""
    parte = dev.iloc[treino]
    g, y = parte[ALVO_REG], parte[ALVO_CLF]
    ajustados = {
        nome: ajustar(clone(m["estimador"]), parte[ATRIBUTOS], alvo(m["tarefa"], g, y), m["peso"])
        for nome, m in modelos.items()
    }
    limiares = estimar_limiares(g, y)
    pred = prever(ajustados, dev.iloc[valid], limiares, g).assign(**dict(zip(COLUNAS_LIMIARES, limiares)))
    return limiares, pred


def validacao_repetida(modelos, dev):
    """Predições fora da dobra das 5 × 5 dobras estratificadas e os limiares de cada dobra."""
    cv = RepeatedStratifiedKFold(n_splits=N_DOBRAS, n_repeats=N_REPETICOES, random_state=SEED)
    bases = {nome: {**m, "estimador": clone(m["estimador"])} for nome, m in modelos.items()}
    dobras = cv.split(dev[ATRIBUTOS], dev[ALVO_CLF])
    saidas = Parallel(n_jobs=N_JOBS)(delayed(avaliar_dobra)(bases, dev, t, v) for t, v in dobras)
    predicoes = pd.concat(
        [p.assign(repeticao=i // N_DOBRAS + 1, dobra=i % N_DOBRAS + 1) for i, (_, p) in enumerate(saidas)],
        ignore_index=True,
    )
    return predicoes, [limiares for limiares, _ in saidas]


def metricas_regressao(g, g_hat):
    return {"rmse": root_mean_squared_error(g, g_hat), "mae": mean_absolute_error(g, g_hat), "r2": r2_score(g, g_hat)}


def metricas_classificacao(y, y_hat):
    """Acurácia, precision/recall/F1 por classe e macro, e subtriagem/sobretriagem (contagem e taxa)."""
    p, r, f, _ = precision_recall_fscore_support(y, y_hat, labels=CLASSES, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(y, y_hat, labels=CLASSES, average="macro", zero_division=0)
    sub, sobre = y_hat > y, y_hat < y
    return {
        "acuracia": accuracy_score(y, y_hat),
        **{f"precision_{k}": v for k, v in zip(CLASSES, p)},
        **{f"recall_{k}": v for k, v in zip(CLASSES, r)},
        **{f"f1_{k}": v for k, v in zip(CLASSES, f)},
        "precision_macro": pm,
        "recall_macro": rm,
        "f1_macro": fm,
        "subtriagem_n": int(sub.sum()),
        "subtriagem_taxa": sub.mean(),
        "sobretriagem_n": int(sobre.sum()),
        "sobretriagem_taxa": sobre.mean(),
    }


def metricas(pred):
    """Métricas de um conjunto de predições em formato longo: tarefa, variante, metrica, valor."""
    linhas = [
        ("regressao", nome, m, v)
        for nome in REGRESSORES
        for m, v in metricas_regressao(pred["g"], pred[f"g_pred_{nome}"]).items()
    ]
    linhas += [
        ("classificacao", var, m, v)
        for var in VARIANTES
        for m, v in metricas_classificacao(pred["classe"].to_numpy(), pred[var].to_numpy()).items()
    ]
    return pd.DataFrame(linhas, columns=["tarefa", "variante", "metrica", "valor"])


def metricas_por_dobra(predicoes):
    partes = [metricas(p).assign(repeticao=r, dobra=d) for (r, d), p in predicoes.groupby(["repeticao", "dobra"])]
    return pd.concat(partes, ignore_index=True)[["repeticao", "dobra", "tarefa", "variante", "metrica", "valor"]]


def resumir(dobras):
    """Média e desvio-padrão (ddof=0) de cada métrica nas 25 dobras."""
    agrupado = dobras.groupby(["tarefa", "variante", "metrica"], sort=False)["valor"]
    return agrupado.agg(media="mean", desvio=lambda v: v.std(ddof=0)).reset_index()


def matrizes_confusao(pred):
    """Matriz de confusão de cada variante (linhas = classe real, colunas = classe prevista)."""
    return {
        "linhas": "classe real",
        "colunas": "classe prevista",
        "rotulos": CLASSES,
        **{var: confusion_matrix(pred["classe"], pred[var], labels=CLASSES).tolist() for var in VARIANTES},
    }


def rmse_regressor(grupo, nome):
    return root_mean_squared_error(grupo["g"], grupo[f"g_pred_{nome}"])


def desvio_entre_repeticoes(grupo, nome):
    """Desvio-padrão do RMSE entre as repetições da CV; NaN quando não há repetições (teste)."""
    if "repeticao" not in grupo:
        return np.nan
    return float(np.std([rmse_regressor(r, nome) for _, r in grupo.groupby("repeticao")]))


def rmse_por_classe(pred, conjunto):
    """RMSE de cada regressor por classe real, sobre todas as predições do conjunto."""
    grupos = [("todas", pred), *((k, pred[pred["classe"] == k]) for k in CLASSES)]
    return pd.DataFrame(
        [
            {
                "conjunto": conjunto,
                "regressor": nome,
                "classe": classe,
                "n_predicoes": len(grupo),
                "rmse": rmse_regressor(grupo, nome),
                "desvio_entre_repeticoes": desvio_entre_repeticoes(grupo, nome),
            }
            for nome in REGRESSORES
            for classe, grupo in grupos
        ]
    )


def erros_por_distancia(pred):
    """Taxa de erro de cada variante por faixa de distância entre o g real e o corte (25/50/75) mais próximo."""
    distancia = np.abs(pred["g"].to_numpy()[:, None] - np.array(LIMIARES_HIPOTESE)).min(axis=1)
    faixa = np.asarray(pd.cut(distancia, FAIXAS_DISTANCIA, right=False, labels=ROTULOS_DISTANCIA))
    linhas = []
    for var in VARIANTES:
        erro = pred[var].to_numpy() != pred["classe"].to_numpy()
        for rotulo in ROTULOS_DISTANCIA:
            na_faixa = faixa == rotulo
            linhas.append(
                {
                    "variante": var,
                    "faixa_distancia": rotulo,
                    "n_predicoes": int(na_faixa.sum()),
                    "erros": int(erro[na_faixa].sum()),
                    "taxa_erro": erro[na_faixa].mean(),
                }
            )
    return pd.DataFrame(linhas)


def extrapolacao(pred, conjunto):
    """Quantas predições de cada regressor ficam fora da faixa de g do treino que as gerou (além da resolução de 1e-6 dos dados), e o ĝ mínimo e máximo."""
    return pd.DataFrame(
        [
            {
                "conjunto": conjunto,
                "regressor": nome,
                "n_predicoes": len(pred),
                "abaixo_do_min_treino": int((pred[f"g_pred_{nome}"] < pred["g_min_treino"] - TOL_EXTRAPOLACAO).sum()),
                "acima_do_max_treino": int((pred[f"g_pred_{nome}"] > pred["g_max_treino"] + TOL_EXTRAPOLACAO).sum()),
                "g_pred_min": pred[f"g_pred_{nome}"].min(),
                "g_pred_max": pred[f"g_pred_{nome}"].max(),
            }
            for nome in REGRESSORES
        ]
    )


def classes_deslocadas(dobra, nome, delta):
    """Classe em 2 estágios da dobra com os limiares dela somados a δ."""
    limiares = dobra[COLUNAS_LIMIARES].iloc[0].to_numpy() + delta
    return classe_por_limiares(dobra[f"g_pred_{nome}"].to_numpy(), limiares)


def conferir_delta_zero(predicoes):
    """Levanta RuntimeError se δ = 0 não reproduzir as classes em 2 estágios gravadas na CV."""
    for (r, d), dobra in predicoes.groupby(["repeticao", "dobra"]):
        for var, nome in VARIANTES_2_ESTAGIOS.items():
            if not np.array_equal(classes_deslocadas(dobra, nome, 0.0), dobra[var].to_numpy()):
                raise RuntimeError(f"δ = 0 não reproduz {var} na repetição {r}, dobra {d}")


def limiar_conservador(predicoes):
    """Triagem, acurácia e F1-macro das variantes em 2 estágios com os limiares de cada dobra somados a δ."""
    conferir_delta_zero(predicoes)
    linhas = [
        {"variante": var, "delta": delta, **metricas_classificacao(dobra["classe"].to_numpy(), classes_deslocadas(dobra, nome, delta))}
        for _, dobra in predicoes.groupby(["repeticao", "dobra"])
        for var, nome in VARIANTES_2_ESTAGIOS.items()
        for delta in DELTAS
    ]
    agrupado = pd.DataFrame(linhas).groupby(["variante", "delta"], sort=False)
    resumo = agrupado[["subtriagem_n", "sobretriagem_n"]].sum()
    for m in METRICAS_CONSERVADOR:
        resumo[f"{m}_media"] = agrupado[m].mean()
        resumo[f"{m}_desvio"] = agrupado[m].std(ddof=0)
    return resumo.reset_index()


def limiares_de(d):
    """Pontos médios e limiares da árvore de profundidade 2 num conjunto rotulado."""
    medio = estimar_limiares(d[ALVO_REG], d[ALVO_CLF])
    arvore = limiares_arvore(d[ALVO_REG], d[ALVO_CLF])
    confere = len(arvore) == len(medio) and np.allclose(arvore, medio, atol=1e-4)
    return {"ponto_medio": medio, "arvore_prof2": arvore, "arvore_confere": bool(confere)}


def resumo_limiares(dev, df, limiares_dobras):
    dobras = np.array(limiares_dobras)
    return {
        "hipotese": LIMIARES_HIPOTESE,
        "desenvolvimento": limiares_de(dev),
        "todos_rotulados": limiares_de(df),
        "dobras_cv": {
            "media": dobras.mean(axis=0).tolist(),
            "desvio": dobras.std(axis=0).tolist(),
            "min": dobras.min(axis=0).tolist(),
            "max": dobras.max(axis=0).tolist(),
        },
    }


def importancia(ajustados, X, g):
    """Importância por permutação no teste: aumento do RMSE ao embaralhar cada atributo."""
    linhas = []
    for nome in REGRESSORES:
        r = permutation_importance(
            ajustados[f"{nome}_reg"],
            X,
            g,
            scoring="neg_root_mean_squared_error",
            n_repeats=N_REPETICOES_PERMUTACAO,
            random_state=SEED,
            n_jobs=N_JOBS,
        )
        linhas += [
            {"regressor": nome, "atributo": a, "aumento_rmse_media": m, "aumento_rmse_desvio": s}
            for a, m, s in zip(ATRIBUTOS, r.importances_mean, r.importances_std)
        ]
    return pd.DataFrame(linhas)


def treinar_finais(modelos, df):
    """Re-treina cada modelo em 100% dos dados rotulados e mede o tempo de treino."""
    X, g, y = df[ATRIBUTOS], df[ALVO_REG], df[ALVO_CLF]
    finais, tempos = {}, {}
    for nome, m in modelos.items():
        inicio = time.perf_counter()
        finais[nome] = ajustar(clone(m["estimador"]), X, alvo(m["tarefa"], g, y), m["peso"])
        tempos[nome] = time.perf_counter() - inicio
    return finais, tempos


def salvar_modelos(finais, limiares):
    """Grava os .joblib e modelos/limiares.json; devolve o tamanho de cada arquivo em KB."""
    tamanhos = {}
    for nome, modelo in finais.items():
        caminho = MODELOS / f"{nome.lower()}.joblib"
        joblib.dump(modelo, caminho)
        tamanhos[nome] = caminho.stat().st_size / 1024
    escrever_json(
        MODELOS / "limiares.json",
        {"limiares": limiares, "origem": "ponto médio entre classes vizinhas nos 1500 rotulados"},
    )
    return tamanhos


def latencia_ms(modelo, X):
    """Mediana do tempo de predição de X, em ms por amostra."""
    tempos = []
    for _ in range(N_REPETICOES_LATENCIA):
        inicio = time.perf_counter()
        modelo.predict(X)
        tempos.append(time.perf_counter() - inicio)
    return 1000 * float(np.median(tempos)) / len(X)


def mlp_interno(modelo):
    return getattr(modelo, "regressor_", modelo).named_steps["mlp"]


def resumo_busca(busca, espec):
    metrica, sinal = METRICA_SELECAO[espec["tarefa"]]
    return {
        "busca": type(busca).__name__,
        "candidatos": len(busca.cv_results_["params"]),
        "metrica_selecao": metrica,
        "score_cv": float(sinal * busca.best_score_),
        "score_cv_desvio": float(busca.cv_results_["std_test_score"][busca.best_index_]),
        "params": {k.split("__")[-1]: v for k, v in sorted(busca.best_params_.items())},
    }


def melhores_parametros(especs, buscas, origem, finais):
    resumo = {nome: resumo_busca(buscas[o], especs[o]) for nome, o in origem.items()}
    resumo["MLP_clf"]["peso"] = "balanced" if especs[origem["MLP_clf"]]["peso"] else None
    resumo["MLP_clf"]["f1_macro_cv_por_peso"] = {n: float(buscas[n].best_score_) for n in VARIANTES_PESO_MLP}
    for nome in ("MLP_reg", "MLP_clf"):
        resumo[nome]["iteracoes_modelo_final"] = int(mlp_interno(finais[nome]).n_iter_)
        resumo[nome]["max_iter"] = MLP_FIXOS["max_iter"]
    return resumo


def tabela_custo(buscas, tempos_busca, tempos_treino, latencias, tamanhos):
    """Custo de cada modelo final; o MLP de classificação soma as buscas das duas variantes de peso."""
    linhas = []
    for nome in MODELOS_FINAIS:
        fontes = [n for n in buscas if n.startswith(nome)]
        candidatos = sum(len(buscas[n].cv_results_["params"]) for n in fontes)
        linhas.append(
            {
                "modelo": nome,
                "busca": type(buscas[fontes[0]]).__name__,
                "candidatos": candidatos,
                "ajustes_busca": candidatos * N_DOBRAS,
                "tempo_busca_s": sum(tempos_busca[n] for n in fontes),
                "tempo_treino_final_s": tempos_treino[nome],
                "latencia_lote_ms_por_amostra": latencias[nome][0],
                "latencia_unitaria_ms": latencias[nome][1],
                "tamanho_joblib_kb": tamanhos[nome],
            }
        )
    return pd.DataFrame(linhas)


def escrever_json(caminho, dados):
    caminho.write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def figura_pred_vs_real(pred, limiares, caminho):
    """ĝ × g no teste, RF e MLP lado a lado, com diagonal, limiares e erros da classificação em 2 estágios."""
    valores = pred[["g", *(f"g_pred_{n}" for n in REGRESSORES)]].to_numpy()
    limites = (min(10.0, np.floor(valores.min())), max(90.0, np.ceil(valores.max())))
    estilo = ESTILO_FIGURA | {"axes.grid": False, "axes.spines.left": True, "ytick.left": True}
    with plt.rc_context(estilo):
        fig, eixos = plt.subplots(1, 2, figsize=(3.4, 2.3), sharey=True, layout="constrained")
        for ax, nome in zip(eixos, REGRESSORES):
            errou = (pred[f"{nome} 2 estágios"] != pred["classe"]).to_numpy()
            ax.plot(limites, limites, color=TINTA_SECUNDARIA, linewidth=0.6, zorder=1)
            for corte in limiares:
                ax.axvline(corte, color=COR_EIXO, linestyle=(0, (3, 2)), linewidth=0.6, zorder=1)
                ax.axhline(corte, color=COR_EIXO, linestyle=(0, (3, 2)), linewidth=0.6, zorder=1)
            ax.scatter(pred["g"][~errou], pred[f"g_pred_{nome}"][~errou], s=4, color=CORES_SERIES[0], linewidths=0, zorder=2, label="classe correta")
            ax.scatter(
                pred["g"][errou], pred[f"g_pred_{nome}"][errou], s=10, color=CORES_SERIES[1],
                edgecolors=SUPERFICIE, linewidths=0.3, zorder=3, label="classe errada (2 estágios)",
            )
            ax.set(xlim=limites, ylim=limites, xticks=LIMIARES_HIPOTESE, yticks=LIMIARES_HIPOTESE)
            ax.set_box_aspect(1)
            ax.set_title(f"{nome} · RMSE {fmt(rmse_regressor(pred, nome), 2)}")
            ax.set_xlabel("g real")
        eixos[0].set_ylabel("ĝ previsto")
        fig.legend(*eixos[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
        fig.savefig(caminho, dpi=300, facecolor=SUPERFICIE)
        plt.close(fig)


def figura_matrizes(pred, caminho):
    """Matrizes de confusão das 4 variantes no teste; cor = fração da linha, número = contagem."""
    estilo = ESTILO_FIGURA | {"axes.grid": False, "axes.spines.bottom": False, "xtick.bottom": False}
    with plt.rc_context(estilo):
        fig, eixos = plt.subplots(2, 2, figsize=(3.4, 3.5), layout="constrained")
        for ax, var in zip(eixos.flat, VARIANTES):
            matriz = confusion_matrix(pred["classe"], pred[var], labels=CLASSES)
            fracao = matriz / matriz.sum(axis=1, keepdims=True)
            ax.imshow(fracao, cmap=MAPA_AZUL, vmin=0, vmax=1)
            for (i, j), n in np.ndenumerate(matriz):
                cor = SUPERFICIE if fracao[i, j] > FRACAO_TEXTO_CLARO else TINTA
                ax.text(j, i, str(n), ha="center", va="center", fontsize=6.5, color=cor)
            ax.set(xticks=range(len(CLASSES)), yticks=range(len(CLASSES)), xticklabels=CLASSES, yticklabels=CLASSES)
            ax.set_title(f"{var}\nacurácia {fmt(accuracy_score(pred['classe'], pred[var]), 3)}")
        for ax in eixos[1]:
            ax.set_xlabel("Classe prevista")
        for ax in eixos[:, 0]:
            ax.set_ylabel("Classe real")
        fig.savefig(caminho, dpi=300, facecolor=SUPERFICIE)
        plt.close(fig)


def figura_importancia(imp, caminho):
    """Barras horizontais do aumento do RMSE por atributo permutado, RF e MLP lado a lado, com desvio."""
    ordem = imp.groupby("atributo")["aumento_rmse_media"].mean().sort_values().index.tolist()
    altura = 0.38
    estilo = ESTILO_FIGURA | {"axes.grid.axis": "x", "axes.spines.left": True}
    with plt.rc_context(estilo):
        fig, ax = plt.subplots(figsize=(3.4, 1.9), layout="constrained")
        for deslocamento, nome, cor in zip((altura / 2, -altura / 2), REGRESSORES, CORES_SERIES):
            dados = imp[imp["regressor"] == nome].set_index("atributo").loc[ordem]
            posicoes = np.arange(len(ordem)) + deslocamento
            media, desvio = dados["aumento_rmse_media"].to_numpy(), dados["aumento_rmse_desvio"].to_numpy()
            ax.barh(
                posicoes, media, height=altura, xerr=desvio, color=cor, edgecolor=SUPERFICIE, linewidth=0.5,
                error_kw={"elinewidth": 0.6, "ecolor": TINTA_SECUNDARIA, "capsize": 0}, label=nome,
            )
            for p, m, s in zip(posicoes, media, desvio):
                ax.text(m + s, p, f" {fmt(m, 2)}", va="center", ha="left", fontsize=6)
        ax.set_yticks(range(len(ordem)), ordem)
        extremos = imp["aumento_rmse_media"] - imp["aumento_rmse_desvio"], imp["aumento_rmse_media"] + imp["aumento_rmse_desvio"]
        ax.set_xlim(min(0.0, extremos[0].min() * 1.2), extremos[1].max() * 1.2)
        ax.set_xlabel("Aumento do RMSE ao permutar o atributo")
        ax.legend(frameon=False, loc="lower right")
        fig.savefig(caminho, dpi=300, facecolor=SUPERFICIE)
        plt.close(fig)


def cel_cv(cv, variante, metrica, casas=4):
    linha = cv.loc[(variante, metrica)]
    return f"{fmt(linha['media'], casas)} ± {fmt(linha['desvio'], casas)}"


def cel_teste(teste, variante, metrica, casas=4):
    return fmt(teste.loc[(variante, metrica)], casas)


def secao_regressao(cv, teste):
    metricas_reg = ["rmse", "mae", "r2"]
    linhas = [
        [nome, *(cel_cv(cv, nome, m) for m in metricas_reg), *(cel_teste(teste, nome, m) for m in metricas_reg)]
        for nome in REGRESSORES
    ]
    return [
        "## 1. Regressão",
        "",
        *tabela_md(["Modelo", "RMSE CV", "MAE CV", "R² CV", "RMSE teste", "MAE teste", "R² teste"], linhas),
    ]


def secao_classificacao(cv, teste):
    metricas_clf = ["acuracia", "f1_macro", "precision_macro", "recall_macro"]
    nomes = ["Acurácia", "F1-macro", "Precision-macro", "Recall-macro"]
    linhas = [
        [var, *(cel_cv(cv, var, m) for m in metricas_clf), *(cel_teste(teste, var, m) for m in metricas_clf)]
        for var in VARIANTES
    ]
    cabecalho = ["Variante", *(f"{n} CV" for n in nomes), *(f"{n} teste" for n in nomes)]
    return ["## 2. Classificação — 4 variantes", "", *tabela_md(cabecalho, linhas)]


def secao_por_classe(cv, teste):
    linhas = [
        [
            var,
            k,
            *(cel_cv(cv, var, f"{m}_{k}") for m in ("precision", "recall", "f1")),
            *(cel_teste(teste, var, f"{m}_{k}") for m in ("precision", "recall", "f1")),
        ]
        for var in VARIANTES
        for k in CLASSES
    ]
    cabecalho = ["Variante", "Classe", "Precision CV", "Recall CV", "F1 CV", "Precision teste", "Recall teste", "F1 teste"]
    return ["## 3. Classificação por classe", "", *tabela_md(cabecalho, linhas)]


def secao_triagem(cv, cv_agregado, teste, n_cv, n_teste):
    def contagem(fonte, var, tipo, total):
        return f"{int(fonte.loc[(var, f'{tipo}_n')])} de {total}"

    linhas = [
        [
            var,
            cel_cv(cv, var, "recall_1"),
            f"{cel_cv(cv, var, 'subtriagem_taxa')} ({contagem(cv_agregado, var, 'subtriagem', n_cv)})",
            f"{cel_cv(cv, var, 'sobretriagem_taxa')} ({contagem(cv_agregado, var, 'sobretriagem', n_cv)})",
            cel_teste(teste, var, "recall_1"),
            f"{cel_teste(teste, var, 'subtriagem_taxa')} ({contagem(teste, var, 'subtriagem', n_teste)})",
            f"{cel_teste(teste, var, 'sobretriagem_taxa')} ({contagem(teste, var, 'sobretriagem', n_teste)})",
        ]
        for var in VARIANTES
    ]
    cabecalho = [
        "Variante", "Recall classe 1 CV", "Subtriagem CV", "Sobretriagem CV",
        "Recall classe 1 teste", "Subtriagem teste", "Sobretriagem teste",
    ]
    return [
        "## 4. Triagem",
        "",
        "Subtriagem = classe prevista > real (vítima tratada como mais estável do que está); sobretriagem = prevista < real. "
        f"Taxa = fração das vítimas; na CV, média ± dp nas {N_REPETICOES * N_DOBRAS} dobras e, entre parênteses, "
        f"a soma das contagens nas {N_REPETICOES * N_DOBRAS} dobras.",
        "",
        *tabela_md(cabecalho, linhas),
    ]


def secao_matrizes(matrizes):
    linhas = ["## 5. Matrizes de confusão no teste", "", "Linhas = classe real; colunas = classe prevista."]
    for var in VARIANTES:
        linhas += [
            "",
            f"**{var}**",
            "",
            *tabela_md(["Real × prevista", *map(str, CLASSES)], [[k, *linha] for k, linha in zip(CLASSES, matrizes[var])]),
        ]
    return linhas


def secao_limiares(limiares):
    dev, todos, dobras = limiares["desenvolvimento"], limiares["todos_rotulados"], limiares["dobras_cv"]
    linhas = [
        [
            f"{k}→{k + 1}",
            fmt(limiares["hipotese"][i], 0),
            fmt(dev["ponto_medio"][i]),
            fmt(dev["arvore_prof2"][i]),
            fmt(todos["ponto_medio"][i]),
            f"{fmt(dobras['media'][i])} ± {fmt(dobras['desvio'][i])}",
            f"{fmt(dobras['min'][i])}–{fmt(dobras['max'][i])}",
        ]
        for i, k in enumerate(CLASSES[:-1])
    ]
    cabecalho = [
        "Corte", "Hipótese", "Desenvolvimento (ponto médio)", "Desenvolvimento (árvore prof. 2)",
        "100% rotulados", "Dobras CV: média ± dp", "Dobras CV: mín–máx",
    ]
    return [
        "## 6. Limiares do classificador em 2 estágios",
        "",
        "Desenvolvimento → 2 estágios no teste; 100% rotulados → modelos finais (teste cego); dobras → 2 estágios na CV.",
        "",
        *tabela_md(cabecalho, linhas),
    ]


def secao_hiperparametros(melhores):
    linhas = [
        [
            nome,
            f"{r['busca']} ({r['candidatos']} candidatos)",
            "`" + ", ".join(f"{k}={v!r}" for k, v in r["params"].items()) + "`",
            f"{r['metrica_selecao']} {fmt(r['score_cv'])} ± {fmt(r['score_cv_desvio'])}",
        ]
        for nome, r in melhores.items()
    ]
    pesos = melhores["MLP_clf"]["f1_macro_cv_por_peso"]
    return [
        "## 7. Hiperparâmetros escolhidos",
        "",
        *tabela_md(["Modelo", "Busca", "Hiperparâmetros", "Score de seleção (CV da busca)"], linhas),
        "",
        "MLP de classificação, melhor F1-macro na busca por variante de peso: "
        + "; ".join(f"{n.removeprefix('MLP_clf ')} = {fmt(v)}" for n, v in pesos.items())
        + f". Escolhido: peso {melhores['MLP_clf']['peso'] or 'nenhum'}.",
    ]


def secao_custo(custo, tempo_total):
    linhas = [
        [
            r["modelo"],
            r["busca"],
            r["candidatos"],
            r["ajustes_busca"],
            fmt(r["tempo_busca_s"], 1),
            fmt(r["tempo_treino_final_s"], 3),
            fmt(r["latencia_lote_ms_por_amostra"], 4),
            fmt(r["latencia_unitaria_ms"], 3),
            fmt(r["tamanho_joblib_kb"], 1),
        ]
        for r in custo.to_dict("records")
    ]
    cabecalho = [
        "Modelo", "Busca", "Candidatos", "Ajustes na busca", "Busca (s)", "Treino final (s)",
        "Latência em lote (ms/vítima)", "Latência unitária (ms)", ".joblib (KB)",
    ]
    return [
        "## 8. Custo",
        "",
        f"Busca com {N_JOBS} processos; treino final e latência com 1 processo. MLP_clf soma as buscas sem peso e balanceada. "
        "O custo do 2 estágios é o do regressor.",
        "",
        *tabela_md(cabecalho, linhas),
        "",
        f"Tempo total da execução: {fmt(tempo_total, 0)} s ({fmt(tempo_total / 60, 1)} min).",
    ]


def secao_rmse_classe(rmse_classe):
    cv = rmse_classe[rmse_classe["conjunto"] == "cv"].set_index(["regressor", "classe"])
    teste = rmse_classe[rmse_classe["conjunto"] == "teste"].set_index(["regressor", "classe"])
    linhas = [
        [
            nome,
            classe,
            int(cv.loc[(nome, classe), "n_predicoes"]),
            f"{fmt(cv.loc[(nome, classe), 'rmse'])} ({fmt(cv.loc[(nome, classe), 'desvio_entre_repeticoes'])})",
            int(teste.loc[(nome, classe), "n_predicoes"]),
            fmt(teste.loc[(nome, classe), "rmse"]),
        ]
        for nome in REGRESSORES
        for classe in ["todas", *CLASSES]
    ]
    return [
        "## 9. RMSE por classe real",
        "",
        f"CV: RMSE sobre as predições fora da dobra das {N_REPETICOES} repetições juntas "
        "(entre parênteses, o desvio do RMSE entre as repetições).",
        "",
        *tabela_md(["Regressor", "Classe", "n CV (predições)", "RMSE CV", "n teste", "RMSE teste"], linhas),
    ]


def secao_distancia(erros):
    tabela = erros.set_index(["faixa_distancia", "variante"])
    linhas = [
        [
            rotulo,
            int(tabela.loc[(rotulo, VARIANTES[0]), "n_predicoes"]),
            *(f"{fmt(tabela.loc[(rotulo, var), 'taxa_erro'])} ({int(tabela.loc[(rotulo, var), 'erros'])})" for var in VARIANTES),
        ]
        for rotulo in ROTULOS_DISTANCIA
    ]
    return [
        "## 10. Taxa de erro × distância ao corte mais próximo",
        "",
        f"Distância entre o g real e o corte (25, 50 ou 75) mais próximo; predições fora da dobra das {N_REPETICOES} repetições agregadas. "
        "Célula = taxa de erro (número de erros).",
        "",
        *tabela_md(["Distância ao corte", "n (predições)", *VARIANTES], linhas),
    ]


def secao_importancia(imp):
    tabela = imp.set_index(["atributo", "regressor"])
    linhas = [
        [
            f"`{a}`",
            *(
                f"{fmt(tabela.loc[(a, nome), 'aumento_rmse_media'])} ± {fmt(tabela.loc[(a, nome), 'aumento_rmse_desvio'])}"
                for nome in REGRESSORES
            ),
        ]
        for a in ATRIBUTOS
    ]
    return [
        "## 11. Importância por permutação no teste",
        "",
        f"Aumento do RMSE ao embaralhar o atributo ({N_REPETICOES_PERMUTACAO} repetições; média ± dp).",
        "",
        *tabela_md(["Atributo", *REGRESSORES], linhas),
    ]


def secao_limiar_conservador(conservador, n_cv):
    def media_dp(r, m):
        return f"{fmt(r[f'{m}_media'])} ± {fmt(r[f'{m}_desvio'])}"

    linhas = [
        [
            r["variante"],
            fmt(r["delta"], 1),
            f"{media_dp(r, 'subtriagem_taxa')} ({int(r['subtriagem_n'])} de {n_cv})",
            f"{media_dp(r, 'sobretriagem_taxa')} ({int(r['sobretriagem_n'])} de {n_cv})",
            media_dp(r, "recall_1"),
            media_dp(r, "acuracia"),
            media_dp(r, "f1_macro"),
        ]
        for r in conservador.to_dict("records")
    ]
    return [
        "## 12. Limiar conservador (variantes em 2 estágios)",
        "",
        "Limiares de cada dobra somados a δ: a vítima vai para a classe mais grave quando ĝ < limiar + δ. "
        f"Predições fora da dobra das {N_REPETICOES} repetições; média ± dp nas {N_REPETICOES * N_DOBRAS} dobras e, "
        "entre parênteses, a soma das contagens. δ = 0 reproduz as seções 2 a 4; as métricas principais usam δ = 0.",
        "",
        *tabela_md(["Variante", "δ", "Subtriagem", "Sobretriagem", "Recall classe 1", "Acurácia", "F1-macro"], linhas),
    ]


def escrever_tabelas(caminho, secoes, n_dev, n_teste):
    linhas = [
        "# Tabelas para o artigo",
        "",
        "Gerado por `python -m src.experimentos`; não editar à mão. "
        f"CV = validação cruzada repetida ({N_REPETICOES} repetições × {N_DOBRAS} dobras estratificadas) no desenvolvimento "
        f"({n_dev} vítimas): média ± desvio-padrão (ddof=0) nas {N_REPETICOES * N_DOBRAS} dobras. "
        f"Teste = hold-out estratificado ({n_teste} vítimas), usado uma única vez, "
        "com os modelos treinados no desenvolvimento.",
    ]
    for secao in secoes:
        linhas += ["", *secao]
    caminho.write_text("\n".join(linhas) + "\n", encoding="utf-8")


def main():
    inicio = time.perf_counter()
    for pasta in (RESULTADOS, FIGURAS, MODELOS):
        pasta.mkdir(parents=True, exist_ok=True)

    df = ler_rotulados()
    dev, teste = dividir_hold_out(df)
    pd.DataFrame({"id": np.sort(teste["id"].to_numpy())}).to_csv(RESULTADOS / "ids_teste.csv", index=False)
    X, g, y = dev[ATRIBUTOS], dev[ALVO_REG], dev[ALVO_CLF]
    dobras = list(StratifiedKFold(n_splits=N_DOBRAS, shuffle=True, random_state=SEED).split(X, y))

    especs = especificacoes()
    buscas, tempos_busca = buscar(especs, X, g, y, dobras)
    modelos, origem = escolher_modelos(especs, buscas)

    print("Validação cruzada repetida...", flush=True)
    predicoes_cv, limiares_dobras = validacao_repetida(modelos, dev)
    metricas_dobras = metricas_por_dobra(predicoes_cv)
    resumo_cv = resumir(metricas_dobras)
    conservador = limiar_conservador(predicoes_cv)

    print("Teste no hold-out...", flush=True)
    limiares_dev = estimar_limiares(g, y)
    ajustados_dev = {nome: m["estimador"] for nome, m in modelos.items()}
    pred_teste = prever(ajustados_dev, teste, limiares_dev, g)
    metricas_teste = metricas(pred_teste)
    imp = importancia(ajustados_dev, teste[ATRIBUTOS], teste[ALVO_REG])

    print("Modelos finais...", flush=True)
    limiares = resumo_limiares(dev, df, limiares_dobras)
    finais, tempos_treino = treinar_finais(modelos, df)
    tamanhos = salvar_modelos(finais, limiares["todos_rotulados"]["ponto_medio"])
    latencias = {nome: (latencia_ms(m, teste[ATRIBUTOS]), latencia_ms(m, teste[ATRIBUTOS].iloc[:1])) for nome, m in finais.items()}

    melhores = melhores_parametros(especs, buscas, origem, finais)
    matrizes_teste, matrizes_cv = matrizes_confusao(pred_teste), matrizes_confusao(predicoes_cv)
    rmse_classe = pd.concat([rmse_por_classe(predicoes_cv, "cv"), rmse_por_classe(pred_teste, "teste")], ignore_index=True)
    erros = erros_por_distancia(predicoes_cv)
    extrapolacoes = pd.concat([extrapolacao(predicoes_cv, "cv"), extrapolacao(pred_teste, "teste")], ignore_index=True)
    custo = tabela_custo(buscas, tempos_busca, tempos_treino, latencias, tamanhos)

    escrever_json(RESULTADOS / "melhores_params.json", melhores)
    escrever_json(RESULTADOS / "limiares.json", limiares)
    escrever_json(RESULTADOS / "matrizes_confusao_teste.json", matrizes_teste)
    escrever_json(RESULTADOS / "matrizes_confusao_cv.json", matrizes_cv)
    metricas_dobras.to_csv(RESULTADOS / "metricas_cv_dobras.csv", index=False)
    resumo_cv.to_csv(RESULTADOS / "metricas_cv_resumo.csv", index=False)
    metricas_teste.to_csv(RESULTADOS / "metricas_teste.csv", index=False)
    rmse_classe.to_csv(RESULTADOS / "rmse_por_classe.csv", index=False)
    erros.to_csv(RESULTADOS / "erros_vs_distancia_corte.csv", index=False)
    imp.to_csv(RESULTADOS / "importancia_permutacao.csv", index=False)
    custo.to_csv(RESULTADOS / "custo.csv", index=False)
    predicoes_cv[["repeticao", "dobra", *COLUNAS_PREDICAO, *COLUNAS_LIMIARES]].to_csv(RESULTADOS / "predicoes_cv.csv", index=False)
    pred_teste[COLUNAS_PREDICAO].to_csv(RESULTADOS / "predicoes_teste.csv", index=False)
    extrapolacoes.to_csv(RESULTADOS / "extrapolacao.csv", index=False)
    conservador.to_csv(RESULTADOS / "limiar_conservador.csv", index=False)

    figura_pred_vs_real(pred_teste, limiares_dev, FIGURAS / "fig2_pred_vs_real.png")
    figura_matrizes(pred_teste, FIGURAS / "fig3_matrizes_confusao.png")
    figura_importancia(imp, FIGURAS / "fig4_importancia_permutacao.png")

    cv = resumo_cv.set_index(["variante", "metrica"])
    cv_agregado = metricas(predicoes_cv).set_index(["variante", "metrica"])["valor"]
    te = metricas_teste.set_index(["variante", "metrica"])["valor"]
    tempo_total = time.perf_counter() - inicio
    secoes = [
        secao_regressao(cv, te),
        secao_classificacao(cv, te),
        secao_por_classe(cv, te),
        secao_triagem(cv, cv_agregado, te, len(predicoes_cv), len(pred_teste)),
        secao_matrizes(matrizes_teste),
        secao_limiares(limiares),
        secao_hiperparametros(melhores),
        secao_custo(custo, tempo_total),
        secao_rmse_classe(rmse_classe),
        secao_distancia(erros),
        secao_importancia(imp),
        secao_limiar_conservador(conservador, len(predicoes_cv)),
    ]
    escrever_tabelas(RESULTADOS / "tabelas.md", secoes, len(dev), len(teste))
    print(f"Tempo total: {tempo_total:.0f} s", flush=True)


if __name__ == "__main__":
    main()

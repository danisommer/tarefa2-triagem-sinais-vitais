"""Fase 3 — teste cego: lê `i, qPA, pulso, resp` e grava `i,gravid,classe` para cada modelo.

A classe sai em 2 estágios: o regressor prevê a gravidade, limitada à faixa de g dos 1500 rotulados e
arredondada a 4 casas, e a classe é a faixa desse valor segundo `modelos/limiares.json`, de modo que
gravid e classe gravados são sempre coerentes.
"""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.comum import (
    ATRIBUTOS,
    CASAS_GRAVID,
    FAIXAS_ENUNCIADO,
    MODELOS,
    RESULTADOS,
    classe_por_limiares,
    faixa_g,
    ler_rotulados,
    limitar_g,
)

N_CAMPOS = 1 + len(ATRIBUTOS)
MODELOS_ESCOLHA = {"rf": ["rf"], "mlp": ["mlp"], "ambos": ["rf", "mlp"]}
SAIDA_PADRAO = RESULTADOS / "teste_cego"


class EntradaInvalida(ValueError):
    """Arquivo de entrada fora do formato `i, qPA, pulso, resp`."""


def como_numero(texto):
    """O número finito escrito no texto, ou None se o texto não for um."""
    try:
        valor = float(texto)
    except ValueError:
        return None
    return valor if math.isfinite(valor) else None


def classificar_linha(texto):
    """('vazia',), ('texto',) se nenhum campo for número, ('dados', id, valores) ou ('invalida',)."""
    campos = [c.strip() for c in texto.split(",")]
    numeros = [como_numero(c) for c in campos]
    if not texto.strip():
        return ("vazia",)
    if all(n is None for n in numeros):
        return ("texto",)
    if len(campos) == N_CAMPOS and all(n is not None for n in numeros):
        return ("dados", campos[0], numeros[1:])
    return ("invalida",)


def ler_entrada(caminho):
    """Ids (texto original), atributos das vítimas na ordem do arquivo e as linhas tratadas como cabeçalho."""
    ids, valores, cabecalho = [], [], []
    for numero, texto in enumerate(caminho.read_text(encoding="utf-8-sig").splitlines(), start=1):
        match classificar_linha(texto):
            case ("vazia",):
                pass
            case ("texto",) if not ids:
                cabecalho.append(texto.strip())
            case ("dados", ident, numeros):
                ids.append(ident)
                valores.append(numeros)
            case _:
                raise EntradaInvalida(
                    f"linha {numero}: esperado `i, qPA, pulso, resp` com {N_CAMPOS} números "
                    f"(cabeçalho só antes da primeira vítima): {texto!r}"
                )
    if not ids:
        detalhe = f" (tratadas como cabeçalho: {cabecalho})" if cabecalho else ""
        raise EntradaInvalida(f"nenhuma vítima no arquivo{detalhe}")
    return ids, pd.DataFrame(valores, columns=ATRIBUTOS), cabecalho


def avisos(ids, X):
    """Valores fora dos intervalos do enunciado e ids repetidos; nenhum deles interrompe a predição."""
    mensagens = []
    for atributo in ATRIBUTOS:
        minimo, maximo = FAIXAS_ENUNCIADO[atributo]
        fora = np.flatnonzero(~X[atributo].between(minimo, maximo))
        mensagens += [f"vítima {ids[i]}: {atributo} = {X.at[i, atributo]:g} fora de [{minimo:g}, {maximo:g}]" for i in fora]
    repetidos = [i for i, n in Counter(ids).items() if n > 1]
    if repetidos:
        mensagens.append(f"ids repetidos: {', '.join(repetidos)}")
    return mensagens


def exigir(caminho):
    if not caminho.is_file():
        raise FileNotFoundError(f"{caminho} não existe; rode `python -m src.experimentos` antes")
    return caminho


def carregar_limiares():
    return json.loads(exigir(MODELOS / "limiares.json").read_text(encoding="utf-8"))["limiares"]


def carregar_regressor(nome):
    return joblib.load(exigir(MODELOS / f"{nome}_reg.joblib"))


def prever(regressor, X, limiares, faixa):
    """Gravidade limitada à faixa e arredondada a 4 casas, a classe da faixa desse valor e quantas previsões a limitação mudou."""
    limitado, alteradas = limitar_g(regressor.predict(X), *faixa)
    gravid = np.round(limitado, CASAS_GRAVID)
    return gravid, classe_por_limiares(gravid, limiares), int(alteradas.sum())


def escrever(caminho, ids, gravid, classe):
    linhas = (f"{i},{g:.{CASAS_GRAVID}f},{c}\n" for i, g, c in zip(ids, gravid, classe))
    caminho.write_text("".join(linhas), encoding="utf-8")


def ler_argumentos(argv):
    parser = argparse.ArgumentParser(prog="python -m src.prever", description="Teste cego: gravidade e classe de cada vítima.")
    parser.add_argument("--entrada", type=Path, required=True, help="arquivo com `i, qPA, pulso, resp`, uma vítima por linha")
    parser.add_argument("--modelo", choices=list(MODELOS_ESCOLHA), default="ambos")
    parser.add_argument("--saida-dir", type=Path, default=SAIDA_PADRAO)
    return parser.parse_args(argv)


def main(argv=None):
    args = ler_argumentos(argv)
    if not args.entrada.is_file():
        sys.exit(f"erro: {args.entrada} não existe")
    try:
        ids, X, cabecalho = ler_entrada(args.entrada)
    except EntradaInvalida as erro:
        sys.exit(f"erro: {args.entrada}: {erro}")
    for linha in cabecalho:
        print(f"cabeçalho ignorado: {linha!r}")
    for aviso in avisos(ids, X):
        print(f"aviso: {aviso}", file=sys.stderr)
    limiares = carregar_limiares()
    faixa = faixa_g(ler_rotulados())
    args.saida_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} vítimas lidas; classe em 2 estágios com limiares {', '.join(f'{l:.4f}' for l in limiares)}")
    print(f"ĝ limitado à faixa de g dos rotulados: [{faixa[0]:.4f}; {faixa[1]:.4f}]")
    for nome in MODELOS_ESCOLHA[args.modelo]:
        gravid, classe, limitadas = prever(carregar_regressor(nome), X, limiares, faixa)
        caminho = args.saida_dir / f"predicoes_{nome}.txt"
        escrever(caminho, ids, gravid, classe)
        print(f"{caminho}: {len(ids)} linhas; {limitadas} previsões limitadas")


if __name__ == "__main__":
    main()

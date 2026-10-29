"""Teste de fumaça da Fase 3: roda o prever.py em duas entradas e confere as saídas.

Casos: as vítimas do hold-out (cabeçalho de uma linha e espaço após a vírgula) e o exemplo do enunciado
(cabeçalho de duas linhas). Confere número de linhas, formato `i,gravid,classe` (gravid com 4 casas), ordem
dos ids, coerência gravid ↔ classe pelos limiares dos modelos finais e gravid dentro da faixa de g dos
rotulados. Termina com código 1 se alguma checagem falhar.
"""

import json
import re
import subprocess
import sys

from src.comum import (
    ATRIBUTOS,
    CASAS_GRAVID,
    MODELOS,
    RAIZ,
    RESULTADOS,
    classe_por_limiares,
    dividir_hold_out,
    faixa_g,
    ler_rotulados,
)

PASTA = RESULTADOS / "teste_cego"
PASTA_ENUNCIADO = PASTA / "enunciado"
ENTRADA_HOLDOUT = PASTA / "exemplo_entrada.txt"
ENTRADA_ENUNCIADO = PASTA_ENUNCIADO / "exemplo_entrada.txt"
LINHAS_ENUNCIADO = ["i si3 si4 si5", "qPA pulso resp", "1,    -8.5577, 56.8004,   9.0000"]
SAIDAS = ["rf", "mlp"]
FORMATO_LINHA = re.compile(r"^(-?\d+(?:\.\d+)?),(-?\d+\.\d{4}),([1-4])$")


def gerar_entrada_holdout(teste):
    """Grava id, qPA, pulso e resp do hold-out com cabeçalho e espaço após a vírgula; devolve os ids."""
    linhas = ["i, qPA, pulso, resp"]
    linhas += [", ".join([str(r.id), *(repr(float(getattr(r, a))) for a in ATRIBUTOS)]) for r in teste.itertuples()]
    ENTRADA_HOLDOUT.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return [str(i) for i in teste["id"]]


def gerar_entrada_enunciado():
    """Grava o exemplo do enunciado, com o cabeçalho de duas linhas e os espaços originais; devolve os ids."""
    ENTRADA_ENUNCIADO.write_text("\n".join(LINHAS_ENUNCIADO) + "\n", encoding="utf-8")
    return ["1"]


def rodar_prever(entrada, saida):
    comando = [sys.executable, "-m", "src.prever", "--entrada", str(entrada), "--modelo", "ambos", "--saida-dir", str(saida)]
    return subprocess.run(comando, cwd=RAIZ, capture_output=True, text=True)


def conferir(caminho, ids, limiares, faixa):
    """(descrição, ok) de cada checagem de um arquivo de predições."""
    linhas = caminho.read_text(encoding="utf-8").splitlines()
    casadas = [FORMATO_LINHA.match(linha) for linha in linhas]
    campos = [m.groups() for m in casadas if m]
    gravid = [float(g) for _, g, _ in campos]
    classe = [int(c) for _, _, c in campos]
    minimo, maximo = (round(v, CASAS_GRAVID) for v in faixa)
    return [
        (f"{len(linhas)} linhas para {len(ids)} vítimas", len(linhas) == len(ids)),
        ("todas as linhas no formato i,gravid,classe (gravid com 4 casas, classe de 1 a 4)", all(casadas)),
        ("ids na mesma ordem da entrada", [i for i, _, _ in campos] == ids),
        ("classe coerente com gravid pelos limiares", len(campos) == len(linhas) and classe_por_limiares(gravid, limiares).tolist() == classe),
        (f"todo gravid dentro de [{minimo:.4f}; {maximo:.4f}]", len(campos) == len(linhas) and all(minimo <= g <= maximo for g in gravid)),
    ]


def caso(nome, entrada, saida, ids, limiares, faixa):
    """Roda o prever.py numa entrada e devolve as checagens do caso."""
    print(f"--- caso {nome}: {entrada} ({len(ids)} vítima(s))")
    execucao = rodar_prever(entrada, saida)
    print(execucao.stdout, end="")
    print(execucao.stderr, end="", file=sys.stderr)
    checagens = [(f"{nome}: prever.py terminou com código 0", execucao.returncode == 0)]
    if execucao.returncode != 0:
        return checagens
    return checagens + [
        (f"{nome}: predicoes_{m}.txt: {d}", ok)
        for m in SAIDAS
        for d, ok in conferir(saida / f"predicoes_{m}.txt", ids, limiares, faixa)
    ]


def main():
    PASTA_ENUNCIADO.mkdir(parents=True, exist_ok=True)
    rotulados = ler_rotulados()
    _, teste = dividir_hold_out(rotulados)
    limiares = json.loads((MODELOS / "limiares.json").read_text(encoding="utf-8"))["limiares"]
    faixa = faixa_g(rotulados)
    checagens = caso("hold-out", ENTRADA_HOLDOUT, PASTA, gerar_entrada_holdout(teste.sort_values("id")), limiares, faixa)
    checagens += caso("enunciado", ENTRADA_ENUNCIADO, PASTA_ENUNCIADO, gerar_entrada_enunciado(), limiares, faixa)
    print("--- checagens")
    for descricao, ok in checagens:
        print(f"[{'ok' if ok else 'NÃO'}] {descricao}")
    falhas = sum(not ok for _, ok in checagens)
    if falhas:
        sys.exit(f"{falhas} checagem(ns) falharam")
    print("Teste de fumaça passou.")


if __name__ == "__main__":
    main()

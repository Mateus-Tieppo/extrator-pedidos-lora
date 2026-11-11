import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

from contrato import CAMPOS, parse
from dados import alvo, carregar_teste, treino, verificar_particoes


def carregar_auditoria():
    caminho = Path(__file__).parent / "dados" / "auditoria.tsv"
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        return [
            (linha["grupo"], linha["pedido"], alvo(
                int(linha["centavos"]) if linha["centavos"] else None,
                int(linha["minutos"]) if linha["minutos"] else None,
                linha["restricoes"].split(",") if linha["restricoes"] else (),
                linha["ingredientes"].split(",") if linha["ingredientes"] else (),
            ))
            for linha in csv.DictReader(arquivo, delimiter="\t")
        ]


def tirar_cerca(texto):
    cerca = re.fullmatch(r"```json\s*([\s\S]*?)\s*```", texto.strip(), flags=re.I)
    return cerca[1] if cerca else texto


def resumir(casos, saidas):
    if len(casos) != len(saidas) or not casos:
        raise ValueError("Cada caso deve ter uma resposta")
    cont = Counter(dict.fromkeys((
        "total", "validos", "exatos", "com_restricao", "com_exclusao",
        "restricao_omitida_em_saida_valida", "restricao_em_saida_invalida",
        "restricao_inventada", "ingrediente_omitido_em_saida_valida", "exclusao_em_saida_invalida",
        "ingrediente_inventado",
    ), 0))
    for (_, _, esperado), raw in zip(casos, saidas, strict=True):
        cont["total"] += 1
        restricoes, ingredientes = (set(esperado[c]) for c in CAMPOS[2:])
        cont["com_restricao"] += bool(restricoes)
        cont["com_exclusao"] += bool(ingredientes)
        try:
            obtido = parse(raw)
        except ValueError:
            cont["restricao_em_saida_invalida"] += bool(restricoes)
            cont["exclusao_em_saida_invalida"] += bool(ingredientes)
            continue
        cont["validos"] += 1
        cont["exatos"] += obtido == esperado
        r, ing = (set(obtido[c]) for c in CAMPOS[2:])
        cont["restricao_omitida_em_saida_valida"] += bool(restricoes - r)
        cont["restricao_inventada"] += bool(r - restricoes)
        cont["ingrediente_omitido_em_saida_valida"] += bool(ingredientes - ing)
        cont["ingrediente_inventado"] += bool(ing - ingredientes)
    return dict(cont)


def prever_regras(backend, casos):
    sys.path.insert(0, str(backend.resolve(strict=True)))
    from app.demo.interpretador import interpretar

    resultados = []
    for _, pedido, _ in casos:
        dados = interpretar(pedido, None).criterios.model_dump(mode="json", include=set(CAMPOS))
        resultados.append(json.dumps(dados, ensure_ascii=False))
    return resultados


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--backend", required=True, type=Path)
    cli.add_argument("--adaptador", type=Path, default=Path("resultados/adaptador"))
    cli.add_argument("--saida", type=Path, default=Path("resultados/auditoria"))
    cli.add_argument("--somente-regras", action="store_true")
    args = cli.parse_args()
    casos = carregar_auditoria()
    pares = [(texto, esperado) for _, texto, esperado in casos]
    verificar_particoes(treino() + carregar_teste(), pares)
    saidas = {"regras": prever_regras(args.backend, casos)}
    if not args.somente_regras:
        from peft import PeftModel
        from experimento import carregar_modelo, prever

        if not (args.adaptador / "adapter_config.json").is_file():
            cli.error("Adaptador não encontrado")
        modelo, tokenizer, device = carregar_modelo()
        modelo = PeftModel.from_pretrained(modelo, str(args.adaptador), local_files_only=True)
        with modelo.disable_adapter():
            saidas["base_original"], _ = prever(modelo, tokenizer, pares, device)
        saidas["base_sem_cerca"] = [tirar_cerca(p) for p in saidas["base_original"]]
        saidas["lora"], _ = prever(modelo, tokenizer, pares, device)

    resumo = {nome: resumir(casos, respostas) for nome, respostas in saidas.items()}
    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / "metricas.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.saida / "predicoes.jsonl").open("w", encoding="utf-8") as arq:
        for i, (grupo, pedido, esperado) in enumerate(casos):
            linha = {"grupo": grupo, "pedido": pedido, "esperado": esperado}
            linha.update({nome: respostas[i] for nome, respostas in saidas.items()})
            arq.write(json.dumps(linha, ensure_ascii=False) + "\n")
    print(json.dumps(resumo, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

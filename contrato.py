import json
import re


CAMPOS = ("orcamento_max_centavos", "tempo_max_min", "restricoes", "ingredientes_excluidos")
RESTRICOES = {
    "sem_lactose", "sem_gluten", "vegetariano", "vegano", "sem_amendoim", "sem_frutos_do_mar", "sem_ovo",
}


def _sem_duplicatas(pares):
    resultado = dict(pares)
    if len(resultado) != len(pares):
        raise ValueError("Chave JSON repetida")
    return resultado


def parse(texto):
    dados = json.loads(texto, object_pairs_hook=_sem_duplicatas)
    if not isinstance(dados, dict) or set(dados) != set(CAMPOS):
        raise ValueError("O JSON deve conter exatamente os quatro campos")
    for campo in CAMPOS[:2]:
        valor = dados[campo]
        if valor is not None and (type(valor) is not int or valor <= 0):
            raise ValueError(f"{campo} deve ser um inteiro positivo ou null")
    for campo in CAMPOS[2:]:
        itens = dados[campo]
        if not isinstance(itens, list) or any(not isinstance(item, str) for item in itens):
            raise ValueError(f"{campo} deve ser uma lista de strings")
        if len(set(itens)) != len(itens):
            raise ValueError(f"{campo} tem valores repetidos")
        dados[campo] = sorted(itens)
    if not set(dados["restricoes"]) <= RESTRICOES:
        raise ValueError("Restrição fora do catálogo")
    if any(not re.fullmatch(r"[a-z]+(?: [a-z]+)*", item) for item in dados["ingredientes_excluidos"]):
        raise ValueError("Ingredientes devem usar minúsculas, sem acentos")
    return {campo: dados[campo] for campo in CAMPOS}


def avaliar(esperados, respostas):
    if not esperados or len(esperados) != len(respostas):
        raise ValueError("Cada pedido deve ter exatamente uma resposta")
    validos = exatos = 0
    acertos = dict.fromkeys(CAMPOS, 0)
    for esperado, resposta in zip(esperados, respostas, strict=True):
        try:
            previsto = parse(resposta)
        except ValueError:
            continue
        validos += 1
        exatos += previsto == esperado
        for campo in CAMPOS:
            acertos[campo] += previsto[campo] == esperado[campo]
    return {"total": len(esperados), "validos": validos, "exatos": exatos, "acertos_por_campo": acertos}

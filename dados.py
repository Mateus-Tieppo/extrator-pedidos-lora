import csv
import json
import random
from pathlib import Path

from contrato import CAMPOS, parse


def alvo(preco=None, tempo=None, restricoes=(), ingredientes=()):
    dados = dict(zip(CAMPOS, (preco, tempo, list(restricoes), list(ingredientes)), strict=True))
    return parse(json.dumps(dados))


def carregar_teste():
    caminho = Path(__file__).parent / "dados" / "teste.tsv"
    with caminho.open(encoding="utf-8", newline="") as arquivo:
        linhas = csv.DictReader(arquivo, delimiter="\t")
        return [
            (
                l["pedido"],
                alvo(
                    int(l["centavos"]) if l["centavos"] else None,
                    int(l["minutos"]) if l["minutos"] else None,
                    l["restricoes"].split(",") if l["restricoes"] else (),
                    l["ingredientes"].split(",") if l["ingredientes"] else (),
                ),
            )
            for l in linhas
        ]


RESTRICOES = (
    ("sem glúten", "sem_gluten"),
    ("zero lactose", "sem_lactose"),
    ("tem que ser vegano", "vegano"),
    ("opção vegetariana", "vegetariano"),
    ("sem amendoim", "sem_amendoim"),
    ("sem frutos do mar", "sem_frutos_do_mar"),
    ("não pode ter ovo", "sem_ovo"),
)
INGREDIENTES = (
    ("cebola", "cebola"), ("alho", "alho"), ("camarão", "camarao"),
    ("coentro", "coentro"), ("tomate", "tomate"), ("pimentão", "pimentao"),
    ("azeitona", "azeitona"), ("presunto", "presunto"), ("cogumelo", "cogumelo"),
    ("pimenta", "pimenta"), ("carne de porco", "carne de porco"), ("maionese", "maionese"),
)
PREFERENCIAS = (
    "tô pensando num prato leve", "quero almoçar", "hoje queria comida caseira",
    "se tiver algo quente melhor", "quero algo baratinho", "me indica um lanche",
    "alguma coisa pra jantar", "me surpreenda", "quero algo gostoso e rápido",
)


def treino(total=360, seed=42):
    rng = random.Random(seed)
    exemplos = {}
    while len(exemplos) < total:
        partes = []
        preco = tempo = None
        restricoes, ingredientes = [], []
        if rng.random() < 0.65:
            reais = rng.randrange(18, 86)
            centavos = rng.choice((0, 0, 0, 50, 90, 75))
            preco = reais * 100 + centavos
            valor = f"{reais},{centavos:02d}" if centavos else str(reais)
            partes.append(rng.choice(("até R$ {}", "no máximo {} reais", "limite de R$ {}")).format(valor))
        if rng.random() < 0.60:
            if rng.random() < 0.14:
                frase, tempo = rng.choice((("em até meia hora", 30), ("dentro de uma hora", 60)))
            else:
                tempo = rng.randrange(15, 76)
                frase = rng.choice(("até {} min", "em no máximo {} minutos", "não pode passar de {} minutos")).format(tempo)
            partes.append(frase)
        if rng.random() < 0.59:
            grupos = rng.sample(RESTRICOES, 2 if rng.random() < 0.21 else 1)
            if {r for _, r in grupos} == {"vegano", "vegetariano"}:
                grupos = grupos[:1]
            partes.extend(frase for frase, _ in grupos)
            restricoes = [r for _, r in grupos]
        if rng.random() < 0.56:
            itens = rng.sample(INGREDIENTES, 2 if rng.random() < 0.38 else 1)
            nomes = " nem ".join(frase for frase, _ in itens)
            partes.append(rng.choice(("sem {}", "não coloca {}", "tira {} por favor")).format(nomes))
            ingredientes = [item for _, item in itens]
        if partes and rng.random() < 0.55:
            rng.shuffle(partes)
        if not partes or rng.random() < 0.65:
            partes.insert(0, rng.choice(PREFERENCIAS))
        pedido = rng.choice((", ", " e ", ". ")).join(partes)
        exemplos[pedido] = alvo(preco, tempo, restricoes, ingredientes)
    return list(exemplos.items())


def verificar_particoes(treino_, teste_):
    def chaves(linhas):
        return {" ".join(texto.casefold().split()) for texto, _ in linhas}

    a, b = chaves(treino_), chaves(teste_)
    if a & b or len(a) != len(treino_) or len(b) != len(teste_):
        raise ValueError("Pedido repetido dentro ou entre conjuntos")

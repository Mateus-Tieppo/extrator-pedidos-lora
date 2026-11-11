import unittest

from contrato import parse, avaliar


VALIDO = '{"orcamento_max_centavos":4500,"tempo_max_min":30,"restricoes":["sem_gluten"],"ingredientes_excluidos":["cebola","camarao"]}'
VAZIO = '{"orcamento_max_centavos":null,"tempo_max_min":null,"restricoes":[],"ingredientes_excluidos":[]}'


class ContratoTest(unittest.TestCase):
    def test_ordena_ingredientes_sem_mudar_seu_significado(self):
        self.assertEqual(parse(VALIDO)["ingredientes_excluidos"], ["camarao", "cebola"])

    def test_rejeita_chave_extra_duplicada_e_valores_fora_do_contrato(self):
        invalidos = (
            VALIDO.replace('"tempo_max_min":30', '"tempo_max_min":true'),
            VALIDO.replace('"tempo_max_min":30', '"tempo_max_min":-30'),
            VALIDO.replace('"tempo_max_min":30', '"tempo_max_min":30.0'),
            VALIDO.replace('"tempo_max_min":30', '"tempo_max_min":30,"tempo_max_min":30'),
            VALIDO.replace('"tempo_max_min":30', '"tempo_max_min":30,"extra":3'),
            VALIDO.replace('"tempo_max_min":30,', ''),
            VALIDO.replace('"sem_gluten"', '"picante"'),
            VALIDO.replace('"cebola","camarao"', '"cebola","cebola"'),
            "Aqui está: " + VALIDO,
        )
        for entrada in invalidos:
            with self.subTest(entrada=entrada), self.assertRaises(ValueError):
                parse(entrada)

    def test_erros_de_formato_contam_no_denominador(self):
        resultado = avaliar([parse(VALIDO), parse(VAZIO)], [VALIDO, "erro"])
        self.assertEqual(resultado["total"], 2)
        self.assertEqual(resultado["validos"], 1)
        self.assertEqual(resultado["exatos"], 1)
        self.assertEqual(resultado["acertos_por_campo"]["tempo_max_min"], 1)

    def test_nao_ignora_previsoes_ausentes(self):
        with self.assertRaises(ValueError):
            avaliar([parse(VALIDO)], [])


if __name__ == "__main__":
    unittest.main()

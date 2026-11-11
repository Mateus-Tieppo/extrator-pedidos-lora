import json
import unittest

from auditoria import carregar_auditoria, resumir, tirar_cerca
from dados import alvo, carregar_teste, treino, verificar_particoes


class AuditoriaTest(unittest.TestCase):
    def test_separa_omissao_em_json_valido_de_saida_invalida(self):
        esperado = alvo(restricoes=("sem_ovo",), ingredientes=("cebola",))
        casos = [("teste", "sem ovo nem cebola", esperado)] * 2
        respostas = [json.dumps(alvo(restricoes=("sem_gluten",))), "texto"]
        medidas = resumir(casos, respostas)
        self.assertEqual(medidas["validos"], 1)
        self.assertEqual(medidas["com_restricao"], 2)
        self.assertEqual(medidas["restricao_omitida_em_saida_valida"], 1)
        self.assertEqual(medidas["restricao_em_saida_invalida"], 1)
        self.assertEqual(medidas["restricao_inventada"], 1)
        self.assertEqual(medidas["ingrediente_omitido_em_saida_valida"], 1)
        self.assertEqual(medidas["exclusao_em_saida_invalida"], 1)

    def test_remove_so_cerca_markdown_completa(self):
        corpo = json.dumps(alvo())
        self.assertEqual(tirar_cerca(f"```json\n{corpo}\n```"), corpo)
        frase = f"Resultado: ```json\n{corpo}\n```"
        self.assertEqual(tirar_cerca(frase), frase)

    def test_auditoria_tem_casos_novos_em_cinco_grupos(self):
        casos = carregar_auditoria()
        self.assertEqual(len(casos), 40)
        self.assertEqual(len({g for g, _, _ in casos}), 5)
        verificar_particoes(treino() + carregar_teste(), [(t, e) for _, t, e in casos])


if __name__ == "__main__":
    unittest.main()

# Extrator de pedidos de delivery em português

Um pedido como “Até R$ 45 com entrega, sem camarão e em 30 minutos” vira uma estrutura que um serviço pode validar e usar para filtrar um cardápio.

```json
{
  "orcamento_max_centavos": 4500,
  "tempo_max_min": 30,
  "restricoes": [],
  "ingredientes_excluidos": ["camarao"]
}
```

O contrato usa quatro dos campos do Sugestão da Casa. Restrições cadastradas, como `sem_lactose`, têm um campo separado de ingredientes específicos. Preferências vagas não viram números. Um limite ausente vira `null` e listas ausentes ficam vazias.

## Reproduzir

Use o [notebook para Google Colab](notebooks/Extrator-LoRA-Colab.ipynb) com GPU ou siga as instruções abaixo para rodar localmente.

Crie um ambiente com Python 3.11 e instale o PyTorch 2.8.0 adequado à sua máquina. O exemplo abaixo usa CUDA 12.6.

```sh
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m unittest -v
python experimento.py --passos 120 --seed 42
```

O script baixa a revisão fixada do [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Se você já tiver essa revisão em disco, informe `--modelo-local caminho/do/modelo`. O script usa CUDA quando disponível e CPU nos demais casos.

## Como a comparação funciona

O gerador oferece 360 exemplos sintéticos reproduzíveis. O piloto usa 120 desses exemplos, com um exemplo por passo. O teste contém 30 pedidos escritos à mão, com construções diferentes, correções, ingredientes múltiplos e limites em texto. A auditoria impede frases duplicadas ou compartilhadas entre os conjuntos. Ambos são sintéticos.

O treinamento atualiza apenas adaptadores LoRA nas projeções `q_proj` e `v_proj`. O cálculo da perda ignora os tokens da instrução e do pedido. Para comparar, o script desliga e religa o adaptador, com a mesma instrução, conjunto de teste e geração determinística nas duas versões.

**JSON válido** exige exatamente quatro campos, tipos corretos, ausência de chaves repetidas e valores compatíveis com o contrato. **Acerto completo** exige que os quatro campos coincidam com o gabarito. A ordem das listas não afeta o resultado. Saídas inválidas contam como erro em todas as métricas.

O script salva as métricas, as respostas brutas por pedido e o adaptador em `resultados/`. Ele não salva os pesos do modelo base.

## Resultado do primeiro treino

Foram 120 passos, com um pedido por passo e semente 42. O treino levou 21,32 s em uma Tesla T4 no Colab. O tempo não inclui download do modelo ou avaliação.

| Versão | JSON dentro do contrato | Quatro campos corretos |
| --- | ---: | ---: |
| Qwen base, resposta original | 0/30 | 0/30 |
| Qwen base, removendo a cerca Markdown | 28/30 | 1/30 |
| Qwen ajustado com LoRA | 27/30 | 15/30 |

O modelo base cercou todas as respostas com Markdown. A segunda linha remove só essa formatação, mantém as demais regras do contrato e ajuda a comparar a extração. O LoRA acertou orçamento em 25 pedidos, prazo em 24, restrições em 20 e ingredientes excluídos em 22, contando saídas inválidas como erro.

Ainda houve seis pedidos em que a saída passou pelo contrato mas omitiu pelo menos uma restrição cadastrada. Por exemplo, no pedido “Sem lactose e sem amendoim. Se tiver cebola, também não serve”, o LoRA excluiu cebola e deixou a lista de restrições vazia. O adaptador serve para pesquisa e demonstração. Antes de integrá-lo ao recomendador, precisamos avaliar restrições em um conjunto maior e manter a validação das exigências do pedido.

Para conferir o piloto, consulte as [métricas](resultados/metricas.json) e as [respostas por pedido](resultados/predicoes.jsonl). O notebook também recarrega o adaptador em um novo processo e testa o pedido de exemplo.

## Auditoria com 40 pedidos inéditos

Recarreguei o mesmo adaptador e comparei suas respostas com as do Qwen base e com o interpretador por palavras-chave do Sugestão da Casa. O conjunto tem cinco grupos de oito casos, sem sobreposição de texto com o treino ou com o teste inicial.

| Abordagem | JSON válido | Pedido correto | Restrição omitida em JSON válido | Saída inválida entre pedidos com restrição |
| --- | ---: | ---: | ---: | ---: |
| Regras do Sugestão da Casa | 40/40 | 18/40 | 6/26 | 0/26 |
| Qwen base, resposta original | 1/40 | 0/40 | 0/26 | 26/26 |
| Qwen base, sem cerca Markdown | 35/40 | 8/40 | 8/26 | 4/26 |
| Qwen ajustado com LoRA | 34/40 | 14/40 | 15/26 | 6/26 |

O denominador 26 corresponde aos pedidos com pelo menos uma restrição cadastrada no gabarito. A auditoria conta pedidos afetados e separa omissões em saídas válidas de respostas que não podem ser usadas. Assim, zero omissões do Qwen base na resposta original não indica que ele preservou restrições: todas as 26 saídas relevantes eram inválidas.

Em ingredientes específicos, o LoRA omitiu um em saída válida em 1 dos 10 pedidos com exclusões, contra 8 das regras. Também inventou pelo menos uma exclusão em 14 dos 40 pedidos, contra 2 das regras. Em um pedido direto com “sem lactose”, o LoRA devolveu uma lista vazia e as regras preservaram a restrição.

A decisão foi manter o LoRA como experimento de portfólio. Nesta amostra, as regras existentes acertaram mais pedidos completos e perderam menos restrições cadastradas. Para reproduzir, obtenha o [Sugestão da Casa na revisão avaliada](https://github.com/Mateus-Tieppo/sugestao-da-casa/tree/8902aa0c20e47f667beb27a93a5326938b91b391), instale `pydantic>=2,<3` e execute:

```sh
python auditoria.py --backend caminho/sugestao-da-casa/backend
```

Use `--somente-regras` para rodar sem carregar o Qwen. A pasta [resultados/auditoria](resultados/auditoria) contém as predições, as métricas e o SHA-256 do adaptador avaliado.

## Experimentar com um pedido próprio

Depois do treino, carregue o adaptador salvo e passe um pedido pela linha de comando:

```sh
python inferir.py --adaptador resultados/adaptador "Até R$ 45 com entrega, sem camarão e em 30 minutos"
```

O comando escreve apenas o JSON validado em caso de sucesso. Se a saída do modelo não seguir o contrato, termina com código 2. Omitir `--adaptador` usa o modelo base.

## Limitações

Os dois conjuntos têm 30 e 40 casos sintéticos e exploram distribuições diferentes. Os números não estimam desempenho com pedidos reais. Uma integração precisa verificar a interpretação do pedido e manter os filtros de cardápio antes de oferecer uma recomendação.

---
base_model: Qwen/Qwen2.5-0.5B-Instruct
base_model_relation: adapter
library_name: peft
language:
  - pt
tags:
  - lora
  - information-extraction
  - structured-output
---

# Extrator de pedidos de delivery em português

Adaptador LoRA experimental para o Qwen2.5-0.5B-Instruct. Converte um pedido em quatro campos: valor máximo em centavos, prazo máximo em minutos, restrições cadastradas e ingredientes excluídos.

O treino usou 120 exemplos de um conjunto de 360 pedidos sintéticos, com semente 42, rank 8 e projeções `q_proj` e `v_proj`. A revisão do modelo base foi `7ae557604adf67be50417f59c2c2f167def9a775`. O ajuste levou 21,32 s em uma Tesla T4.

Em 30 pedidos sintéticos escritos à mão e separados do treino, o modelo ajustado produziu 27 saídas dentro do contrato e acertou os quatro campos em 15 pedidos. O modelo base produziu 0 saídas dentro do contrato no formato original. Após remover apenas a cerca Markdown das respostas, ficou com 28 saídas válidas e 1 pedido completamente correto.

Em seis pedidos, o adaptador devolveu JSON válido e omitiu pelo menos uma restrição cadastrada. O teste pequeno e sintético limita a interpretação dos resultados. O adaptador é experimental e não é adequado para decidir sozinho sobre pedidos com restrições alimentares.

Em uma segunda auditoria com 40 pedidos inéditos, o adaptador acertou os quatro campos em 14 casos. O Qwen base, após remover a cerca Markdown, acertou 8. O interpretador por regras do Sugestão da Casa acertou 18. Dos 26 pedidos com restrição cadastrada, o LoRA omitiu pelo menos uma em 15 saídas válidas e retornou formato inválido em outros 6. Esses resultados motivaram a decisão de manter o modelo fora do recomendador.

## Usar o adaptador

Baixe `adapter_config.json` e `adapter_model.safetensors` para a mesma pasta. O script `inferir.py` do projeto carrega a revisão fixada do Qwen, aplica o adaptador e valida a resposta.

```sh
python inferir.py --adaptador caminho/do/adaptador "Até R$ 45 com entrega, sem camarão e em 30 minutos"
```

O adaptador foi treinado sobre o [Qwen2.5-0.5B-Instruct na revisão indicada acima](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/tree/7ae557604adf67be50417f59c2c2f167def9a775), distribuído pelo Qwen sob Apache 2.0.

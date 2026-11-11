import argparse
import json
import sys
from pathlib import Path

from contrato import parse


def main():
    cli = argparse.ArgumentParser(description="Extrai os critérios de um pedido e valida o JSON.")
    cli.add_argument("pedido")
    cli.add_argument("--adaptador", type=Path, help="Pasta local do adaptador LoRA. Omitir usa o modelo base.")
    cli.add_argument("--modelo-local", type=Path, help="Cópia local da revisão fixada do Qwen.")
    args = cli.parse_args()
    if not args.pedido.strip():
        cli.error("O pedido não pode ser vazio.")
    if args.adaptador and not (args.adaptador / "adapter_config.json").is_file():
        cli.error("A pasta do adaptador deve conter adapter_config.json.")

    import torch
    from peft import PeftModel

    from experimento import carregar_modelo, mensagens

    model, tokenizer, device = carregar_modelo(args.modelo_local)
    if args.adaptador:
        model = PeftModel.from_pretrained(model, str(args.adaptador), local_files_only=True)
    model.eval()
    entrada = tokenizer.apply_chat_template(
        mensagens(args.pedido), add_generation_prompt=True, return_dict=True, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        tokens = model.generate(**entrada, max_new_tokens=120, do_sample=False)
    resposta = tokenizer.decode(tokens[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        dados = parse(resposta)
    except ValueError as erro:
        print(f"Saída fora do contrato: {erro}", file=sys.stderr)
        return 2
    print(json.dumps(dados, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

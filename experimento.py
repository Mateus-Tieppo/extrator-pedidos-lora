import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from contrato import avaliar
from dados import carregar_teste, treino, verificar_particoes


MODELO = "Qwen/Qwen2.5-0.5B-Instruct"
REVISAO = "7ae557604adf67be50417f59c2c2f167def9a775"
PROMPT = """Extraia somente requisitos explícitos de um pedido em português.
Retorne apenas um objeto JSON com quatro chaves:
- orcamento_max_centavos: valor máximo em centavos, ou null. R$ 25,90 vira 2590.
- tempo_max_min: prazo máximo em minutos, ou null. Meia hora vira 30.
- restricoes: lista com itens aplicáveis dentre sem_lactose, sem_gluten, vegetariano,
  vegano, sem_amendoim, sem_frutos_do_mar, sem_ovo.
- ingredientes_excluidos: lista de ingredientes específicos recusados, minúsculos e sem acentos.
Use lista vazia quando não houver itens. Não repita na lista de ingredientes uma
restrição que já está na lista de restrições. Preferências vagas como barato ou rápido
não definem números. Considere a última exigência quando houver uma correção."""


def mensagens(pedido):
    return [{"role": "system", "content": PROMPT}, {"role": "user", "content": pedido}]


def carregar_modelo(modelo_local=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    origem = str(modelo_local) if modelo_local else MODELO
    versao = {"local_files_only": True} if modelo_local else {"revision": REVISAO}
    tokenizer = AutoTokenizer.from_pretrained(origem, padding_side="left", **versao)
    model = AutoModelForCausalLM.from_pretrained(
        origem, torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        use_safetensors=True, **versao
    ).to(device)
    return model, tokenizer, device


def tokenizar_treino(tokenizer, pedido, alvo):
    prefixo = tokenizer.apply_chat_template(mensagens(pedido), add_generation_prompt=True)
    resposta = {"role": "assistant", "content": json.dumps(alvo, ensure_ascii=False, separators=(",", ":"))}
    tokens = tokenizer.apply_chat_template(mensagens(pedido) + [resposta])
    if tokens[:len(prefixo)] != prefixo or len(tokens) <= len(prefixo):
        raise RuntimeError("O template do tokenizer não isolou a resposta do assistente")
    return tokens, [-100] * len(prefixo) + tokens[len(prefixo):]


def ajustar(base, tokenizer, exemplos, device, passos, seed, destino):
    model = get_peft_model(
        base,
        LoraConfig(task_type="CAUSAL_LM", revision=REVISAO, r=8, lora_alpha=16, lora_dropout=0.05,
                   target_modules=["q_proj", "v_proj"]),
    )
    model.print_trainable_parameters()
    model.train()
    otim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = random.Random(seed)
    ordem = rng.sample(exemplos, k=len(exemplos))
    inicio = perf_counter()
    for i in range(passos):
        if i and i % len(ordem) == 0:
            rng.shuffle(ordem)
        texto, alvo = ordem[i % len(ordem)]
        tokens, labels = tokenizar_treino(tokenizer, texto, alvo)
        entradas = torch.tensor([tokens], device=device)
        gabarito = torch.tensor([labels], device=device)
        otim.zero_grad(set_to_none=True)
        loss = model(input_ids=entradas, labels=gabarito, use_cache=False).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss inválida no passo {i + 1}")
        scaler.scale(loss).backward()
        scaler.unscale_(otim)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        scaler.step(otim)
        scaler.update()
        if i == 0 or (i + 1) % 15 == 0 or i + 1 == passos:
            print(f"Passo {i + 1}/{passos}: loss={loss.item():.4f}", flush=True)
    model.save_pretrained(destino, safe_serialization=True)
    return model, round(perf_counter() - inicio, 2)


@torch.inference_mode()
def prever(model, tokenizer, exemplos, device, lote=5):
    model.eval()
    saidas = []
    inicio = perf_counter()
    for i in range(0, len(exemplos), lote):
        conversas = [mensagens(pedido) for pedido, _ in exemplos[i:i + lote]]
        entradas = tokenizer.apply_chat_template(
            conversas, add_generation_prompt=True, padding=True, return_dict=True, return_tensors="pt"
        ).to(device)
        tokens = model.generate(**entradas, max_new_tokens=120, do_sample=False)
        novos = tokens[:, entradas["input_ids"].shape[1]:]
        saidas.extend(tokenizer.batch_decode(novos, skip_special_tokens=True))
        print(f"  {len(saidas)}/{len(exemplos)} previsões", flush=True)
    return saidas, round(perf_counter() - inicio, 2)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--passos", type=int, default=120)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--modelo-local", type=Path, help="Cópia local da revisão fixada")
    cli.add_argument("--saida", type=Path, default=Path("resultados"))
    args = cli.parse_args()
    if args.passos <= 0:
        cli.error("--passos deve ser positivo")
    args.saida.mkdir(parents=True, exist_ok=True)
    treino_, teste_ = treino(seed=args.seed), carregar_teste()
    verificar_particoes(treino_, teste_)
    torch.manual_seed(args.seed)
    model, tokenizer, device = carregar_modelo(args.modelo_local)
    print(f"Dispositivo: {device}. Treino: {len(treino_)}. Teste: {len(teste_)}.", flush=True)
    # Guardar a referência publicável mesmo quando os pesos vierem de uma cópia local.
    model.config._name_or_path = MODELO
    model.name_or_path = MODELO
    model, segundos_treino = ajustar(
        model, tokenizer, treino_, device, args.passos, args.seed, args.saida / "adaptador"
    )
    print("Avaliação do modelo base", flush=True)
    with model.disable_adapter():
        base, segundos_base = prever(model, tokenizer, teste_, device)
    print("Avaliação do modelo ajustado", flush=True)
    lora, segundos_lora = prever(model, tokenizer, teste_, device)
    esperados = [alvo for _, alvo in teste_]
    relatorio = {
        "modelo": MODELO, "revisao": REVISAO, "seed": args.seed,
        "dispositivo": str(device), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "exemplos_de_treino_disponiveis": len(treino_),
        "passos_treino": args.passos, "segundos_treino": segundos_treino,
        "base": {**avaliar(esperados, base), "segundos_inferencia": segundos_base},
        "lora": {**avaliar(esperados, lora), "segundos_inferencia": segundos_lora},
    }
    (args.saida / "metricas.json").write_text(json.dumps(relatorio, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (args.saida / "predicoes.jsonl").open("w", encoding="utf-8") as arq:
        for (pedido, esperado), b, l in zip(teste_, base, lora, strict=True):
            arq.write(json.dumps({"pedido": pedido, "esperado": esperado, "base": b, "lora": l}, ensure_ascii=False) + "\n")
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

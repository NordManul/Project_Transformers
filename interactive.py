"""Ввод текста, потоковое продолжение и просмотр выбора следующего токена."""
import argparse
import math

import torch

from baseline_gpt import load_checkpoint, load_tokenizer, select_device


def generate_text(model, tok, prompt, max_tokens=100, temperature=0.8, top_k=40, trace=False):
    ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
    device = next(model.parameters()).device
    context = torch.tensor([ids], dtype=torch.long, device=device)
    if len(ids) > model.cfg.block_size:
        print(f"Контекст: последние {model.cfg.block_size} из {len(ids)} токенов.")
    decode = lambda values: tok.decode(values, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)
    initial = decode(ids)
    printed = initial
    if not trace:
        print("Модель> " + initial, end="", flush=True)
    for step, (next_id, probabilities) in enumerate(model.generate_tokens(
            context, max_tokens, temperature, top_k, tok.eos_token_id), start=1):
        token_id = next_id.item()
        ids.append(token_id)
        if trace:
            values, candidates = torch.topk(probabilities[0], min(5, tok.vocab_size))
            options = ", ".join(f"{tok.convert_ids_to_tokens(index)!r}: {probability:.1%}"
                                for index, probability in zip(candidates.tolist(), values.tolist()))
            print(f"{step:3d}. token={token_id} {tok.convert_ids_to_tokens(token_id)!r} "
                  f"p={probabilities[0, token_id].item():.1%} | варианты: {options}", flush=True)
        else:
            decoded = decode(ids)
            # Byte-fallback может выдавать UTF-8 символ несколькими токенами.
            if not decoded.endswith("\ufffd"):
                print(decoded[len(printed):], end="", flush=True)
                printed = decoded
    result = decode(ids)
    if trace:
        print("Модель> " + result, flush=True)
    else:
        print(result[len(printed):], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/tinystories.best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--prompt", help="Один запрос вместо интерактивного режима")
    args = parser.parse_args()
    if args.max_tokens <= 0 or args.top_k <= 0 or args.threads <= 0:
        parser.error("max_tokens, top_k и threads должны быть положительными")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("temperature должна быть положительной")
    torch.set_num_threads(args.threads)
    try:
        device = select_device(args.device)
        model, checkpoint = load_checkpoint(args.checkpoint, device)
    except (FileNotFoundError, ValueError) as error:
        parser.error(f"{error}. Сначала обучите модель или укажите --checkpoint.")
    model.eval()
    tok = load_tokenizer()
    torch.manual_seed(args.seed)
    print(f"Загружена модель: шаг {checkpoint['step']}, val loss={checkpoint['best_val_loss']:.3f}, "
          f"устройство {device}")
    print("Модель продолжает текст. После обучения на TinyStories вводите начало истории на английском.")

    def run(prompt):
        return generate_text(model, tok, prompt, args.max_tokens, args.temperature, args.top_k, args.trace)

    if args.prompt is not None:
        run(args.prompt)
        return
    print("Команды: /exit — выход; /trace on — выбор токенов; /trace off — потоковый текст.")
    while True:
        try:
            prompt = input("\nТы> ")
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершено.")
            break
        if prompt.strip() in ("/exit", "/quit"):
            break
        if prompt.strip() in ("/trace on", "/trace off"):
            args.trace = prompt.strip() == "/trace on"
            print("Просмотр токенов " + ("включён." if args.trace else "выключен."))
            continue
        if not prompt.strip():
            continue
        try:
            run(prompt)
        except KeyboardInterrupt:
            print("\nГенерация остановлена; можно ввести новый текст.")


if __name__ == "__main__":
    main()

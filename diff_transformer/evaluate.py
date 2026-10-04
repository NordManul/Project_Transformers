"""
Оценка обученной модели:
  1) perplexity на валидационной части данных (последние 10% токенов, как при обучении);
  2) генерация продолжений для случайных фраз из валидации тремя способами —
     жадно, сэмплированием без эвристик и с эвристиками (sampling.py) —
     и сравнение их эвристическими метриками (metrics.py).

    python evaluate.py                         # последняя модель из checkpoints/
    python evaluate.py --ckpt checkpoints/baseline_small.pt --samples 50

Примеры генераций сохраняются в checkpoints/eval_<модель>.txt.
"""
import argparse
import math
import os
import random
import sys

import torch
import torch.nn.functional as F

from diff_transformer import encode_text, load_checkpoint, load_tokenizer
from chat import generate_stream, latest_checkpoint, stop_position
from metrics import build_vocab, format_table, summarize
from sampling import SamplingConfig


@torch.no_grad()
def perplexity(model, tokens, max_tokens=100_000, batch=16):
    """exp(средний cross-entropy) по непересекающимся окнам длины block_size."""
    T = model.cfg.block_size
    device = next(model.parameters()).device
    tokens = tokens[: max_tokens + 1]
    n_windows = (len(tokens) - 1) // T
    if n_windows == 0:
        raise ValueError("слишком мало токенов для оценки perplexity")
    windows = torch.stack([tokens[i * T: i * T + T + 1] for i in range(n_windows)])
    total, count = 0.0, 0
    for i in range(0, n_windows, batch):
        w = windows[i:i + batch].to(device)
        logits, _ = model(w[:, :-1])
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), w[:, 1:].reshape(-1),
                               reduction="sum")
        total += loss.item()
        count += w[:, 1:].numel()
    return math.exp(total / count)


def pick_prompts(val_text, k, n_words, seed):
    """Случайные строки из валидационного текста: первые n_words слов как начало фразы."""
    lines = [ln.split() for ln in val_text.splitlines()]
    lines = [ln for ln in lines if len(ln) >= n_words + 3]
    rng = random.Random(seed)
    return [" ".join(ln[:n_words]) for ln in rng.sample(lines, min(k, len(lines)))]


def main():
    parser = argparse.ArgumentParser(description="Оценка модели: perplexity и метрики генерации")
    parser.add_argument("--ckpt", default=None, help="по умолчанию самый свежий в checkpoints/")
    parser.add_argument("--data", default=None, help="по умолчанию тот файл, на котором училась модель")
    parser.add_argument("--samples", type=int, default=30, help="сколько фраз продолжать")
    parser.add_argument("--len", type=int, default=120, dest="length", help="токенов в продолжении")
    parser.add_argument("--prompt_words", type=int, default=6)
    parser.add_argument("--stop", default="none", choices=("none", "sentence", "paragraph"))
    parser.add_argument("--ppl_tokens", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for stream in (sys.stdout,):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    ckpt_path = args.ckpt or latest_checkpoint()
    if not ckpt_path or not os.path.exists(ckpt_path):
        sys.exit("Не найдена обученная модель в checkpoints/. Сначала запустите train.py.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(ckpt_path, device)
    tok = load_tokenizer()
    data_path = args.data or ckpt.get("data")
    if not data_path or not os.path.exists(data_path):
        sys.exit(f"Не найден файл данных {data_path!r}. Укажите его через --data.")
    print(f"Модель: {ckpt_path} (шаг {ckpt.get('step')}), данные: {data_path}, {device}")

    # --- 1. perplexity на валидации ---
    text = open(data_path, encoding="utf-8").read()
    cache = f"{data_path}.mistral.pt"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(data_path):
        tokens = torch.load(cache)
    else:
        print("Токенизирую данные...")
        tokens = torch.tensor(encode_text(tok, text), dtype=torch.long)
    val_tokens = tokens[int(0.9 * len(tokens)):]
    ppl = perplexity(model, val_tokens, args.ppl_tokens)
    print(f"\nPerplexity на валидации: {ppl:.2f}  (loss {math.log(ppl):.3f}; "
          f"случайная модель дала бы {tok.vocab_size})")

    # --- 2. генерация и метрики ---
    split = int(0.9 * len(text))
    vocab = build_vocab(text[max(0, split - 20_000_000):split])   # слова из обучающей части
    prompts = pick_prompts(text[split:], args.samples, args.prompt_words, args.seed)
    heur = SamplingConfig()
    configs = {
        "жадный": SamplingConfig(temperature=0).plain(),
        "без эвристик": heur.plain(),
        "с эвристиками": heur,
    }
    print(f"\nГенерирую {len(prompts)} продолжений по {args.length} токенов для каждой настройки...")
    generations = {}
    for name, cfg in configs.items():
        outs = []
        for i, prompt in enumerate(prompts):
            torch.manual_seed(args.seed + i)          # одинаковая случайность для всех настроек
            ids = tok.encode(prompt, add_special_tokens=False)
            new = list(generate_stream(model, ids, args.length, eos_id=tok.eos_token_id, cfg=cfg))
            full = tok.decode(ids + new, skip_special_tokens=True)
            gen = full[len(tok.decode(ids, skip_special_tokens=True)):]
            cut = stop_position(gen, args.stop)
            outs.append(gen[:cut] if cut is not None else gen)
        generations[name] = outs

    results = {name: summarize(outs, vocab) for name, outs in generations.items()}
    print("\n" + format_table(results))
    print("\nНастройки: " + "; ".join(f"{n}: {c.describe()}" for n, c in configs.items()))

    # --- примеры в файл ---
    report = os.path.join(os.path.dirname(ckpt_path) or ".",
                          f"eval_{os.path.splitext(os.path.basename(ckpt_path))[0]}.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write(f"Модель {ckpt_path}, perplexity {ppl:.2f}\n\n{format_table(results)}\n")
        for i, prompt in enumerate(prompts):
            f.write(f"\n{'=' * 70}\nНАЧАЛО: {prompt}\n")
            for name in configs:
                f.write(f"\n--- {name} ---\n{prompt}{generations[name][i]}\n")
    print(f"\nПримеры генераций: {report}")
    print(f"\nПример — {prompts[0]!r}:")
    for name in configs:
        print(f"  [{name}] {prompts[0]}{generations[name][0][:300]!s}".replace("\n", " "))


if __name__ == "__main__":
    main()

"""
Обучение Transformer / Differential Transformer на текстовом файле.

    python train.py --arch diff --preset small --data data/tinystories.txt --max_iters 3000 --lr 6e-4
    python train.py --arch baseline ...            # обычный трансформер для сравнения
    python train.py --arch diff --head_norm off    # абляция: без GroupNorm
    python train.py --arch diff --lambda_init 0.8  # абляция: постоянный λ_init

Модель, токенизатор, оптимизатор и расписание lr — из ../baseline_gpt.py (без изменений),
дифференциальное внимание — из diff_transformer.py. Лучшая по val модель сохраняется
в checkpoints/<arch>_<preset>.pt.

Эвристики обучения: ранняя остановка по val (--patience), автоподбор batch под видеопамять
с накоплением градиентов, пропуск шагов с nan/inf.
"""
import argparse
import math
import os
import sys
import time
from contextlib import nullcontext

import torch

from diff_transformer import (
    PRESETS, DiffConfig, DiffGPT, encode_text, load_tokenizer, lr_at, make_optimizer, save_checkpoint,
)


def fit_batch_size(model, make_batch, batch_size, autocast=nullcontext()):
    """Пробует сделать forward+backward с batch_size; при нехватке видеопамяти
    уменьшает batch вдвое. Возвращает наибольший batch, который поместился."""
    bs = batch_size
    while True:
        try:
            x, y = make_batch(bs)
            with autocast:
                _, loss = model(x, y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            return bs
        except torch.cuda.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            bs //= 2


# ============================ Обучение ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="tiny", choices=PRESETS.keys())
    parser.add_argument("--data", default="input.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200, help="в статье 2000 (на длинных прогонах)")
    parser.add_argument("--lr", type=float, default=1e-3, help="в статье подбирался под задачу")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--eval_iters", type=int, default=50, help="батчей на одну оценку loss")
    parser.add_argument("--log_every", type=int, default=50, help="как часто печатать loss батча")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="в статье 0; на маленьком тексте 0.1-0.2 уменьшает переобучение")
    parser.add_argument("--out", default=None,
                        help="куда сохранять лучшую модель (по умолчанию checkpoints/baseline_<preset>.pt)")
    # --- архитектура (Differential Transformer, arXiv:2410.05258) ---
    parser.add_argument("--arch", default="diff", choices=("baseline", "diff"))
    parser.add_argument("--n_head", type=int, default=None,
                        help="число голов обычного внимания (для diff голов будет вдвое меньше)")
    parser.add_argument("--head_norm", choices=("auto", "on", "off"), default="auto",
                        help="GroupNorm на каждую голову: auto — да для diff, нет для baseline")
    parser.add_argument("--lambda_init", type=float, default=None,
                        help="константа λ_init для diff (по умолчанию 0.8 - 0.6·exp(-0.3·(l-1)))")
    parser.add_argument("--name", default=None, help="имя модели (по умолчанию <arch>_<preset>)")
    parser.add_argument("--seed", type=int, default=42)
    # --- эвристики обучения ---
    parser.add_argument("--patience", type=int, default=3,
                        help="ранняя остановка: сколько оценок подряд val может не улучшаться (0 — выкл.)")
    parser.add_argument("--min_delta", type=float, default=0.005,
                        help="улучшение val меньше этого не считается улучшением")
    parser.add_argument("--no_auto_batch", action="store_true",
                        help="не подбирать batch под видеопамять автоматически")
    args = parser.parse_args()
    name = args.name or f"{args.arch}_{args.preset}"
    out_path = args.out or os.path.join("checkpoints", f"{name}.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    # --- данные (в статье — OpenWebText) ---
    if os.path.exists(args.data):
        text = open(args.data, encoding="utf-8").read()
    else:
        print(f"{args.data} не найден — использую короткий встроенный текст")
        text = "Привет, мир! Это базовый GPT из статьи про nGPT. " * 2000

    # --- токенизация: Mistral 7B, словарь 32 000 ---
    tok = load_tokenizer()
    vocab_size = tok.vocab_size
    cache = f"{args.data}.mistral.pt"
    if os.path.exists(args.data) and os.path.exists(cache) \
            and os.path.getmtime(cache) >= os.path.getmtime(args.data):
        data = torch.load(cache)
        print(f"Токены загружены из кэша {cache}")
    else:
        data = torch.tensor(encode_text(tok, text), dtype=torch.long)
        if os.path.exists(args.data):
            torch.save(data, cache)
    bos_id = tok.bos_token_id
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True)

    print(f"Символов: {len(text):,} | токенов: {len(data):,} | "
          f"символов на токен: {len(text) / len(data):.2f}")

    preset = dict(PRESETS[args.preset])
    if args.n_head:
        preset["n_head"] = args.n_head
    cfg = DiffConfig(vocab_size=vocab_size, dropout=args.dropout, arch=args.arch,
                      head_norm={"auto": None, "on": True, "off": False}[args.head_norm],
                      lambda_init=args.lambda_init, **preset)
    T = cfg.block_size
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    if len(splits["val"]) <= T + 1:
        raise ValueError(f"Слишком мало данных: в val-части {len(splits['val'])} токенов, "
                         f"нужно больше block_size={T}. Возьмите текст побольше.")

    micro_bs = args.batch_size   # может уменьшиться автоподбором (тогда включится накопление градиентов)
    # отдельный генератор для батчей: у разных архитектур одинаковая последовательность данных
    train_gen = torch.Generator().manual_seed(args.seed)

    def get_batch(split, bs=None, gen=None):
        d = splits[split]
        ix = torch.randint(len(d) - T - 1, (bs or micro_bs,), generator=gen or train_gen)
        x = torch.stack([d[i:i + T] for i in ix])
        y = torch.stack([d[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model, iters=args.eval_iters):
        model.eval()
        out = {}
        for split in ("train", "val"):
            gen = torch.Generator().manual_seed(0)   # одни и те же батчи при каждой оценке и у всех моделей
            losses = torch.zeros(iters)
            for i in range(iters):
                x, y = get_batch(split, gen=gen)
                with autocast:
                    _, loss = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model = DiffGPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Модель {name}: {cfg.arch}, пресет {args.preset}, {n_params / 1e6:.2f}M параметров, "
          f"устройство {device}, bf16={use_bf16}")
    if cfg.arch == "diff":
        lams = [round(b.attn.lambda_init, 3) for b in model.blocks]
        print(f"  diff-голов {cfg.n_head // 2} x d={cfg.d_model // cfg.n_head}, λ_init по слоям: {lams}, "
              f"GroupNorm: {model.blocks[0].attn.head_norm is not None}")
    print(f"Начальный loss должен быть около ln({vocab_size}) = {math.log(vocab_size):.2f}")
    optimizer = make_optimizer(model, args.lr)

    # Эвристика 1: автоподбор batch под видеопамять. Если полный batch не влезает,
    # уменьшаем его вдвое и накапливаем градиенты, чтобы эффективный batch не изменился.
    if device == "cuda" and not args.no_auto_batch:
        micro_bs = fit_batch_size(model, lambda bs: get_batch("train", bs), args.batch_size, autocast)
    grad_accum = math.ceil(args.batch_size / micro_bs)
    if grad_accum > 1:
        print(f"Batch {args.batch_size} не влезает в память: {micro_bs} x {grad_accum} шагов накопления")

    best_val = float("inf")
    interrupted = False
    evals_without_improvement = 0
    bad_steps = 0
    t0 = time.time()
    try:
        for step in range(args.max_iters + 1):
            lr = lr_at(step, args.lr, args.warmup, args.max_iters)
            for g in optimizer.param_groups:
                g["lr"] = lr
            if step % args.eval_every == 0 or step == args.max_iters:
                l = estimate_loss(model)
                mark = ""
                if l["val"] < best_val - args.min_delta:
                    best_val = l["val"]
                    evals_without_improvement = 0
                    save_checkpoint(out_path, model, step=step, val_loss=best_val,
                                    preset=args.preset, data=args.data, name=name)
                    mark = f"  -> сохранено в {out_path}"
                else:
                    evals_without_improvement += 1
                    mark = f"  (без улучшения {evals_without_improvement}/{args.patience or '∞'})"
                print(f"шаг {step:5d} | lr {lr:.2e} | train {l['train']:.3f} | val {l['val']:.3f} "
                      f"| ppl {math.exp(min(l['val'], 20)):.1f}{mark}", flush=True)
                # Эвристика 2: ранняя остановка — val перестал улучшаться, дальше только переобучение
                if args.patience and evals_without_improvement >= args.patience:
                    print(f"Ранняя остановка: val не улучшался {args.patience} оценки подряд "
                          "— дальше модель обычно только запоминает обучающий текст.")
                    break
            if step == args.max_iters:
                break

            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(grad_accum):
                x, y = get_batch("train")
                with autocast:
                    _, loss = model(x, y)
                (loss / grad_accum).backward()
                total += loss.item() / grad_accum
            # Эвристика 3: защита от расходимости — шаг с nan/inf пропускаем, а не портим веса
            if not math.isfinite(total):
                bad_steps += 1
                optimizer.zero_grad(set_to_none=True)
                print(f"  шаг {step}: loss = {total}, шаг пропущен ({bad_steps}/10)", flush=True)
                if bad_steps >= 10:
                    print("Обучение разошлось. Уменьшите --lr (например, вдвое) и запустите заново.")
                    break
                continue
            bad_steps = 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if args.log_every and step % args.log_every == 0 and step % args.eval_every != 0:
                elapsed = time.time() - t0
                print(f"  шаг {step:5d} | loss {total:.3f} | {elapsed / (step + 1):.2f} с/шаг", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print("\nОбучение остановлено (Ctrl+C). Лучшая модель уже сохранена.")

    if best_val == float("inf"):
        print("Чекпоинт не был сохранён — обучение прервано до первой оценки.")
        return
    print(f"\nЛучший val loss {best_val:.3f}, модель: {out_path}")
    print(f"Интерактивный режим: python chat.py --ckpt {out_path}")
    if interrupted:
        sys.exit(130)

    # пример генерации лучшей моделью
    model.load_state_dict(torch.load(out_path, map_location=device, weights_only=True)["model"])
    model.eval()
    start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    with autocast:
        out = model.generate(start, max_new_tokens=300, temperature=0.8, top_k=20)
    print("\n--- Сгенерированный текст ---\n" + decode(out[0].tolist()))


if __name__ == "__main__":
    main()

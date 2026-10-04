"""
Интерактивный режим: вводите начало фразы, обученная модель продолжает его.

    python chat.py                                   # самая свежая модель из checkpoints/
    python chat.py --ckpt checkpoints/baseline_small.pt --temp 0.7
    python chat.py --prompt "Once upon a time"       # один ответ без диалога
    python chat.py --plain                           # без эвристик (только temperature и top_k)

Эвристики генерации (см. sampling.py) включены по умолчанию.
Команды внутри диалога:
    /temp 0.7   температура (меньше — предсказуемее, больше — разнообразнее)
    /topk 40    выбирать только из k самых вероятных токенов (0 — без ограничения)
    /topp 0.9   nucleus sampling: из токенов с суммарной вероятностью p (1 — выкл.)
    /rep 1.15   штраф за повтор уже встречавшихся токенов (1 — выкл.)
    /ngram 4    запрет повторять n-граммы токенов (0 — выкл.)
    /stop paragraph | sentence | none   где останавливаться
    /plain      выключить эвристики, /heur — включить обратно
    /compare    следующую фразу продолжить двумя способами: без эвристик и с ними
    /len 200    максимум токенов в ответе
    /seed 1     зафиксировать случайность (одинаковый ввод -> одинаковый ответ)
    \\n          в тексте фразы означает перевод строки, например: ROMEO:\\n
    /help       подсказка
    /exit       выход (или пустой Enter / Ctrl+C)
"""
import argparse
import os
import re
import sys
from dataclasses import replace

import torch

from diff_transformer import load_checkpoint, load_tokenizer
from sampling import SamplingConfig, sample_next

CKPT_DIR = "checkpoints"
STOP_MODES = ("none", "sentence", "paragraph")
_SENTENCE_END = re.compile(r"[.!?][\"')]*(?=\s)")


def latest_checkpoint():
    """Самый свежий .pt в папке checkpoints/ (последняя обученная модель)."""
    if not os.path.isdir(CKPT_DIR):
        return None
    files = [os.path.join(CKPT_DIR, f) for f in os.listdir(CKPT_DIR) if f.endswith(".pt")]
    return max(files, key=os.path.getmtime) if files else None


def stop_position(generated, mode):
    """Где обрезать сгенерированный текст (индекс) или None, если ещё рано.
    sentence  — после первого конца предложения (. ! ? и пробел/перевод строки за ним);
    paragraph — перед первой пустой строкой (конец рассказа / реплики)."""
    body_start = len(generated) - len(generated.lstrip())   # пропускаем пробелы в начале
    if mode == "paragraph":
        pos = generated.find("\n\n", body_start)
        return pos if pos >= 0 else None
    if mode == "sentence":
        m = _SENTENCE_END.search(generated, body_start)
        return m.end() if m else None
    return None


@torch.no_grad()
def generate_stream(model, ids, max_new_tokens, temperature=0.8, top_k=40, eos_id=None, cfg=None):
    """Генерирует токены по одному и сразу отдаёт их (yield), чтобы печатать текст по мере появления.
    cfg — SamplingConfig с эвристиками; без него используются только temperature и top_k.
    Контекст длиннее block_size обрезается слева."""
    if cfg is None:
        cfg = SamplingConfig(temperature=temperature, top_k=top_k).plain()
    device = next(model.parameters()).device
    context = list(ids)
    idx = torch.tensor([context], dtype=torch.long, device=device)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.cfg.block_size:])
        next_id = sample_next(logits[0, -1], context[-model.cfg.block_size:], cfg)
        if eos_id is not None and next_id == eos_id:
            return
        yield next_id
        context.append(next_id)
        idx = torch.cat([idx, torch.tensor([[next_id]], device=device)], dim=1)


def complete(model, tok, prompt, max_new_tokens, temperature=0.8, top_k=40, out=sys.stdout,
             cfg=None, stop="none"):
    """Печатает промпт и его продолжение по мере генерации. Возвращает весь текст."""
    ids = tok.encode(prompt, add_special_tokens=False) if prompt else [tok.bos_token_id]
    prompt_text = tok.decode(ids, skip_special_tokens=True)
    printed = prompt_text
    out.write(printed)
    out.flush()
    for next_id in generate_stream(model, ids, max_new_tokens, temperature, top_k,
                                   tok.eos_token_id, cfg):
        ids.append(next_id)
        text = tok.decode(ids, skip_special_tokens=True)
        # байтовые токены могут дать «половину» русской буквы — ждём следующий токен
        if text.endswith("\ufffd") or not text.startswith(printed):
            continue
        cut = stop_position(text[len(prompt_text):], stop) if text.startswith(prompt_text) else None
        if cut is not None:
            text = text[:len(prompt_text) + cut]
        if text.startswith(printed):
            out.write(text[len(printed):])
        printed = text
        out.flush()
        if cut is not None:
            break
    else:
        final = tok.decode(ids, skip_special_tokens=True)
        if final.startswith(printed) and final != printed:
            out.write(final[len(printed):])
            printed = final
    out.write("\n")
    out.flush()
    return printed


HELP = ("Введите начало фразы — модель продолжит его. Перевод строки внутри фразы — \\n.\n"
        "Команды: /temp /topk /topp /rep /ngram /stop /plain /heur /compare /len /seed /help /exit")


def main():
    parser = argparse.ArgumentParser(description="Интерактивное продолжение текста обученной моделью")
    parser.add_argument("--ckpt", default=None,
                        help="путь к чекпоинту (по умолчанию самый свежий в checkpoints/)")
    d = SamplingConfig()
    parser.add_argument("--temp", type=float, default=d.temperature)
    parser.add_argument("--top_k", type=int, default=d.top_k)
    parser.add_argument("--top_p", type=float, default=d.top_p)
    parser.add_argument("--rep", type=float, default=d.repetition_penalty, help="штраф за повторы")
    parser.add_argument("--ngram", type=int, default=d.no_repeat_ngram, help="запрет повтора n-грамм")
    parser.add_argument("--plain", action="store_true", help="выключить эвристики")
    parser.add_argument("--stop", default="paragraph", choices=STOP_MODES)
    parser.add_argument("--len", type=int, default=200, dest="length", help="максимум токенов в ответе")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--prompt", default=None, help="одно продолжение без интерактивного режима")
    args = parser.parse_args()

    args.ckpt = args.ckpt or latest_checkpoint()
    if not args.ckpt or not os.path.exists(args.ckpt):
        sys.exit(f"Не найдена обученная модель ({args.ckpt or 'папка checkpoints/ пуста'}). "
                 "Сначала обучите: python train.py --arch diff --preset small --data data/tinystories.txt")

    for stream in (sys.stdout, sys.stdin):          # кириллица в консоли Windows
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(args.ckpt, device)
    tok = load_tokenizer()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Модель {args.ckpt}: {n_params:.1f}M параметров, шаг {ckpt.get('step')}, "
          f"val loss {ckpt.get('val_loss', float('nan')):.3f}, данные {ckpt.get('data')}, {device}")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    cfg = SamplingConfig(args.temp, args.top_k, args.top_p, args.rep, args.ngram)
    use_heur = not args.plain
    stop, length = args.stop, args.length

    if args.prompt is not None:
        complete(model, tok, args.prompt, length, cfg=cfg if use_heur else cfg.plain(), stop=stop)
        return

    print(HELP)
    compare_next = False
    while True:
        active = cfg if use_heur else cfg.plain()
        mode = "эвристики" if use_heur else "без эвристик"
        try:
            line = input(f"\n[{mode}: {active.describe()} stop={stop} len={length}] > ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line.strip() or line.strip() in ("/exit", "/quit"):
            break
        if line.startswith("/"):
            cmd, _, value = line.strip().partition(" ")
            try:
                if cmd == "/temp":
                    cfg = replace(cfg, temperature=float(value))
                elif cmd == "/topk":
                    cfg = replace(cfg, top_k=int(value))
                elif cmd == "/topp":
                    cfg = replace(cfg, top_p=float(value))
                elif cmd == "/rep":
                    cfg = replace(cfg, repetition_penalty=float(value))
                elif cmd == "/ngram":
                    cfg = replace(cfg, no_repeat_ngram=int(value))
                elif cmd == "/stop" and value in STOP_MODES:
                    stop = value
                elif cmd == "/plain":
                    use_heur = False
                elif cmd == "/heur":
                    use_heur = True
                elif cmd == "/compare":
                    compare_next = True
                    print("Следующая фраза будет продолжена двумя способами.")
                elif cmd == "/len":
                    length = int(value)
                elif cmd == "/seed":
                    torch.manual_seed(int(value))
                else:
                    print(HELP)
            except ValueError:
                print(f"Не понял значение: {line}")
            continue
        prompt = line.replace("\\n", "\n")
        try:
            if compare_next:
                compare_next = False
                seed = int(torch.randint(10**6, (1,)))
                for title, c in (("без эвристик", cfg.plain()), ("с эвристиками", cfg)):
                    print(f"--- {title} ({c.describe()}) ---")
                    torch.manual_seed(seed)
                    complete(model, tok, prompt, length, cfg=c, stop=stop)
            else:
                complete(model, tok, prompt, length, cfg=active, stop=stop)
        except KeyboardInterrupt:
            print("\n(генерация прервана)")


if __name__ == "__main__":
    main()

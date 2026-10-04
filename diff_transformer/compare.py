"""
Сравнение обученных моделей по методике статьи «Differential Transformer» (Ye et al., ICLR 2025,
arXiv:2410.05258) и с её числами.

  - табл. 6: loss на валидации и его разбиение на AR-Hit / Others (по Zoology, Arora et al. 2023):
      AR-Hit — токены, которые завершают n-грамму, уже встречавшуюся раньше в контексте
               (проверка «вспоминания» контекста), Others — все остальные токены;
  - табл. 5: выбросы активаций — самые большие по модулю логиты внимания и скрытые состояния;
  - дополнительно: эвристические метрики генерации (metrics.py) с эвристиками сэмплирования.

    python compare.py                                     # все модели из checkpoints/ablation/
    python compare.py checkpoints/ablation/transformer.pt checkpoints/ablation/diff.pt

Отчёт сохраняется в results/comparison.md и results/comparison.json.
"""
import argparse
import glob
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

from diff_transformer import load_checkpoint, load_tokenizer
from chat import generate_stream
from evaluate import pick_prompts
from metrics import build_vocab, summarize
from sampling import SamplingConfig

# ---- числа из статьи (модели 1.4B, 40K шагов; табл. 6) ----
PAPER_TABLE6 = {
    # имя: (описание, #heads, d, GN, valid, AR-Hit, Others)
    "transformer":         ("Transformer",               16, 128, False, 3.087, 0.898, 3.272),
    "transformer_8h":      ("Transformer (вдвое меньше голов)", 8, 256, False, 3.088, 0.899, 3.273),
    "transformer_8h_gn":   ("  + GroupNorm",              8, 256, True,  3.086, 0.899, 3.271),
    "diff":                ("DIFF Transformer",           8, 128, True,  3.062, 0.880, 3.247),
    "diff_no_gn":          ("  - GroupNorm",              8, 128, False, 3.122, 0.911, 3.309),
    "diff_lambda08":       ("  with λ_init = 0.8",        8, 128, True,  3.065, 0.883, 3.250),
    "diff_lambda05":       ("  with λ_init = 0.5",        8, 128, True,  3.066, 0.882, 3.251),
}
# табл. 5 (модели 3B): top-1, top-2, top-3, top-10, top-100, median
PAPER_TABLE5 = {
    ("Transformer", "attention logits"): (318.0, 308.2, 304.9, 284.7, 251.5, 5.4),
    ("DIFF", "attention logits"): (38.8, 38.8, 37.3, 32.0, 27.4, 3.3),
    ("Transformer", "hidden states"): (3608.6, 3607.4, 3603.6, 3552.1, 2448.2, 0.6),
    ("DIFF", "hidden states"): (1688.2, 1672.5, 1672.1, 1624.3, 740.9, 1.2),
}
TOPS = (1, 2, 3, 10, 100)


@torch.no_grad()
def split_loss(model, tokens, max_tokens, n=2, batch=16):
    """Средний loss на всех токенах, на AR-Hit и на Others; окна длины block_size без перекрытия."""
    T = model.cfg.block_size
    device = next(model.parameters()).device
    tokens = tokens[: max_tokens + 1]
    n_win = (len(tokens) - 1) // T
    windows = torch.stack([tokens[i * T: i * T + T + 1] for i in range(n_win)])
    sums = {"all": 0.0, "hit": 0.0, "other": 0.0}
    counts = {"all": 0, "hit": 0, "other": 0}
    for b in range(0, n_win, batch):
        w = windows[b:b + batch]
        logits, _ = model(w[:, :-1].to(device))
        losses = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)),
                                 w[:, 1:].reshape(-1).to(device), reduction="none").view(w.shape[0], T).cpu()
        hits = torch.tensor([ar_hit_mask(row.tolist(), n) for row in w], dtype=torch.bool)
        sums["all"] += losses.sum().item()
        counts["all"] += losses.numel()
        sums["hit"] += losses[hits].sum().item()
        counts["hit"] += int(hits.sum())
        sums["other"] += losses[~hits].sum().item()
        counts["other"] += int((~hits).sum())
    res = {k: sums[k] / max(1, counts[k]) for k in sums}
    res["hit_share"] = counts["hit"] / max(1, counts["all"])
    return res


def ar_hit_mask(window, n=2):
    """window — T+1 токенов. Для каждого предсказываемого токена window[j] (j = 1..T): True, если
    n-грамма window[j-n+1..j] уже встречалась раньше в этом окне (её последний токен «вспоминается»)."""
    seen, mask = set(), []
    for j in range(1, len(window)):
        if j - n + 1 >= 0:
            gram = tuple(window[j - n + 1: j + 1])
            mask.append(gram in seen)
            seen.add(gram)
        else:
            mask.append(False)
    return mask


@torch.no_grad()
def activation_outliers(model, tokens, n_windows=48, batch=4):
    """Табл. 5: top-k по модулю логитов внимания (до softmax) и скрытых состояний (выходов слоёв)."""
    T = model.cfg.block_size
    device = next(model.parameters()).device
    attn_modules = [b.attn for b in model.blocks]
    for m in attn_modules:
        m.record = []
    hidden = []

    def hook(_module, _inp, out):
        a = out.detach().float().abs().flatten()
        hidden.append((a.topk(min(100, a.numel())).values.cpu(), a.median().item()))
    handles = [b.register_forward_hook(hook) for b in model.blocks]
    n_windows = min(n_windows, (len(tokens) - 1) // T)
    windows = torch.stack([tokens[i * T: i * T + T] for i in range(n_windows)])
    for b in range(0, n_windows, batch):
        model(windows[b:b + batch].to(device))
    result = {}
    for name, records in (("attention logits", [r for m in attn_modules for r in m.record]),
                          ("hidden states", hidden)):
        tops = torch.cat([t for t, _ in records]).sort(descending=True).values
        median = float(torch.tensor([m for _, m in records]).median())
        result[name] = tuple(float(tops[k - 1]) for k in TOPS) + (median,)
    for m in attn_modules:
        m.record = None
    for h in handles:
        h.remove()
    return result


def model_label(ckpt):
    cfg = ckpt["config"]
    if cfg.get("arch", "baseline") == "diff":
        heads, d = cfg["n_head"] // 2, cfg["d_model"] // cfg["n_head"]
        gn = True if cfg.get("head_norm") is None else cfg["head_norm"]
    else:
        heads, d, gn = cfg["n_head"], cfg["d_model"] // cfg["n_head"], bool(cfg.get("head_norm"))
    return heads, d, gn


def fmt(x, nd=3):
    return f"{x:.{nd}f}"


def main():
    parser = argparse.ArgumentParser(description="Сравнение моделей по методике статьи Diff Transformer")
    parser.add_argument("ckpts", nargs="*", help="чекпоинты (по умолчанию checkpoints/ablation/*.pt)")
    parser.add_argument("--data", default=None, help="по умолчанию файл, на котором училась первая модель")
    parser.add_argument("--eval_tokens", type=int, default=300_000, help="токенов валидации для loss")
    parser.add_argument("--ngram", type=int, default=2, help="n для AR-Hit")
    parser.add_argument("--samples", type=int, default=30, help="фраз для метрик генерации (0 — пропустить)")
    parser.add_argument("--gen_len", type=int, default=120)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    paths = args.ckpts or sorted(glob.glob(os.path.join("checkpoints", "ablation", "*.pt")),
                                 key=lambda p: list(PAPER_TABLE6).index(os.path.basename(p)[:-3])
                                 if os.path.basename(p)[:-3] in PAPER_TABLE6 else 99)
    if not paths:
        sys.exit("Нет моделей в checkpoints/ablation/. Сначала запустите: python run_experiments.py")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()

    first = torch.load(paths[0], map_location="cpu", weights_only=True)
    data_path = args.data or first.get("data")
    cache = f"{data_path}.mistral.pt"
    if not os.path.exists(cache):
        sys.exit(f"Нет токенов {cache}. Сначала обучите модель на {data_path}.")
    tokens = torch.load(cache)
    val = tokens[int(0.9 * len(tokens)):]
    text = open(data_path, encoding="utf-8").read()
    split = int(0.9 * len(text))
    vocab = build_vocab(text[max(0, split - 20_000_000):split])
    prompts = pick_prompts(text[split:], args.samples, 6, seed=0) if args.samples else []
    print(f"Данные: {data_path}, валидация {len(val):,} токенов (берём {args.eval_tokens:,}), {device}\n")

    rows = []
    for path in paths:
        model, ckpt = load_checkpoint(path, device)
        name = os.path.basename(path)[:-3]
        heads, d, gn = model_label(ckpt)
        cfg = ckpt["config"]
        print(f"== {name}: {cfg.get('arch', 'baseline')}, голов {heads}, d={d}, GN={gn}, "
              f"λ_init={cfg.get('lambda_init') or 'по слоям'}, шаг {ckpt.get('step')}")
        loss = split_loss(model, val, args.eval_tokens, n=args.ngram)
        print(f"   valid {loss['all']:.3f} (ppl {math.exp(loss['all']):.2f}) | AR-Hit {loss['hit']:.3f} "
              f"| Others {loss['other']:.3f} | доля AR-Hit {loss['hit_share']:.1%}")
        acts = activation_outliers(model, val)
        for k, v in acts.items():
            print(f"   {k:16s} top-1 {v[0]:.1f} | top-10 {v[3]:.1f} | top-100 {v[4]:.1f} | median {v[5]:.2f}")
        gen = {}
        if prompts:
            outs, cfg_s = [], SamplingConfig()
            for i, p in enumerate(prompts):
                torch.manual_seed(i)
                ids = tok.encode(p, add_special_tokens=False)
                new = list(generate_stream(model, ids, args.gen_len, eos_id=tok.eos_token_id, cfg=cfg_s))
                full = tok.decode(ids + new, skip_special_tokens=True)
                outs.append(full[len(tok.decode(ids, skip_special_tokens=True)):])
            gen = summarize(outs, vocab)
            gen["example"] = prompts[0] + outs[0]
            print(f"   генерация: distinct-2 {gen['distinct-2']:.3f}, repeat-4 {gen['repeat-4']:.3f}, "
                  f"known_words {gen['known_words']:.3f}")
        rows.append(dict(name=name, arch=cfg.get("arch", "baseline"), heads=heads, d=d, gn=gn,
                         lambda_init=cfg.get("lambda_init"), step=ckpt.get("step"),
                         params=sum(p.numel() for p in model.parameters()),
                         loss=loss, acts=acts, gen=gen))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    report = build_report(rows, data_path, args)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "comparison.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(args.out_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n" + report)
    print(f"Отчёт: {os.path.join(args.out_dir, 'comparison.md')}")


def build_report(rows, data_path, args):
    base = next((r for r in rows if r["name"] == "transformer"), rows[0])
    L = ["# Diff Transformer vs Transformer: наши результаты и статья\n",
         f"Данные: `{data_path}`, loss на {args.eval_tokens:,} токенах валидации, AR-Hit по {args.ngram}-граммам.\n",
         "## Табл. 6 — loss на валидации (меньше — лучше)\n",
         "| Модель | #heads | d | GN | Params | Valid | AR-Hit | Others | Δ Valid | Статья: Valid | AR-Hit | Others | Δ Valid |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    p_base = PAPER_TABLE6["transformer"][4]
    for r in rows:
        lo = r["loss"]
        p = PAPER_TABLE6.get(r["name"])
        paper = (f"{p[4]:.3f} | {p[5]:.3f} | {p[6]:.3f} | {p[4] - p_base:+.3f}" if p else "— | — | — | —")
        title = p[0] if p else r["name"]
        L.append(f"| {title} | {r['heads']} | {r['d']} | {'✓' if r['gn'] else '✗'} | {r['params'] / 1e6:.1f}M "
                 f"| {fmt(lo['all'])} | {fmt(lo['hit'])} | {fmt(lo['other'])} | {lo['all'] - base['loss']['all']:+.3f} "
                 f"| {paper} |")
    L.append(f"\nДоля AR-Hit токенов у нас: {base['loss']['hit_share']:.1%}. "
             "Статья: модели 1.4B, 40K шагов, свои данные и токенизатор — абсолютные значения "
             "loss несравнимы, сравниваем знак и порядок Δ относительно Transformer.\n")

    L += ["## Табл. 5 — выбросы активаций (меньше top — меньше выбросов)\n",
          "| Модель | Активации | Top-1 | Top-2 | Top-3 | Top-10 | Top-100 | Median |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["name"] not in ("transformer", "diff") and len(rows) > 2:
            continue
        for kind, v in r["acts"].items():
            L.append(f"| {r['name']} (наш) | {kind} | " + " | ".join(f"{x:.1f}" for x in v[:5]) + f" | {v[5]:.2f} |")
    for (m, kind), v in PAPER_TABLE5.items():
        L.append(f"| {m} (статья, 3B) | {kind} | " + " | ".join(f"{x:.1f}" for x in v[:5]) + f" | {v[5]:.1f} |")

    if rows[0]["gen"]:
        L += ["\n## Генерация с эвристиками (metrics.py)\n",
              "| Модель | distinct-1 | distinct-2 | repeat-4 ↓ | known_words | clean_end |",
              "|---|---|---|---|---|---|"]
        for r in rows:
            g = r["gen"]
            L.append(f"| {r['name']} | {fmt(g['distinct-1'])} | {fmt(g['distinct-2'])} | {fmt(g['repeat-4'])} "
                     f"| {fmt(g['known_words'])} | {fmt(g['clean_end'])} |")

    L.append("\n## Выводы (автоматически)\n")
    by = {r["name"]: r for r in rows}
    checks = []
    if "transformer" in by and "diff" in by:
        dv = by["diff"]["loss"]["all"] - by["transformer"]["loss"]["all"]
        dh = by["diff"]["loss"]["hit"] - by["transformer"]["loss"]["hit"]
        checks.append(f"- Diff vs Transformer: Δ valid {dv:+.3f} (статья −0.025), Δ AR-Hit {dh:+.3f} (статья −0.018) — "
                      + ("совпадает со статьёй: Diff лучше." if dv < 0 else "не совпадает: Diff не лучше на нашем масштабе."))
        a_t, a_d = by["transformer"]["acts"]["attention logits"][0], by["diff"]["acts"]["attention logits"][0]
        checks.append(f"- Выбросы логитов внимания: top-1 {a_t:.1f} → {a_d:.1f} "
                      + ("(меньше у Diff, как в статье)." if a_d < a_t else "(у Diff не меньше, в отличие от статьи)."))
    if "diff" in by and "diff_no_gn" in by:
        d = by["diff_no_gn"]["loss"]["all"] - by["diff"]["loss"]["all"]
        checks.append(f"- Без GroupNorm Diff хуже на {d:+.3f} (статья +0.060) — "
                      + ("подтверждается." if d > 0 else "не подтверждается."))
    lam = [by[k]["loss"]["all"] for k in ("diff", "diff_lambda08", "diff_lambda05") if k in by]
    if len(lam) > 1:
        checks.append(f"- Разброс loss при разных λ_init: {max(lam) - min(lam):.3f} (статья 0.004) — "
                      + ("модель устойчива к выбору λ_init, как в статье." if max(lam) - min(lam) < 0.03
                         else "заметная зависимость от λ_init."))
    if "transformer_8h" in by and "transformer_8h_gn" in by:
        d = by["transformer_8h_gn"]["loss"]["all"] - by["transformer_8h"]["loss"]["all"]
        checks.append(f"- GroupNorm в обычном Transformer: Δ {d:+.3f} (статья −0.002, почти без эффекта).")
    L += checks or ["- Для выводов нужны хотя бы модели transformer и diff."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()

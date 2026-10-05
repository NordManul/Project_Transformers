"""
Сравнение трёх трансформеров на одинаковых данных:
    gpt   — базовый трансформер (baseline_gpt.py)
    ngpt  — нормализованный трансформер (ngpt.py, arXiv:2410.01131)
    diff  — Differential Transformer (diff_transformer.py, arXiv:2410.05258)

Команды:
    python compare.py train --model gpt  --preset small --data data/tinystories.txt --max_iters 5000
    python compare.py train --model ngpt --preset small --data data/tinystories.txt --max_iters 5000
    python compare.py train --model diff --preset small --data data/tinystories.txt --max_iters 5000
    python compare.py plot  runs/gpt runs/ngpt runs/diff          # графики -> runs/plots/
    python compare.py eval  runs/gpt --data data/test.txt          # loss на отложенной выборке

Что делает сравнение честным:
  * одинаковые батчи: у данных свой генератор (--data_seed), порядок не зависит от модели;
  * одинаковая валидация: фиксированный набор окон из отложенной выборки на каждом eval
    и полный проход по ней в конце обучения;
  * рецепты из статей: gpt и diff — AdamW, weight decay 0.1, warmup;
    ngpt — Adam без weight decay и без warmup, нормировка весов после каждого шага.

Что пишется в runs/<имя>/:
  config.json    аргументы, конфигурация модели, число параметров
  metrics.jsonl  по строке на событие: train (loss, lr, норма градиента, скорость),
                 val (loss на фиксированном наборе), stats (активации и градиенты по слоям,
                 λ у diff, α и s_z у ngpt), final (полный проход по отложенной выборке)
  model.pt       веса в конце обучения
"""
import argparse
import importlib
import json
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch

# ---- модули проекта: baseline_gpt.py и ngpt.py в корне, diff_transformer.py рядом или в подпапке ----
HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))


def _import_diff():
    try:
        return importlib.import_module("diff_transformer")
    except ModuleNotFoundError as e:
        if e.name != "diff_transformer":
            raise
        for candidate in sorted(HERE.glob("*/diff_transformer.py")):
            sys.path.append(str(candidate.parent))
            return importlib.import_module("diff_transformer")
        raise


dt = _import_diff()
import baseline_gpt as bg  # noqa: E402
import ngpt as ng          # noqa: E402

MODELS = ("gpt", "ngpt", "diff")
PRESETS = dt.PRESETS       # tiny, small, 0.5B, 1B — общие для всех трёх моделей


# ============================ Модели ============================
def build_model(name, vocab_size, dims, scale_ref_dim=0):
    """Возвращает (модель, функция создания оптимизатора, функция после шага, warmup по умолчанию)."""
    if name == "gpt":
        model = bg.GPT(bg.ModelConfig(vocab_size=vocab_size, **dims))
        return model, bg.make_optimizer, (lambda: None), 200
    if name == "diff":
        model = dt.DiffGPT(dt.DiffConfig(vocab_size=vocab_size, arch="diff", **dims))
        return model, bg.make_optimizer, (lambda: None), 200
    if name == "ngpt":
        model = ng.NGPT(ng.NGPTConfig(vocab_size=vocab_size, scale_ref_dim=scale_ref_dim, **dims))
        return model, ng.make_ngpt_optimizer, model.normalize_weights, 0
    raise ValueError(f"неизвестная модель {name!r}")


def model_from_saved(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    cfg = dict(saved["config"])
    name = saved["model_name"]
    vocab = cfg.pop("vocab_size")
    if name == "diff":
        cfg.pop("arch", None)
    scale_ref_dim = cfg.pop("scale_ref_dim", 0)
    model, *_ = build_model(name, vocab, cfg, scale_ref_dim)
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), name


# ============================ Мониторинг активаций и градиентов ============================
class Monitor:
    """Forward-хуки на каждом блоке и его подслоях attn и mlp (имена одинаковые у всех трёх моделей).
    Считает статистику, только когда active=True, чтобы не замедлять обычные шаги.

    Для каждого выхода записываются:
      norm    — средняя L2-норма вектора токена (у nGPT выход блока всегда 1);
      rms     — среднеквадратичное значение элементов;
      absmax  — максимальный модуль элемента (выбросы активаций, табл. 5 статьи про Diff);
      outlier — absmax / rms: во сколько раз выброс больше типичного значения."""

    def __init__(self, model):
        self.active = False
        self.acts = {}
        self.handles = []
        for i, block in enumerate(model.blocks):
            for part, module in (("block", block), ("attn", block.attn), ("mlp", block.mlp)):
                self.handles.append(module.register_forward_hook(self._hook(part, i)))

    def _hook(self, part, layer):
        def fn(module, inputs, output):
            if not self.active:
                return
            x = output.detach().float()
            rms = x.pow(2).mean().sqrt()
            absmax = x.abs().max()
            self.acts.setdefault(part, {})[layer] = dict(
                norm=x.norm(dim=-1).mean().item(), rms=rms.item(), absmax=absmax.item(),
                outlier=(absmax / rms.clamp_min(1e-12)).item())
        return fn

    def take_activations(self):
        out = {part: [layers[i] for i in sorted(layers)] for part, layers in self.acts.items()}
        self.acts = {}
        return out


def grad_norm(params):
    total = sum(p.grad.detach().float().pow(2).sum() for p in params if p.grad is not None)
    return math.sqrt(float(total)) if torch.is_tensor(total) else 0.0


def gradient_stats(model):
    return dict(
        layers=[grad_norm(b.parameters()) for b in model.blocks],
        attn=[grad_norm(b.attn.parameters()) for b in model.blocks],
        mlp=[grad_norm(b.mlp.parameters()) for b in model.blocks],
        emb_in=grad_norm([model.emb_in.weight]),
        emb_out=grad_norm([model.emb_out.weight]),
    )


@torch.no_grad()
def special_stats(model, name):
    """Выученные величины, ради которых придуманы архитектуры."""
    if name == "diff":
        return dict(lambda_=[b.attn.lam().item() for b in model.blocks])
    if name == "ngpt":
        return dict(alpha_attn=[b.alpha_a().abs().mean().item() for b in model.blocks],
                    alpha_mlp=[b.alpha_m().abs().mean().item() for b in model.blocks],
                    s_qk=[b.attn.s_qk().abs().mean().item() for b in model.blocks],
                    s_z=model.s_z().mean().item())
    return {}


# ============================ Данные и оценка ============================
def fixed_windows(data, T, n, seed):
    """n случайных, но фиксированных окон длины T+1 — один и тот же набор для всех моделей."""
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(data) - T, (n,), generator=g)
    return torch.stack([data[i:i + T + 1] for i in ix])


@torch.no_grad()
def mean_loss(model, windows, batch_size, device, autocast):
    """Средний loss на токен по набору окон (все окна одной длины)."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(windows), batch_size):
        w = windows[i:i + batch_size].to(device)
        with autocast:
            _, loss = model(w[:, :-1], w[:, 1:])
        total += loss.item() * len(w)
        count += len(w)
    model.train()
    return total / count


@torch.no_grad()
def full_eval(model, data, T, batch_size, device, autocast, max_tokens=0):
    """Полный проход по отложенной выборке непересекающимися окнами."""
    n = (len(data) - 1) // T
    if max_tokens:
        n = min(n, max(1, max_tokens // T))
    windows = torch.stack([data[i * T:i * T + T + 1] for i in range(n)])
    loss = mean_loss(model, windows, batch_size, device, autocast)
    return dict(val_loss=loss, perplexity=math.exp(loss), tokens=n * T)


def setup_device(requested, threads=0):
    if threads > 0:
        torch.set_num_threads(threads)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = requested == "cuda" and torch.cuda.is_bf16_supported()
    if requested == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    return requested, use_bf16, autocast


def load_splits(args, tok):
    data = bg.load_training_tokens(args.data, tok)
    if args.val_data:
        return data, bg.load_training_tokens(args.val_data, tok)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


# ============================ Обучение ============================
def cmd_train(args):
    device, use_bf16, autocast = setup_device(args.device, args.threads)
    run_dir = Path(args.out) / (args.name or args.model)
    if (run_dir / "metrics.jsonl").exists() and not args.overwrite:
        sys.exit(f"{run_dir} уже существует: задайте другое --name или добавьте --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    tok = bg.load_tokenizer()
    train_data, val_data = load_splits(args, tok)
    dims = dict(PRESETS[args.preset])
    for key in ("n_layer", "n_head", "d_model", "block_size"):
        if getattr(args, key) is not None:
            dims[key] = getattr(args, key)
    T = dims["block_size"]
    for part, d in (("train", train_data), ("val", val_data)):
        if len(d) < T + 1:
            sys.exit(f"мало данных в {part}: {len(d)} токенов, нужно больше {T}")

    torch.manual_seed(args.seed)
    model, make_opt, after_step, default_warmup = build_model(args.model, tok.vocab_size, dims, args.scale_ref_dim)
    model.to(device)
    optimizer = make_opt(model, args.lr)
    warmup = default_warmup if args.warmup is None else args.warmup
    monitor = Monitor(model)
    n_params = sum(p.numel() for p in model.parameters())

    data_gen = torch.Generator().manual_seed(args.data_seed)
    val_windows = fixed_windows(val_data, T, args.eval_batches * args.batch_size, args.data_seed + 1)

    def train_batch():
        ix = torch.randint(len(train_data) - T, (args.batch_size,), generator=data_gen)
        w = torch.stack([train_data[i:i + T + 1] for i in ix]).to(device)
        return w[:, :-1], w[:, 1:]

    config = dict(model_name=args.model, n_params=n_params, device=device, bf16=use_bf16,
                  warmup=warmup, tokens_per_step=args.batch_size * T,
                  model_config=asdict(model.cfg), args=vars(args))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    log = (run_dir / "metrics.jsonl").open("w", encoding="utf-8")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()

    print(f"{args.model.upper()} | {n_params / 1e6:.2f}M параметров | {dims} | устройство {device}, "
          f"bf16={use_bf16} | lr {args.lr}, warmup {warmup} | лог: {run_dir}", flush=True)

    began = time.monotonic()
    loss_sum, loss_count, tokens = 0.0, 0, 0
    val = mean_loss(model, val_windows, args.batch_size, device, autocast)
    write(dict(type="val", step=0, tokens=0, val_loss=val))
    try:
        for step in range(1, args.max_iters + 1):
            lr = bg.lr_at(step - 1, args.lr, warmup, args.max_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr
            collect = step % args.stats_every == 0 or step == 1
            monitor.active = collect
            x, y = train_batch()
            with autocast:
                _, loss = model(x, y)
            monitor.active = False
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss = {loss.item()} на шаге {step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if collect:
                grads = gradient_stats(model)
            total_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")).item()
            optimizer.step()
            after_step()
            tokens += x.numel()
            loss_sum += loss.item()
            loss_count += 1

            if collect:
                write(dict(type="stats", step=step, tokens=tokens, activations=monitor.take_activations(),
                           grads=grads, special=special_stats(model, args.model)))
            if step % args.log_every == 0 or step == args.max_iters:
                elapsed = time.monotonic() - began
                write(dict(type="train", step=step, tokens=tokens, loss=loss_sum / loss_count, lr=lr,
                           grad_norm=total_grad, elapsed=elapsed, tokens_per_sec=tokens / elapsed))
                loss_sum, loss_count = 0.0, 0
            if step % args.eval_every == 0 or step == args.max_iters:
                val = mean_loss(model, val_windows, args.batch_size, device, autocast)
                write(dict(type="val", step=step, tokens=tokens, val_loss=val))
                print(f"шаг {step:6d} | lr {lr:.2e} | train {loss.item():.3f} | val {val:.3f} | "
                      f"grad {total_grad:.2f} | {time.monotonic() - began:.0f} с", flush=True)
    except KeyboardInterrupt:
        print("\nОстановлено вручную — сохраняю то, что есть.", flush=True)

    torch.save(dict(model_name=args.model, config=asdict(model.cfg), model=model.state_dict(),
                    step=step, tokens=tokens), run_dir / "model.pt")
    final = full_eval(model, val_data, T, args.batch_size, device, autocast, args.final_eval_tokens)
    final.update(type="final", step=step, tokens_trained=tokens, elapsed=time.monotonic() - began)
    write(final)
    log.close()
    print(f"Итог на отложенной выборке ({final['tokens']:,} токенов): loss {final['val_loss']:.4f}, "
          f"perplexity {final['perplexity']:.2f}", flush=True)


def cmd_eval(args):
    device, _, autocast = setup_device(args.device, args.threads)
    model, name = model_from_saved(Path(args.run) / "model.pt", device)
    tok = bg.load_tokenizer()
    data = bg.load_training_tokens(args.data, tok)
    result = full_eval(model, data, model.cfg.block_size, args.batch_size, device, autocast, args.max_tokens)
    print(f"{name} ({args.run}) на {args.data}: loss {result['val_loss']:.4f}, "
          f"perplexity {result['perplexity']:.2f}, токенов {result['tokens']:,}")
    with (Path(args.run) / "eval.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dict(data=args.data, **result)) + "\n")


# ============================ Графики ============================
COLORS = {"gpt": "#4C72B0", "ngpt": "#DD8452", "diff": "#55A868"}
STYLES = {"gpt": "-", "ngpt": "--", "diff": "-."}   # совпадающие кривые не прячутся друг под другом


def read_run(path):
    path = Path(path)
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (path / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line]
    by_type = {}
    for r in records:
        by_type.setdefault(r["type"], []).append(r)
    return dict(name=path.name, model=config["model_name"], config=config, **by_type)


def smooth(values, frac):
    """Скользящее среднее по окну в долю frac от числа точек (без запаздывания EMA на коротких прогонах)."""
    w = max(1, int(len(values) * frac))
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= w:
            acc -= values[i - w]
        out.append(acc / min(i + 1, w))
    return out


def cmd_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [read_run(p) for p in args.runs]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    xkey = "tokens" if args.x == "tokens" else "step"
    xlabel = "токены" if xkey == "tokens" else "шаг"
    color = lambda r, i: COLORS.get(r["model"], f"C{i}")   # noqa: E731
    style = lambda r: STYLES.get(r["model"], "-")          # noqa: E731
    from matplotlib.ticker import MaxNLocator
    label = lambda r: f"{r['name']} ({r['config']['n_params'] / 1e6:.1f}M)"   # noqa: E731
    saved = []

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / name, dpi=130)
        plt.close(fig)
        saved.append(out / name)

    # 1. Loss: train (сглаженный) и val на фиксированном наборе
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, r in enumerate(runs):
        tr = r.get("train", [])
        if tr:
            xs, ys = [t[xkey] for t in tr], [t["loss"] for t in tr]
            axes[0].plot(xs, ys, color=color(r, i), alpha=0.2, lw=0.8)
            axes[0].plot(xs, smooth(ys, args.smooth), color=color(r, i), ls=style(r), lw=1.8, label=label(r))
        va = r.get("val", [])
        if va:
            axes[1].plot([v[xkey] for v in va], [v["val_loss"] for v in va], marker="o", ms=3,
                         ls=style(r), color=color(r, i), label=label(r))
    axes[0].set_title("Train loss (линия — скользящее среднее)")
    axes[1].set_title("Validation loss (фиксированный набор окон)")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
        ax.legend()
    # масштаб val без начальных точек (loss ~ ln V у случайной модели), но так, чтобы были видны все модели
    last_step = max((v["step"] for r in runs for v in r.get("val", [])), default=0)
    late = [v["val_loss"] for r in runs for v in r.get("val", []) if v["step"] >= 0.1 * last_step and v["step"] > 0]
    if late:
        span = max(late) - min(late)
        axes[1].set_ylim(min(late) - 0.05 - 0.05 * span, max(late) + 0.05 + 0.1 * span)
    save(fig, "loss.png")

    # 2. Динамика: норма градиента и learning rate
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, r in enumerate(runs):
        tr = r.get("train", [])
        xs = [t[xkey] for t in tr]
        axes[0].plot(xs, smooth([t["grad_norm"] for t in tr], args.smooth), color=color(r, i), ls=style(r),
                     label=label(r))
        axes[1].plot(xs, [t["lr"] for t in tr], color=color(r, i), ls=style(r), label=label(r))
    axes[0].set_yscale("log")
    axes[0].set_title("Общая норма градиента до клиппинга (скользящее среднее)")
    axes[1].set_title("Learning rate")
    for ax in axes:
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
        ax.legend()
    save(fig, "grad_norm.png")

    # 3. Активации и градиенты по слоям во времени: строки — величины, столбцы — модели
    rows = [("activations", "block", "norm", "Норма выхода блока (residual stream)", False),
            ("activations", "block", "outlier", "Выбросы: max|a| / rms выхода блока", False),
            ("activations", "attn", "absmax", "max|a| выхода attention", True),
            ("activations", "mlp", "absmax", "max|a| выхода MLP", True),
            ("grads", "layers", None, "Норма градиента слоя", True)]
    fig, axes = plt.subplots(len(rows), len(runs), figsize=(4.6 * len(runs), 2.8 * len(rows)), squeeze=False)
    for col, r in enumerate(runs):
        stats = r.get("stats", [])
        xs = [s[xkey] for s in stats]
        n_layer = len(stats[0]["grads"]["layers"]) if stats else 0
        cmap = plt.get_cmap("viridis")
        for row, (group, part, field, title, logy) in enumerate(rows):
            ax = axes[row][col]
            for layer in range(n_layer):
                if group == "grads":
                    ys = [s["grads"][part][layer] for s in stats]
                else:
                    ys = [s["activations"][part][layer][field] for s in stats]
                ax.plot(xs, ys, color=cmap(layer / max(1, n_layer - 1)), lw=1.2,
                        label=f"слой {layer}" if layer in (0, n_layer - 1) else None)
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"{r['name']}: {title}", fontsize=9)
            ax.grid(alpha=0.3)
            ax.set_xlabel(xlabel, fontsize=8)
            if row == 0:
                ax.legend(fontsize=7)
    save(fig, "layers_over_time.png")

    # 4. Профиль по глубине в конце обучения: модели на одном графике
    profiles = [("activations", "block", "norm", "Норма выхода блока"),
                ("activations", "block", "outlier", "max|a| / rms выхода блока"),
                ("grads", "layers", None, "Норма градиента слоя")]
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4))
    for i, r in enumerate(runs):
        if not r.get("stats"):
            continue
        last = r["stats"][-1]
        for ax, (group, part, field, title) in zip(axes, profiles):
            ys = last[group][part] if group == "grads" else [a[field] for a in last[group][part]]
            ax.plot(range(len(ys)), ys, marker="o", ls=style(r), color=color(r, i), label=label(r))
            ax.set_title(f"{title} (шаг {last['step']})")
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("слой")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    save(fig, "layer_profile.png")

    # 5. Выученные параметры архитектур: λ у Diff, α и s_z у nGPT
    special = [(r, i) for i, r in enumerate(runs) if r.get("stats") and r["stats"][-1].get("special")]
    panels = []
    for r, i in special:
        sp = r["stats"][0]["special"]
        if "lambda_" in sp:
            panels.append((r, "lambda_", "λ по слоям (Diff)"))
        if "alpha_attn" in sp:
            panels += [(r, "alpha_attn", "|α_A| по слоям (nGPT)"), (r, "alpha_mlp", "|α_M| по слоям (nGPT)"),
                       (r, "s_z", "s_z (масштаб логитов nGPT)")]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.8), squeeze=False)
        for ax, (r, key, title) in zip(axes[0], panels):
            stats = r["stats"]
            xs = [s[xkey] for s in stats]
            values = [s["special"][key] for s in stats]
            if isinstance(values[0], list):
                n = len(values[0])
                cmap = plt.get_cmap("viridis")
                for layer in range(n):
                    ax.plot(xs, [v[layer] for v in values], color=cmap(layer / max(1, n - 1)),
                            label=f"слой {layer}" if layer in (0, n - 1) else None)
                ax.legend(fontsize=7)
            else:
                ax.plot(xs, values, color=COLORS.get(r["model"]))
            ax.set_title(f"{r['name']}: {title}", fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.grid(alpha=0.3)
        save(fig, "architecture_params.png")

    # Итоговая таблица
    lines = ["| запуск | модель | параметры | шагов | токенов | лучший val (фикс. набор) | итоговый val loss | perplexity | время |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        fin = (r.get("final") or [{}])[-1]
        best = min((v["val_loss"] for v in r.get("val", [])), default=float("nan"))
        lines.append(f"| {r['name']} | {r['model']} | {r['config']['n_params'] / 1e6:.2f}M | {fin.get('step', '—')} | "
                     f"{fin.get('tokens_trained', 0):,} | {best:.4f} | {fin.get('val_loss', float('nan')):.4f} | "
                     f"{fin.get('perplexity', float('nan')):.2f} | {fin.get('elapsed', 0) / 60:.1f} мин |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nГрафики:\n" + "\n".join(f"  {p}" for p in saved + [out / "summary.md"]))


# ============================ CLI ============================
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Сравнение GPT, nGPT и Diff Transformer")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="обучить одну модель с логированием")
    t.add_argument("--model", required=True, choices=MODELS)
    t.add_argument("--name", help="имя запуска (папка в --out); по умолчанию имя модели")
    t.add_argument("--out", default="runs")
    t.add_argument("--overwrite", action="store_true")
    t.add_argument("--data", required=True)
    t.add_argument("--val_data", help="отдельная отложенная выборка; иначе последние 10%% --data")
    t.add_argument("--preset", default="small", choices=PRESETS.keys())
    for key in ("n_layer", "n_head", "d_model", "block_size"):
        t.add_argument(f"--{key}", type=int, help="переопределить значение пресета")
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--max_iters", type=int, default=5000)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, help="по умолчанию 200 для gpt/diff, 0 для ngpt")
    t.add_argument("--grad_clip", type=float, default=1.0, help="0 — без клиппинга")
    t.add_argument("--scale_ref_dim", type=int, default=0, help="только ngpt, см. ngpt.py")
    t.add_argument("--log_every", type=int, default=10)
    t.add_argument("--stats_every", type=int, default=50, help="как часто писать активации и градиенты по слоям")
    t.add_argument("--eval_every", type=int, default=250)
    t.add_argument("--eval_batches", type=int, default=20, help="размер фиксированного val-набора в батчах")
    t.add_argument("--final_eval_tokens", type=int, default=0, help="0 — вся отложенная выборка")
    t.add_argument("--seed", type=int, default=42, help="инициализация модели")
    t.add_argument("--data_seed", type=int, default=1234, help="порядок батчей и val-набор — одинаковы для всех моделей")
    t.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    t.add_argument("--threads", type=int, default=0, help="потоки CPU; 0 — по умолчанию torch")

    e = sub.add_parser("eval", help="loss обученной модели на отложенной выборке")
    e.add_argument("run")
    e.add_argument("--data", required=True)
    e.add_argument("--batch_size", type=int, default=32)
    e.add_argument("--max_tokens", type=int, default=0)
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.add_argument("--threads", type=int, default=0)

    p = sub.add_parser("plot", help="графики по нескольким запускам")
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="runs/plots")
    p.add_argument("--x", default="step", choices=["step", "tokens"])
    p.add_argument("--smooth", type=float, default=0.03,
                   help="окно скользящего среднего как доля длины прогона (0 — без сглаживания)")

    args = parser.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "plot": cmd_plot}[args.command](args)


if __name__ == "__main__":
    main()
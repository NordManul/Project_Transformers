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
  ckpt.pt        чекпойнт для продолжения (модель + оптимизатор + состояние данных);
                 удаляется, когда обучение закончено
  train.log      вывод обучения (пишет ngpt_vs_gpt.py)

Для Kaggle и долгих запусков:
  --data data/owt/train.bin --val_data data/owt/val.bin   токены uint16 с диска (tokenize_dataset.py),
                                                          в память не загружаются
  --micro_batch 16      батч --batch_size набирается из нескольких микробатчей (накопление градиента)
  --dtype auto          bf16 на A100/H100, fp16 + GradScaler на T4/P100/V100 (у них нет быстрого bf16)
  --resume              продолжить с ckpt.pt (те же батчи, тот же lr, лог дописывается)
  --deadline <unixtime> в этот момент сохранить чекпойнт и выйти с кодом 3 (до лимита сессии Kaggle)
"""
import argparse
import importlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# ---- модули проекта: baseline_gpt.py и ngpt.py в корне, diff_transformer.py рядом или в подпапке ----
HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))


def _import_diff():
    """diff_transformer.py рядом с compare.py или в подпапке. Если рядом лежит ПАПКА diff_transformer/,
    Python импортирует её как пустой пакет — тогда ищем сам файл в подпапках.
    Нужен только для --model diff: без него gpt и ngpt работают."""
    try:
        module = importlib.import_module("diff_transformer")
        if hasattr(module, "DiffGPT"):
            return module
        sys.modules.pop("diff_transformer", None)
    except ModuleNotFoundError as e:
        if e.name != "diff_transformer":
            raise
    for candidate in sorted(HERE.glob("*/diff_transformer.py")) + sorted(HERE.glob("*/*/diff_transformer.py")):
        sys.path.insert(0, str(candidate.parent))
        spec = importlib.util.spec_from_file_location("diff_transformer", candidate)
        module = importlib.util.module_from_spec(spec)
        sys.modules["diff_transformer"] = module
        spec.loader.exec_module(module)
        return module
    return None


import importlib.util  # noqa: E402
dt = _import_diff()
import baseline_gpt as bg  # noqa: E402
import ngpt as ng          # noqa: E402

MODELS = ("gpt", "ngpt", "diff")
# tiny, small, 0.5B, 1B — общие для всех моделей
PRESETS = {**bg.PRESETS, "small": dict(n_layer=6, n_head=6, d_model=384, block_size=256),
           **(dt.PRESETS if dt is not None else {})}


# ============================ Модели ============================
def build_model(name, vocab_size, dims, scale_ref_dim=0):
    """Возвращает (модель, функция создания оптимизатора, функция после шага, warmup по умолчанию)."""
    if name == "gpt":
        model = bg.GPT(bg.ModelConfig(vocab_size=vocab_size, **dims))
        return model, bg.make_optimizer, (lambda: None), 200
    if name == "diff":
        if dt is None:
            raise SystemExit("diff_transformer.py не найден — модель diff недоступна")
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
PAUSED = 3   # код выхода: обучение остановлено по --deadline, продолжить можно с --resume


class TokenFile:
    """Токены uint16 из .bin-файла (tokenize_dataset.py) без загрузки в память.
    Срез [a:b] возвращает torch.long, как у обычного тензора токенов."""

    def __init__(self, path):
        self.path = str(path)
        self.mm = np.memmap(self.path, dtype=np.uint16, mode="r")

    def __len__(self):
        return len(self.mm)

    def __getitem__(self, item):
        return torch.from_numpy(np.asarray(self.mm[item], dtype=np.int64))


def load_tokens(path, tok):
    if str(path).endswith(".bin"):
        data = TokenFile(path)
        print(f"{path}: {len(data):,} токенов (memmap)", flush=True)
        return data
    return bg.load_training_tokens(path, tok)


def fixed_windows(data, T, n, seed):
    """n случайных, но фиксированных окон длины T+1 — один и тот же набор для всех моделей."""
    g = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(data) - T, (n,), generator=g)
    return torch.stack([data[int(i):int(i) + T + 1] for i in ix])


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
    """Проход по отложенной выборке непересекающимися окнами (не больше max_tokens, если задано)."""
    n = (len(data) - 1) // T
    if max_tokens:
        n = min(n, max(1, max_tokens // T))
    total, count = 0.0, 0
    for start in range(0, n, 256):   # окна читаются порциями: val.bin может быть большим
        windows = torch.stack([data[i * T:i * T + T + 1] for i in range(start, min(n, start + 256))])
        total += mean_loss(model, windows, batch_size, device, autocast) * len(windows)
        count += len(windows)
    loss = total / count
    return dict(val_loss=loss, perplexity=math.exp(min(loss, 50)), tokens=n * T)


def setup_device(requested, threads=0, dtype="auto"):
    """Возвращает (устройство, имя типа вычислений, autocast). На T4/P100/V100 нет быстрого bf16
    (torch.cuda.is_bf16_supported() там может ответить True из-за эмуляции), поэтому смотрим
    на compute capability: bf16 только с Ampere (8.0) и новее, иначе fp16 + GradScaler."""
    if threads > 0:
        torch.set_num_threads(threads)
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested != "cuda":
        return requested, "fp32", nullcontext()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if dtype == "auto":
        dtype = "bf16" if torch.cuda.get_device_capability()[0] >= 8 else "fp16"
    if dtype == "fp32":
        return requested, dtype, nullcontext()
    return requested, dtype, torch.autocast("cuda", dtype=torch.bfloat16 if dtype == "bf16" else torch.float16)


class NoScaler:
    """Заглушка GradScaler для bf16/fp32: те же методы, ничего не масштабирует."""
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def make_scaler(dtype):
    if dtype != "fp16":
        return NoScaler()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def load_splits(args, tok):
    data = load_tokens(args.data, tok)
    if args.val_data:
        return data, load_tokens(args.val_data, tok)
    n = int(0.9 * len(data))
    return data[:n], data[n:]


def read_records(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ============================ Обучение ============================
def cmd_train(args):
    device, dtype, autocast = setup_device(args.device, args.threads, args.dtype)
    run_dir = Path(args.out) / (args.name or args.model)
    metrics_path, ckpt_path = run_dir / "metrics.jsonl", run_dir / "ckpt.pt"
    records = read_records(metrics_path)
    if args.resume and any(r["type"] == "final" for r in records):
        print(f"{run_dir}: обучение уже закончено", flush=True)
        return
    resume = args.resume and ckpt_path.exists()
    # --resume без чекпойнта (оборвалось до первого сохранения) — начинаем этот прогон заново
    if records and not resume and not args.overwrite and not args.resume:
        sys.exit(f"{run_dir} уже существует: задайте другое --name, --resume или --overwrite")
    run_dir.mkdir(parents=True, exist_ok=True)

    micro = args.micro_batch or args.batch_size
    if args.batch_size % micro:
        sys.exit("--batch_size должен делиться на --micro_batch")
    accum = args.batch_size // micro

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
    scaler = make_scaler(dtype)
    warmup = default_warmup if args.warmup is None else args.warmup
    monitor = Monitor(model)
    n_params = sum(p.numel() for p in model.parameters())

    data_gen = torch.Generator().manual_seed(args.data_seed)
    val_windows = fixed_windows(val_data, T, args.eval_batches * args.batch_size, args.data_seed + 1)

    def train_batch():
        """Один шаг = batch_size окон; индексы берутся сразу на весь шаг, поэтому батчи
        не зависят от --micro_batch. Возвращает список микробатчей (x, y)."""
        ix = torch.randint(len(train_data) - T, (args.batch_size,), generator=data_gen)
        w = torch.stack([train_data[int(i):int(i) + T + 1] for i in ix])
        return [(c[:, :-1].to(device, non_blocking=True), c[:, 1:].to(device, non_blocking=True))
                for c in w.split(micro)]

    start_step, tokens, elapsed_before = 1, 0, 0.0
    if resume:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        data_gen.set_state(ck["data_gen"])
        torch.set_rng_state(ck["cpu_rng"])
        start_step, tokens, elapsed_before = ck["step"] + 1, ck["tokens"], ck["elapsed"]
        # записи, сделанные после чекпойнта, будут повторены — убираем их, чтобы не было дублей
        kept = [r for r in records if r["type"] != "final" and r["step"] <= ck["step"]]
        metrics_path.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        print(f"Продолжаю {run_dir} с шага {start_step} ({tokens:,} токенов)", flush=True)
    else:
        config = dict(model_name=args.model, n_params=n_params, device=device, dtype=dtype,
                      bf16=dtype == "bf16", warmup=warmup, tokens_per_step=args.batch_size * T,
                      model_config=asdict(model.cfg), args=vars(args))
        if device == "cuda":
            config["gpu"] = torch.cuda.get_device_name()
        (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    log = metrics_path.open("a" if resume else "w", encoding="utf-8")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()

    def save_ckpt(step):
        save_atomic(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                         data_gen=data_gen.get_state(), cpu_rng=torch.get_rng_state(), step=step, tokens=tokens,
                         elapsed=elapsed_before + time.monotonic() - began), ckpt_path)

    print(f"{args.model.upper()} | {n_params / 1e6:.2f}M параметров | {dims} | устройство {device}, {dtype} | "
          f"батч {args.batch_size}x{T} ({accum} микробатч. по {micro}) | lr {args.lr}, warmup {warmup} | "
          f"шагов {args.max_iters} | лог: {run_dir}", flush=True)

    began = time.monotonic()
    last_ckpt = began
    loss_sum, loss_count = 0.0, 0
    if not resume:
        write(dict(type="val", step=0, tokens=0,
                   val_loss=mean_loss(model, val_windows, micro, device, autocast)))
    step = start_step - 1
    try:
        for step in range(start_step, args.max_iters + 1):
            if args.deadline and time.time() > args.deadline:
                save_ckpt(step - 1)
                print(f"Время вышло на шаге {step - 1}: чекпойнт {ckpt_path}, продолжить с --resume", flush=True)
                log.close()
                sys.exit(PAUSED)
            lr = bg.lr_at(step - 1, args.lr, warmup, args.max_iters)
            for group in optimizer.param_groups:
                group["lr"] = lr
            collect = step % args.stats_every == 0 or step == 1
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for m, (x, y) in enumerate(train_batch()):
                monitor.active = collect and m == 0
                with autocast:
                    _, loss = model(x, y)
                monitor.active = False
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"loss = {loss.item()} на шаге {step}")
                scaler.scale(loss / accum).backward()
                step_loss += loss.item() / accum
            scaler.unscale_(optimizer)
            if collect:
                grads = gradient_stats(model)
            total_grad = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")).item()
            scaler.step(optimizer)
            scaler.update()
            after_step()
            tokens += args.batch_size * T
            loss_sum += step_loss
            loss_count += 1

            if collect:
                write(dict(type="stats", step=step, tokens=tokens, activations=monitor.take_activations(),
                           grads=grads, special=special_stats(model, args.model)))
            if step % args.log_every == 0 or step == args.max_iters:
                elapsed = elapsed_before + time.monotonic() - began
                record = dict(type="train", step=step, tokens=tokens, loss=loss_sum / loss_count, lr=lr,
                              grad_norm=total_grad, elapsed=elapsed, tokens_per_sec=tokens / elapsed)
                if device == "cuda":
                    record["max_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
                write(record)
                loss_sum, loss_count = 0.0, 0
            if step % args.eval_every == 0 or step == args.max_iters:
                val = mean_loss(model, val_windows, micro, device, autocast)
                write(dict(type="val", step=step, tokens=tokens, val_loss=val))
                print(f"шаг {step:6d}/{args.max_iters} | lr {lr:.2e} | train {step_loss:.3f} | val {val:.3f} | "
                      f"grad {total_grad:.2f} | {elapsed_before + time.monotonic() - began:.0f} с", flush=True)
            if args.ckpt_minutes > 0 and time.monotonic() - last_ckpt > args.ckpt_minutes * 60:
                save_ckpt(step)
                last_ckpt = time.monotonic()
    except FloatingPointError as e:
        # разошлось (обычно слишком большой lr): фиксируем как результат, а не как сбой
        write(dict(type="final", diverged=True, val_loss=float("inf"), perplexity=float("inf"), tokens=0,
                   step=step, tokens_trained=tokens, elapsed=elapsed_before + time.monotonic() - began))
        log.close()
        ckpt_path.unlink(missing_ok=True)
        print(f"Обучение разошлось: {e}", flush=True)
        return
    except KeyboardInterrupt:
        save_ckpt(step - 1)
        log.close()
        print(f"\nОстановлено вручную: чекпойнт {ckpt_path}, продолжить с --resume", flush=True)
        sys.exit(PAUSED)

    torch.save(dict(model_name=args.model, config=asdict(model.cfg), model=model.state_dict(),
                    step=step, tokens=tokens), run_dir / "model.pt")
    final = full_eval(model, val_data, T, micro, device, autocast, args.final_eval_tokens)
    final.update(type="final", step=step, tokens_trained=tokens, elapsed=elapsed_before + time.monotonic() - began)
    write(final)
    log.close()
    ckpt_path.unlink(missing_ok=True)
    print(f"Итог на отложенной выборке ({final['tokens']:,} токенов): loss {final['val_loss']:.4f}, "
          f"perplexity {final['perplexity']:.2f}", flush=True)


def cmd_eval(args):
    device, _, autocast = setup_device(args.device, args.threads, args.dtype)
    model, name = model_from_saved(Path(args.run) / "model.pt", device)
    tok = bg.load_tokenizer()
    data = load_tokens(args.data, tok)
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
    t.add_argument("--batch_size", type=int, default=32, help="окон длины block_size на один шаг оптимизатора")
    t.add_argument("--micro_batch", type=int, default=0,
                   help="окон за один проход (накопление градиента); 0 — весь батч сразу")
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
    t.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"],
                   help="auto: bf16 на Ampere и новее, fp16 + GradScaler на T4/P100/V100")
    t.add_argument("--threads", type=int, default=0, help="потоки CPU; 0 — по умолчанию torch")
    t.add_argument("--resume", action="store_true", help="продолжить с ckpt.pt, если он есть")
    t.add_argument("--deadline", type=float, default=0,
                   help="unix-время: сохранить чекпойнт и выйти с кодом 3 (лимит сессии Kaggle)")
    t.add_argument("--ckpt_minutes", type=float, default=20, help="как часто сохранять ckpt.pt; 0 — никогда")

    e = sub.add_parser("eval", help="loss обученной модели на отложенной выборке")
    e.add_argument("run")
    e.add_argument("--data", required=True)
    e.add_argument("--batch_size", type=int, default=32)
    e.add_argument("--max_tokens", type=int, default=0)
    e.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    e.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
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
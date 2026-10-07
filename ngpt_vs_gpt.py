"""
Эксперимент: даёт ли nGPT ускорение по сравнению с обычным трансформером (GPT)?

Вопрос из статьи (arXiv:2410.01131, рис. 1): сколько токенов нужно GPT, чтобы дойти
до того же val loss, что у nGPT? Ускорение = токены GPT / токены nGPT при равном loss.

Как устроено (всё через compare.py train, рецепты моделей — из статей):
  1. lr-свип на малом бюджете: обе модели с каждым lr из --lrs. Нужен, чтобы сравнивать
     с хорошо настроенным GPT, а не с плохим (иначе «ускорение» ничего не значит).
  2. Основные прогоны: лучшие lr каждой модели на бюджетах побольше. На каждом бюджете
     своё косинусное расписание до нуля, как в статье.
  3. Отчёт: финальный val loss от числа токенов для обеих моделей и ускорение nGPT.

Работает в несколько заходов (сессия Kaggle длится максимум 12 часов): перед лимитом
обучение сохраняет чекпойнт, следующий запуск той же команды продолжает с того же места.
Если GPU две (Kaggle «T4 x2»), на каждой идёт своя модель — вдвое быстрее.

    python ngpt_vs_gpt.py bench  --data data/owt            # скорость и сколько займёт весь план
    python ngpt_vs_gpt.py run    --data data/owt --hours 11 # запуск / продолжение
    python ngpt_vs_gpt.py report                            # таблицы, графики, ускорение

В --data лежат train.bin и val.bin из tokenize_dataset.py.
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
COMPARE = HERE / "compare.py"
MODELS = ("gpt", "ngpt")
PAUSED = 3

# Модель «S»: 8 слоёв, d=512, 8 голов (d_k = 64, как в статье), контекст 1024 (самый короткий в статье).
# ~25M параметров без эмбеддингов + 2 x 16.4M эмбеддинги (раздельные, как в статье).
DEFAULT_PLAN = dict(
    n_layer=8, n_head=8, d_model=512, block_size=1024,
    batch_size=64, micro_batch=8,           # 64 x 1024 = 65 536 токенов на шаг, по 8 окон за проход (память T4)
    budgets="100M,300M,1B",                 # первый бюджет — для lr-свипа
    lrs="1e-3,2e-3,4e-3",
    scale_ref_dim=1024,                     # масштабы nGPT учатся как у модели 0.5B из статьи (см. ngpt.py)
    warmup_frac=0.05,                       # warmup GPT: 5% шагов (не меньше 100, не больше 1000); у nGPT 0
    budget_scale=1.0,
    eval_batches=2,                         # фиксированный val-набор: 2 x 64 окна = 131k токенов
    final_eval_tokens=4_000_000,
    dtype="auto",
)


# ============================ План ============================
def parse_count(text):
    text = str(text).strip().upper()
    mult = {"K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9}.get(text[-1:], 1)
    return int(float(text[:-1] if text[-1:] in "KMBG" else text) * mult)


def fmt_count(n):
    for unit, size in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            value = n / size
            return f"{value:.0f}{unit}" if abs(value - round(value)) < 0.05 else f"{value:.1f}{unit}"
    return str(n)


def make_jobs(plan):
    """Список прогонов. lr=None — «лучший lr из свипа этой модели» (определяется позже)."""
    tokens_per_step = plan["batch_size"] * plan["block_size"]
    budgets = [parse_count(b) for b in plan["budgets"].split(",")]
    lrs = [float(x) for x in plan["lrs"].split(",")]

    def steps(budget):
        return max(1, round(budget * plan["budget_scale"] / tokens_per_step))

    jobs = []
    for lr in lrs:                                   # 1. свип: модели вперемешку, чтобы обе GPU были заняты
        for model in MODELS:
            jobs.append(dict(name=f"{model}_{fmt_count(budgets[0])}_lr{lr:g}", model=model, lr=lr,
                             budget=budgets[0], steps=steps(budgets[0]), stage="sweep"))
    for budget in sorted(budgets[1:], reverse=True):  # 2. основные: длинные первыми
        for model in MODELS:
            jobs.append(dict(name=f"{model}_{fmt_count(budget)}", model=model, lr=None,
                             budget=budget, steps=steps(budget), stage="main"))
    return jobs


def load_plan(out, cli_plan, new_plan=False, explicit=None):
    """План фиксируется в первом запуске (plan.json), иначе продолжение пошло бы с другими шагами/lr."""
    path = Path(out) / "plan.json"
    if path.exists() and not new_plan:
        plan = json.loads(path.read_text(encoding="utf-8"))
        changed = {k: v for k, v in (explicit or {}).items() if k in plan and plan[k] != v}
        if changed:
            print(f"Внимание: план взят из {path}, аргументы {sorted(changed)} проигнорированы "
                  f"(--new_plan, чтобы начать заново в другой папке --out)", flush=True)
        return plan
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cli_plan, indent=2), encoding="utf-8")
    return dict(cli_plan)


# ============================ Состояние прогонов ============================
def read_metrics(run_dir):
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:   # строка, недописанная при обрыве
            pass
    return out


def final_record(run_dir):
    return next((r for r in reversed(read_metrics(run_dir)) if r["type"] == "final"), None)


def job_state(runs, job):
    d = runs / job["name"]
    if final_record(d):
        return "done"
    return "paused" if (d / "ckpt.pt").exists() else "new"


def best_lr(runs, jobs, model):
    """Лучший lr модели по финальному val loss свипа; None, пока свип не закончен."""
    sweep = [j for j in jobs if j["model"] == model and j["stage"] == "sweep"]
    finals = [(final_record(runs / j["name"]), j["lr"]) for j in sweep]
    if any(f is None for f, _ in finals):
        return None
    return min(finals, key=lambda t: t[0]["val_loss"])[1]


# ============================ Запуск ============================
def train_command(plan, job, lr, data, runs, device, deadline):
    steps = job["steps"]
    warmup = 0
    if job["model"] == "gpt":
        warmup = int(min(1000, max(100, plan["warmup_frac"] * steps), steps // 2))
    data = Path(data)
    cmd = [sys.executable, str(COMPARE), "train", "--model", job["model"], "--name", job["name"],
           "--out", str(runs), "--data", str(data / "train.bin"), "--val_data", str(data / "val.bin"),
           "--preset", "tiny", "--n_layer", plan["n_layer"], "--n_head", plan["n_head"],
           "--d_model", plan["d_model"], "--block_size", plan["block_size"],
           "--batch_size", plan["batch_size"], "--micro_batch", plan["micro_batch"],
           "--max_iters", steps, "--lr", lr, "--warmup", warmup,
           "--log_every", max(1, min(20, steps // 50)), "--stats_every", max(1, min(200, steps // 20)),
           "--eval_every", max(1, steps // 40), "--eval_batches", plan["eval_batches"],
           "--final_eval_tokens", plan["final_eval_tokens"], "--dtype", plan["dtype"],
           "--device", "cpu" if device == "cpu" else "cuda", "--resume", "--deadline", deadline]
    if job["model"] == "ngpt":
        cmd += ["--scale_ref_dim", plan["scale_ref_dim"]]
    if plan.get("threads"):
        cmd += ["--threads", plan["threads"]]
    return [str(c) for c in cmd]


def detect_slots(requested):
    if requested == "cpu":
        return ["cpu"]
    try:
        import torch
        n = torch.cuda.device_count()
    except ImportError:
        n = 0
    if n == 0:
        print("GPU не найдены — считаю на CPU (только для проверки)", flush=True)
        return ["cpu"]
    return [str(i) for i in range(n)]


def cmd_run(args):
    out = Path(args.out)
    plan = load_plan(out, plan_from_args(args), args.new_plan, explicit_args(args))
    jobs = make_jobs(plan)
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.hours * 3600
    if args.device == "cpu":
        slots = [f"cpu{i}" for i in range(args.workers_per_slot)]
    else:
        slots = detect_slots(args.device)
    print(f"План: {len(jobs)} прогонов, устройства {slots}, лимит {args.hours:.2f} ч", flush=True)
    print_status(runs, jobs)

    running, failed = {}, set()
    while True:
        for slot, (job, proc, logf) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            logf.close()
            del running[slot]
            if rc == 0:
                fin = final_record(runs / job["name"])
                print(f"[{slot}] {job['name']}: готово, val loss {fin['val_loss']:.4f}", flush=True)
            elif rc == PAUSED:
                print(f"[{slot}] {job['name']}: пауза (время сессии), продолжится при следующем запуске", flush=True)
            else:
                failed.add(job["name"])
                tail = (runs / job["name"] / "train.log").read_text(encoding="utf-8", errors="replace")[-1500:]
                print(f"[{slot}] {job['name']}: ОШИБКА (код {rc}). Конец лога:\n{tail}", flush=True)

        out_of_time = time.time() > deadline
        if not out_of_time:
            for slot in slots:
                if slot in running:
                    continue
                job, lr = next_job(runs, jobs, running, failed)
                if job is None:
                    break
                cmd = train_command(plan, job, lr, args.data, runs, "cpu" if slot.startswith("cpu") else "cuda",
                                    deadline)
                env = dict(os.environ)
                if not slot.startswith("cpu"):
                    env["CUDA_VISIBLE_DEVICES"] = slot
                (runs / job["name"]).mkdir(parents=True, exist_ok=True)
                logf = (runs / job["name"] / "train.log").open("a", encoding="utf-8")
                state = job_state(runs, job)
                print(f"[{slot}] {job['name']}: {'продолжаю' if state == 'paused' else 'старт'} "
                      f"(lr {lr:g}, {job['steps']} шагов, {fmt_count(job['steps'] * plan['batch_size'] * plan['block_size'])} токенов)",
                      flush=True)
                running[slot] = (job, subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env), logf)
        if not running:
            break
        time.sleep(args.poll)

    print_status(runs, jobs)
    left = [j["name"] for j in jobs if job_state(runs, j) != "done"]
    if left:
        print(f"Не закончено: {len(left)} прогонов. Запустите ту же команду ещё раз, чтобы продолжить.", flush=True)
    else:
        print("Все прогоны закончены.", flush=True)
    report(out)


def next_job(runs, jobs, running, failed):
    busy = {job["name"] for job, _, _ in running.values()}
    for job in jobs:
        if job["name"] in busy or job["name"] in failed or job_state(runs, job) == "done":
            continue
        lr = job["lr"]
        if lr is None:
            lr = best_lr(runs, jobs, job["model"])
            if lr is None:
                continue   # ждём конца свипа этой модели
            saved = runs / job["name"] / "config.json"
            if saved.exists():   # продолжение: lr уже был выбран в прошлый раз
                lr = json.loads(saved.read_text(encoding="utf-8"))["args"]["lr"]
        return job, lr
    return None, None


def print_status(runs, jobs):
    marks = {"done": "готово", "paused": "на паузе", "new": "ждёт"}
    for j in jobs:
        state = job_state(runs, j)
        fin = final_record(runs / j["name"]) if state == "done" else None
        extra = f" val {fin['val_loss']:.4f}" if fin else ""
        print(f"  {j['name']:<22} {marks[state]}{extra}", flush=True)


# ============================ Замер скорости ============================
def cmd_bench(args):
    """Несколько шагов каждой модели на первой GPU: токены в секунду, память и сколько займёт план."""
    out = Path(args.out)
    plan = load_plan(out, plan_from_args(args), args.new_plan, explicit_args(args)) \
        if (out / "plan.json").exists() else plan_from_args(args)
    bench_dir = out / "bench"
    slot = detect_slots(args.device)[0]
    speeds = {}
    for model in MODELS:
        job = dict(name=f"bench_{model}", model=model, steps=args.steps)
        cmd = train_command(dict(plan, final_eval_tokens=plan["block_size"]), job, 1e-3, args.data, bench_dir,
                            "cpu" if slot == "cpu" else "cuda", 0)
        cmd = [c for c in cmd if c != "--resume"] + ["--overwrite", "--log_every", "2", "--eval_every",
                                                    str(args.steps), "--stats_every", str(10 ** 9)]
        env = dict(os.environ)
        if slot != "cpu":
            env["CUDA_VISIBLE_DEVICES"] = slot
        print(f"Замер {model}: {args.steps} шагов...", flush=True)
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-2000:] + res.stderr[-3000:])
            sys.exit(f"{model}: ошибка при замере (если CUDA out of memory — уменьшите --micro_batch)")
        train = [r for r in read_metrics(bench_dir / job["name"]) if r["type"] == "train"]
        a, b = train[len(train) // 3], train[-1]   # первые шаги — прогрев CUDA, их не считаем
        speeds[model] = (b["tokens"] - a["tokens"]) / max(1e-9, b["elapsed"] - a["elapsed"])
        mem = b.get("max_mem_gb")
        print(f"  {model}: {speeds[model]:,.0f} токенов/с" + (f", память {mem:.1f} ГБ" if mem else ""), flush=True)

    import shutil
    shutil.rmtree(bench_dir, ignore_errors=True)   # веса замеров не нужны, а место в выводе Kaggle ограничено
    jobs = make_jobs(plan)
    per_model = {m: sum(j["steps"] for j in jobs if j["model"] == m) * plan["batch_size"] * plan["block_size"]
                 for m in MODELS}
    hours = sum(per_model[m] / speeds[m] for m in MODELS) / 3600 * 1.08   # +8% на валидацию
    n_slots = len(detect_slots(args.device))
    wall = hours / n_slots
    print(f"\nПлан (budget_scale={plan['budget_scale']}): " +
          ", ".join(f"{m} {fmt_count(per_model[m])} токенов" for m in MODELS))
    print(f"Устройств: {n_slots}. Оценка: ~{wall:.1f} ч, то есть ~{math.ceil(wall / args.session_hours)} "
          f"сессий по {args.session_hours:g} ч.")
    fit = plan["budget_scale"] * args.session_hours / wall
    print(f"Чтобы уложиться в одну сессию, нужен --budget_scale {fit:.2f} (бюджеты пропорционально меньше).")
    print(f"nGPT тратит на токен в {speeds['gpt'] / speeds['ngpt']:.2f} раза больше времени, чем GPT "
          f"(лишние нормировки; ускорение по токенам должно это перекрыть)")


# ============================ Отчёт ============================
def tokens_to_reach(points, target):
    """points: [(токены, loss)] GPT по бюджетам. Сколько токенов нужно GPT до loss = target.
    Интерполяция loss по log(токены); loss делаем невозрастающим (шум между бюджетами).
    Возвращает (знак, токены): '=' — внутри диапазона, '>' — больше всех бюджетов GPT, '<' — меньше."""
    pts = sorted(points)
    mono, best = [], float("inf")
    for t, l in pts:
        best = min(best, l)
        mono.append((t, best))
    if target >= mono[0][1]:
        return "<", mono[0][0]
    if target < mono[-1][1]:
        return ">", mono[-1][0]
    for (t0, l0), (t1, l1) in zip(mono, mono[1:]):
        if l0 >= target >= l1 and l0 > l1:
            frac = (l0 - target) / (l0 - l1)
            return "=", math.exp(math.log(t0) + frac * (math.log(t1) - math.log(t0)))
    return "=", mono[-1][0]


def curve_speedup(gpt_val, ngpt_val):
    """Ускорение по кривым одного бюджета: на скольких токенах nGPT впервые дошёл до финального val loss GPT.
    gpt_val, ngpt_val: [(токены, val loss)] по ходу обучения (фиксированный набор окон).
    Возвращает (токены nGPT или None, если не дошёл; финальный loss GPT; токены GPT)."""
    gpt_val, ngpt_val = sorted(gpt_val), sorted(ngpt_val)
    target_tokens, target = gpt_val[-1]
    for (t0, l0), (t1, l1) in zip(ngpt_val, ngpt_val[1:]):
        if l1 <= target:
            if l0 <= target:
                return t0, target, target_tokens
            return t0 + (l0 - target) / (l0 - l1) * (t1 - t0), target, target_tokens
    return None, target, target_tokens


def collect(out):
    runs = Path(out) / "runs"
    rows = []
    for d in sorted(runs.iterdir()) if runs.exists() else []:
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        records = read_metrics(d)
        fin = next((r for r in reversed(records) if r["type"] == "final"), None)
        train = [r for r in records if r["type"] == "train"]
        rows.append(dict(name=d.name, dir=d, model=cfg["model_name"], lr=cfg["args"]["lr"],
                         tokens=cfg["args"]["max_iters"] * cfg["tokens_per_step"],
                         stage="sweep" if "_lr" in d.name else "main", final=fin,
                         tok_s=train[-1]["tokens_per_sec"] if train else None,
                         progress=(train[-1]["step"] / cfg["args"]["max_iters"]) if train else 0.0))
    return rows


def report(out, quiet=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import compare

    out = Path(out)
    rows = collect(out)
    if not rows:
        print("Нет прогонов для отчёта")
        return None
    rep = out / "report"
    rep.mkdir(parents=True, exist_ok=True)
    done = [r for r in rows if r["final"]]
    loss = lambda r: r["final"]["val_loss"]   # noqa: E731
    lines = ["# nGPT против GPT", ""]

    # --- lr-свип ---
    sweep = [r for r in rows if r["stage"] == "sweep"]
    best = {}
    if sweep:
        lines += ["## lr-свип", "", "| модель | lr | токенов | val loss |", "|---|---|---|---|"]
        for r in sorted(sweep, key=lambda r: (r["model"], r["lr"])):
            val = ("разошлось" if r["final"].get("diverged") else f"{loss(r):.4f}") if r["final"] \
                else f"идёт ({r['progress']:.0%})"
            lines.append(f"| {r['model']} | {r['lr']:g} | {fmt_count(r['tokens'])} | {val} |")
        for m in MODELS:
            ms = [r for r in sweep if r["model"] == m and r["final"]]
            if ms:
                b = min(ms, key=loss)
                best[m] = b
                lrs = sorted(r["lr"] for r in sweep if r["model"] == m)
                edge = " — на краю сетки, стоит проверить lr за её пределами" if b["lr"] in (lrs[0], lrs[-1]) \
                    and len(lrs) > 2 else ""
                lines.append(f"\nЛучший lr {m}: **{b['lr']:g}**{edge}")
        lines.append("")

    # --- финальный loss по бюджетам: лучшая точка свипа + основные прогоны ---
    curve = {m: [] for m in MODELS}
    for m in MODELS:
        pts = [r for r in done if r["model"] == m and r["stage"] == "main"]
        if m in best:
            pts.append(best[m])
        curve[m] = sorted(((r["tokens"], loss(r), r) for r in pts if not r["final"].get("diverged")),
                          key=lambda t: t[0])
    lines += ["## Итоговый val loss по бюджетам", "", "| токенов | GPT | nGPT | разница |", "|---|---|---|---|"]
    budgets = sorted({t for m in MODELS for t, _, _ in curve[m]})
    for b in budgets:
        g = next((l for t, l, _ in curve["gpt"] if t == b), None)
        n = next((l for t, l, _ in curve["ngpt"] if t == b), None)
        diff = f"{n - g:+.4f}" if g is not None and n is not None else "—"
        lines.append(f"| {fmt_count(b)} | {g if g is None else f'{g:.4f}'} | {n if n is None else f'{n:.4f}'} | {diff} |")

    # --- ускорение ---
    lines += ["", "## Ускорение nGPT", "",
              "Сколько токенов нужно GPT, чтобы дойти до того же val loss, что у nGPT на бюджете D.", ""]
    speedups = []
    if len(curve["gpt"]) >= 1 and curve["ngpt"]:
        gpt_pts = [(t, l) for t, l, _ in curve["gpt"]]
        for t, l, _ in curve["ngpt"]:
            sign, need = tokens_to_reach(gpt_pts, l)
            ratio = need / t
            text = {"=": f"≈ {ratio:.2f}x", ">": f"больше {ratio:.2f}x (GPT не дошёл до этого loss даже на "
                                                 f"{fmt_count(need)})",
                    "<": f"не больше {ratio:.2f}x (GPT уже на {fmt_count(need)} лучше)"}[sign]
            speedups.append(dict(tokens=t, sign=sign, ratio=ratio))
            lines.append(f"- nGPT на {fmt_count(t)} (loss {l:.4f}): GPT нужно {'' if sign == '=' else sign + ' '}"
                         f"{fmt_count(need)} токенов → ускорение {text}")
    else:
        lines.append("- пока не хватает законченных прогонов")
    # ускорение по кривым обучения: прогоны GPT и nGPT с одинаковым бюджетом
    curve_speedups = []
    for t_budget in sorted({r["tokens"] for r in done if r["model"] == "gpt"} &
                           {r["tokens"] for r in done if r["model"] == "ngpt"}):
        pick = {}
        for m in MODELS:
            same = [r for r in done if r["model"] == m and r["tokens"] == t_budget and not r["final"].get("diverged")]
            if same:
                pick[m] = min(same, key=loss)
        if len(pick) < 2:
            continue
        vals = {m: [(v["tokens"], v["val_loss"]) for v in read_metrics(pick[m]["dir"])
                    if v["type"] == "val" and v["step"] > 0] for m in MODELS}
        if len(vals["gpt"]) < 2 or len(vals["ngpt"]) < 2:
            continue
        reached, target, gpt_tokens = curve_speedup(vals["gpt"], vals["ngpt"])
        if not curve_speedups:
            lines += ["", "## Ускорение по кривым обучения (один бюджет)", "",
                      "На скольких токенах nGPT впервые дошёл до финального val loss GPT того же бюджета.", ""]
        if reached is None:
            lines.append(f"- бюджет {fmt_count(t_budget)}: GPT в конце {target:.4f}, nGPT до этого loss не дошёл")
            curve_speedups.append(dict(tokens=t_budget, ratio=None))
        else:
            ratio = gpt_tokens / reached
            lines.append(f"- бюджет {fmt_count(t_budget)}: GPT дошёл до {target:.4f} за {fmt_count(gpt_tokens)} токенов, "
                         f"nGPT — за {fmt_count(int(reached))} → ускорение ≈ {ratio:.2f}x")
            curve_speedups.append(dict(tokens=t_budget, ratio=ratio, ngpt_tokens=reached))
    if curve_speedups:
        lines.append("  (оценка внутри одного прогона: у GPT и nGPT своё расписание lr, поэтому она приблизительная)")

    speed = {m: [r["tok_s"] for r in done if r["model"] == m and r["tok_s"]] for m in MODELS}
    if speed["gpt"] and speed["ngpt"]:
        sg, sn = sum(speed["gpt"]) / len(speed["gpt"]), sum(speed["ngpt"]) / len(speed["ngpt"])
        lines += ["", f"Скорость (с валидацией): GPT {sg:,.0f}, nGPT {sn:,.0f} токенов/с. Время на токен у nGPT "
                      f"в {sg / sn:.2f} раза больше, чем у GPT, поэтому ускорение по времени = "
                      f"ускорение по токенам / {sg / sn:.2f}."]

    # --- графики ---
    colors = compare.COLORS
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    ax = axes[0]
    for m in MODELS:
        if curve[m]:
            ax.plot([t for t, _, _ in curve[m]], [l for _, l, _ in curve[m]], marker="o", color=colors[m],
                    ls=compare.STYLES[m], label=m.upper())
    ax.set_xscale("log")
    ax.set_xlabel("токенов обучения (у каждого бюджета своё расписание lr)")
    ax.set_ylabel("итоговый val loss")
    ax.set_title("Итоговый val loss по бюджетам: левее и ниже — лучше")
    ax.grid(alpha=0.3, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax = axes[1]
    for m in MODELS:
        ms = sorted((r for r in sweep if r["model"] == m and r["final"] and not r["final"].get("diverged")),
                    key=lambda r: r["lr"])
        if ms:
            ax.plot([r["lr"] for r in ms], [loss(r) for r in ms], marker="o", color=colors[m],
                    ls=compare.STYLES[m], label=m.upper())
    ax.set_xscale("log")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("итоговый val loss")
    ax.set_title(f"lr-свип ({fmt_count(sweep[0]['tokens']) if sweep else '—'} токенов)")
    ax.grid(alpha=0.3, which="both")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    fig.tight_layout()
    fig.savefig(rep / "ngpt_vs_gpt.png", dpi=130)
    plt.close(fig)

    # подробные графики compare.py: кривые обучения, градиенты, активации — для лучших прогонов
    detail = [r for r in rows if r["stage"] == "main" or best.get(r["model"]) is r]
    detail = [r for r in detail if read_metrics(r["dir"])]
    if detail:
        old_stdout = sys.stdout
        try:
            sys.stdout = open(os.devnull, "w", encoding="utf-8")
            compare.cmd_plot(argparse.Namespace(runs=[str(r["dir"]) for r in detail], out=str(rep / "details"),
                                                x="tokens", smooth=0.03))
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

    lines += ["", "Графики: `ngpt_vs_gpt.png` (главный), `details/loss.png` (кривые обучения), "
                  "`details/*.png` (градиенты, активации, α и s_z у nGPT).", ""]
    text = "\n".join(lines)
    (rep / "report.md").write_text(text, encoding="utf-8")
    (rep / "summary.json").write_text(json.dumps(dict(
        best_lr={m: best[m]["lr"] for m in best},
        final={m: [(t, l) for t, l, _ in curve[m]] for m in MODELS},
        speedups=speedups, curve_speedups=curve_speedups), indent=2), encoding="utf-8")
    if not quiet:
        print(text)
        print(f"Отчёт: {rep}")
    return rep


def cmd_report(args):
    report(args.out)


# ============================ CLI ============================
def explicit_args(args):
    return {k: getattr(args, k) for k in DEFAULT_PLAN if getattr(args, k, None) is not None}


def plan_from_args(args):
    plan = dict(DEFAULT_PLAN, **explicit_args(args))
    if getattr(args, "threads", 0):
        plan["threads"] = args.threads
    return plan


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="nGPT против GPT: lr-свип, бюджеты, ускорение")
    sub = parser.add_subparsers(dest="command", required=True)

    def plan_args(p):
        p.add_argument("--data", required=True, help="папка с train.bin и val.bin")
        p.add_argument("--out", default="runs/ngpt_vs_gpt")
        p.add_argument("--device", default="auto", choices=["auto", "cpu"])
        p.add_argument("--new_plan", action="store_true", help="перезаписать plan.json (лучше новая папка --out)")
        p.add_argument("--threads", type=int, default=0)
        for key, value in DEFAULT_PLAN.items():   # None = не задано в командной строке
            p.add_argument(f"--{key}", type=type(value), default=None, help=f"по умолчанию {value}")

    r = sub.add_parser("run", help="запустить или продолжить эксперимент")
    plan_args(r)
    r.add_argument("--hours", type=float, default=11.0, help="сколько часов можно работать в этот раз")
    r.add_argument("--workers_per_slot", type=int, default=1, help="только для --device cpu (тесты)")
    r.add_argument("--poll", type=float, default=10.0)

    b = sub.add_parser("bench", help="замерить скорость и оценить время всего плана")
    plan_args(b)
    b.add_argument("--steps", type=int, default=30)
    b.add_argument("--session_hours", type=float, default=11.0)

    rep = sub.add_parser("report", help="таблицы, графики и ускорение")
    rep.add_argument("--out", default="runs/ngpt_vs_gpt")

    args = parser.parse_args()
    {"run": cmd_run, "bench": cmd_bench, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
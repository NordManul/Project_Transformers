"""
Тесты всего, что нужно для запуска на Kaggle:
  tokenize_dataset.py — запись train.bin / val.bin;
  compare.py          — обучение из .bin, накопление градиента, пауза и продолжение, расхождение;
  ngpt_vs_gpt.py      — план, оценка ускорения и полный маленький эксперимент на CPU.

    pytest tests/test_kaggle.py -v
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import compare
import ngpt_vs_gpt as exp
import tokenize_dataset as td

ROOT = Path(__file__).resolve().parent.parent


class DummyTokenizer:
    vocab_size, bos_token_id, eos_token_id = 200, 1, 2

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 190 for c in text]

    def __call__(self, texts, add_special_tokens=False):
        return {"input_ids": [self.encode(t) for t in texts]}


def make_bins(tmp_path, train_tokens=40_000, val_tokens=4_000):
    texts = (f"the quick brown fox {i} jumps over the lazy dog. " * 3 for i in range(100_000))
    td.tokenize_to_bin(texts, DummyTokenizer(), tmp_path / "data", train_tokens, val_tokens)
    return tmp_path / "data"


# ============================ tokenize_dataset.py ============================
def test_tokenize_to_bin(tmp_path):
    data = make_bins(tmp_path, train_tokens=5_000, val_tokens=700)
    train = np.fromfile(data / "train.bin", dtype=np.uint16)
    val = np.fromfile(data / "val.bin", dtype=np.uint16)
    assert len(train) == 5_000 and len(val) == 700
    assert val[0] == 1 and train[0] == 1            # каждый документ начинается с BOS
    doc = [1] + DummyTokenizer().encode("the quick brown fox 0 jumps over the lazy dog. " * 3)
    assert val[:len(doc)].tolist() == doc
    assert not list(data.glob("*.tmp"))


def test_token_file_slices_like_tensor(tmp_path):
    data = make_bins(tmp_path, train_tokens=3_000, val_tokens=300)
    tf = compare.TokenFile(data / "train.bin")
    ref = torch.from_numpy(np.fromfile(data / "train.bin", dtype=np.uint16).astype(np.int64))
    assert len(tf) == len(ref)
    assert torch.equal(tf[100:229], ref[100:229]) and tf[100:229].dtype == torch.long


# ============================ compare.py ============================
@pytest.fixture
def bins(tmp_path, monkeypatch):
    monkeypatch.setattr(compare.bg, "load_tokenizer", lambda: DummyTokenizer())
    return make_bins(tmp_path)


def run_train(monkeypatch, tmp_path, data, name, *extra, model="gpt"):
    argv = ["compare.py", "train", "--model", model, "--name", name, "--out", tmp_path / "runs",
            "--data", data / "train.bin", "--val_data", data / "val.bin",
            "--n_layer", 2, "--n_head", 2, "--d_model", 32, "--block_size", 16, "--batch_size", 8,
            "--max_iters", 8, "--threads", 1, "--log_every", 1, "--stats_every", 4, "--eval_every", 4,
            "--eval_batches", 1, "--final_eval_tokens", 2_000, "--device", "cpu", "--ckpt_minutes", 0, *extra]
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv])
    compare.main()
    path = tmp_path / "runs" / name / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def final_loss(records):
    return [r for r in records if r["type"] == "final"][-1]["val_loss"]


@pytest.mark.parametrize("model", ["gpt", "ngpt"])
def test_micro_batches_match_full_batch(model, tmp_path, monkeypatch, bins):
    full = run_train(monkeypatch, tmp_path, bins, f"{model}_full", model=model)
    micro = run_train(monkeypatch, tmp_path, bins, f"{model}_micro", "--micro_batch", 2, model=model)
    assert final_loss(full) == pytest.approx(final_loss(micro), abs=1e-4)


@pytest.mark.parametrize("model", ["gpt", "ngpt"])
def test_pause_and_resume_is_exact(model, tmp_path, monkeypatch, bins):
    """Прогон с остановкой по --deadline и продолжением даёт тот же результат, что прогон без остановки."""
    straight = run_train(monkeypatch, tmp_path, bins, "straight", model=model)

    class FakeTime:   # time.time() растёт на 1 при каждой проверке дедлайна -> пауза перед 5-м шагом
        now = 0.0
        monotonic = staticmethod(compare.time.monotonic)

        @classmethod
        def time(cls):
            cls.now += 1
            return cls.now

    monkeypatch.setattr(compare, "time", FakeTime)
    with pytest.raises(SystemExit) as stop:
        run_train(monkeypatch, tmp_path, bins, "paused", "--deadline", 4.5, model=model)
    assert stop.value.code == compare.PAUSED
    assert (tmp_path / "runs" / "paused" / "ckpt.pt").exists()
    monkeypatch.undo()
    monkeypatch.setattr(compare.bg, "load_tokenizer", lambda: DummyTokenizer())
    resumed = run_train(monkeypatch, tmp_path, bins, "paused", "--resume", model=model)

    assert final_loss(resumed) == pytest.approx(final_loss(straight), abs=1e-6)
    steps = [r["step"] for r in resumed if r["type"] == "train"]
    assert steps == list(range(1, 9))                       # без дублей и пропусков
    assert not (tmp_path / "runs" / "paused" / "ckpt.pt").exists()
    # повторный --resume законченного прогона ничего не делает
    again = run_train(monkeypatch, tmp_path, bins, "paused", "--resume", model=model)
    assert again == resumed


def test_resume_without_checkpoint_restarts(tmp_path, monkeypatch, bins):
    first = run_train(monkeypatch, tmp_path, bins, "r")
    (tmp_path / "runs" / "r" / "metrics.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in first if r["type"] != "final"), encoding="utf-8")
    again = run_train(monkeypatch, tmp_path, bins, "r", "--resume")
    assert final_loss(again) == pytest.approx(final_loss(first), abs=1e-6)


def test_divergence_is_recorded(tmp_path, monkeypatch, bins):
    original = compare.build_model

    def nan_after_three(name, *args, **kwargs):
        model, *rest = original(name, *args, **kwargs)
        forward, calls = model.forward, [0]

        def broken(x, y=None):
            logits, loss = forward(x, y)
            calls[0] += model.training
            return logits, (loss * float("nan") if calls[0] > 3 and loss is not None else loss)

        model.forward = broken
        return (model, *rest)

    monkeypatch.setattr(compare, "build_model", nan_after_three)
    records = run_train(monkeypatch, tmp_path, bins, "nan")
    fin = records[-1]
    assert fin["type"] == "final" and fin["diverged"] and fin["val_loss"] == float("inf")


# ============================ ngpt_vs_gpt.py ============================
def test_parse_and_format_counts():
    assert exp.parse_count("100M") == 100_000_000 and exp.parse_count("1B") == 10 ** 9
    assert exp.parse_count("2e8") == 200_000_000 and exp.fmt_count(300_000_000) == "300M"


def test_plan_jobs():
    plan = dict(exp.DEFAULT_PLAN)
    jobs = exp.make_jobs(plan)
    sweep = [j for j in jobs if j["stage"] == "sweep"]
    main = [j for j in jobs if j["stage"] == "main"]
    assert len(sweep) == 6 and len(main) == 4
    assert all(j["lr"] is None for j in main)
    assert [j["budget"] for j in main] == [10 ** 9, 10 ** 9, 300_000_000, 300_000_000]   # длинные первыми
    assert sweep[0]["steps"] == round(100e6 / 65536)
    half = exp.make_jobs(dict(plan, budget_scale=0.5))
    assert half[0]["steps"] == round(50e6 / 65536)


def test_tokens_to_reach():
    gpt = [(100, 4.0), (300, 3.6), (1000, 3.3)]
    sign, t = exp.tokens_to_reach(gpt, 3.6)
    assert sign == "=" and t == pytest.approx(300)
    sign, t = exp.tokens_to_reach(gpt, 3.45)      # посередине между 300 и 1000 по log
    assert sign == "=" and t == pytest.approx((300 * 1000) ** 0.5)
    assert exp.tokens_to_reach(gpt, 3.1) == (">", 1000)
    assert exp.tokens_to_reach(gpt, 4.5) == ("<", 100)


@pytest.mark.skipif(not (ROOT / "tokenizer" / "tokenizer_config.json").exists(),
                    reason="нужен токенизатор в tokenizer/ (обучение идёт в отдельных процессах)")
def test_full_experiment_on_cpu(tmp_path, monkeypatch):
    """Маленький полный эксперимент: свип, выбор lr, основные прогоны, пауза, продолжение и отчёт."""
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    rng.integers(3, 300, 30_000).astype(np.uint16).tofile(data / "train.bin")
    rng.integers(3, 300, 3_000).astype(np.uint16).tofile(data / "val.bin")
    out = tmp_path / "exp"
    common = ["--data", data, "--out", out, "--device", "cpu", "--poll", 0.2]
    plan = ["--n_layer", 1, "--n_head", 2, "--d_model", 32, "--block_size", 16, "--batch_size", 4,
            "--micro_batch", 2, "--budgets", "320,640", "--lrs", "1e-3,3e-3", "--final_eval_tokens", 500,
            "--threads", 1]
    monkeypatch.setattr(sys, "argv", ["ngpt_vs_gpt.py", "run", *map(str, common + plan), "--hours", "0"])
    exp.main()   # лимит 0 часов: ничего не запускается, но план сохранён
    assert json.loads((out / "plan.json").read_text())["budgets"] == "320,640"

    monkeypatch.setattr(sys, "argv", ["ngpt_vs_gpt.py", "run", *map(str, common), "--hours", "1",
                                      "--workers_per_slot", "2"])
    exp.main()
    runs = out / "runs"
    names = sorted(p.name for p in runs.iterdir())
    assert names == sorted(["gpt_320_lr0.001", "gpt_320_lr0.003", "ngpt_320_lr0.001", "ngpt_320_lr0.003",
                            "gpt_640", "ngpt_640"])
    for model in ("gpt", "ngpt"):
        best = exp.best_lr(runs, exp.make_jobs(json.loads((out / "plan.json").read_text())), model)
        cfg = json.loads((runs / f"{model}_640" / "config.json").read_text())
        assert cfg["args"]["lr"] == best and cfg["args"]["max_iters"] == 10
    summary = json.loads((out / "report" / "summary.json").read_text())
    assert len(summary["final"]["gpt"]) == 2 and len(summary["speedups"]) == 2
    assert (out / "report" / "ngpt_vs_gpt.png").stat().st_size > 0
    assert (out / "report" / "details" / "loss.png").stat().st_size > 0


def test_curve_speedup():
    gpt = [(100, 5.0), (200, 4.0), (300, 3.5)]
    ngpt = [(100, 4.5), (200, 3.3), (300, 3.0)]
    reached, target, total = exp.curve_speedup(gpt, ngpt)
    assert target == 3.5 and total == 300
    assert reached == pytest.approx(100 + (4.5 - 3.5) / (4.5 - 3.3) * 100)
    assert exp.curve_speedup(gpt, [(100, 6.0), (300, 4.0)])[0] is None


def test_plan_seeds_and_per_model_lrs():
    plan = dict(exp.DEFAULT_PLAN, budgets="775M", lrs="3e-3", lrs_gpt="1.5e-3", lrs_ngpt="6e-3,1.2e-2",
                seed=43, data_seed=1235)
    jobs = exp.make_jobs(plan)
    assert [j["name"] for j in jobs] == ["gpt_775M_lr0.0015_s43", "ngpt_775M_lr0.006_s43", "ngpt_775M_lr0.012_s43"]
    cmd = exp.train_command(plan, jobs[0], 1.5e-3, "data", Path("runs"), "cpu", 0)
    assert cmd[cmd.index("--seed") + 1] == "43" and cmd[cmd.index("--data_seed") + 1] == "1235"
    assert cmd[cmd.index("--val_seed") + 1] == "1235"
    default = exp.make_jobs(dict(exp.DEFAULT_PLAN, budgets="775M", lrs="3e-3"))
    assert [j["name"] for j in default] == ["gpt_775M_lr0.003", "ngpt_775M_lr0.003"]   # имена как в старых прогонах


def _fake_run(root, name, model, lr, seed, final_loss, vals):
    d = root / "runs" / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(dict(model_name=model, tokens_per_step=100, n_params=1,
                                                   args=dict(lr=lr, seed=seed, max_iters=10))), encoding="utf-8")
    recs = [dict(type="val", step=i + 1, tokens=(i + 1) * 100, val_loss=v) for i, v in enumerate(vals)]
    recs += [dict(type="train", step=10, tokens=1000, loss=1.0, lr=lr, grad_norm=1.0, elapsed=1.0, tokens_per_sec=100.0),
             dict(type="final", val_loss=final_loss, perplexity=1.0, tokens=100, step=10, tokens_trained=1000,
                  elapsed=1.0)]
    (d / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def test_report_merges_sweep_from_several_people(tmp_path):
    """Прогоны разных участников (разные lr и сиды) в одной папке -> лучший lr по среднему, разброс сидов."""
    root = tmp_path / "all"
    _fake_run(root, "gpt_1K_lr0.003", "gpt", 3e-3, 42, 3.36, [4.0, 3.5, 3.36])
    _fake_run(root, "ngpt_1K_lr0.003", "ngpt", 3e-3, 42, 3.25, [3.9, 3.3, 3.25])
    _fake_run(root, "gpt_1K_lr0.0015", "gpt", 1.5e-3, 42, 3.34, [4.0, 3.5, 3.34])
    _fake_run(root, "ngpt_1K_lr0.0015", "ngpt", 1.5e-3, 42, 3.28, [3.9, 3.4, 3.28])
    _fake_run(root, "gpt_1K_lr0.0015_s43", "gpt", 1.5e-3, 43, 3.36, [4.0, 3.5, 3.36])
    _fake_run(root, "ngpt_1K_lr0.003_s43", "ngpt", 3e-3, 43, 3.27, [3.9, 3.3, 3.27])
    exp.report(root, quiet=True)
    text = (root / "report" / "report.md").read_text(encoding="utf-8")
    summary = json.loads((root / "report" / "summary.json").read_text())
    assert summary["best_lr"] == {"gpt": 0.0015, "ngpt": 0.003}
    assert "Разброс по сидам" in text and "сид 43" in text
    assert len(summary["curve_speedups"]) == 2           # по одному на сид


def test_accounts_split_balanced():
    """Распределение короткого свипа по 4 аккаунтам: у каждого аккаунта две GPU."""
    base = dict(exp.DEFAULT_PLAN, budgets="150M", lrs="3e-3")
    acc = {1: dict(lrs_gpt="none", lrs_ngpt="7.5e-4,1.5e-3"),
           3: dict(lrs_gpt="7.5e-4,1.5e-3", lrs_ngpt="3e-3"),
           4: dict(lrs_gpt="3e-3,6e-3,1.2e-2", lrs_ngpt="none")}
    names = {k: [j["name"] for j in exp.make_jobs(dict(base, **v))] for k, v in acc.items()}
    assert names[1] == ["ngpt_150M_lr0.00075", "ngpt_150M_lr0.0015"]
    assert names[3] == ["gpt_150M_lr0.00075", "ngpt_150M_lr0.003", "gpt_150M_lr0.0015"]
    assert names[4] == ["gpt_150M_lr0.003", "gpt_150M_lr0.006", "gpt_150M_lr0.012"]


def test_explicit_jobs_ladder():
    """--jobs: свой бюджет и lr у каждого прогона, длинные первыми (лестница бюджетов за одну сессию)."""
    plan = dict(exp.DEFAULT_PLAN, n_layer=6, n_head=6, d_model=384,
                jobs="ngpt:300M:4.5e-3, gpt:1.4B:3e-3, ngpt:450M:3.8e-3")
    jobs = exp.make_jobs(plan)
    assert [j["name"] for j in jobs] == ["gpt_1.4B_lr0.003", "ngpt_450M_lr0.0038", "ngpt_300M_lr0.0045"]
    assert jobs[0]["steps"] == round(1.4e9 / 65536) and all(j["lr"] for j in jobs)
    with pytest.raises(ValueError):
        exp.make_jobs(dict(plan, jobs="bert:1B:1e-3"))


def _fake_ctx(root, name, ctx):
    cfg_path = root / "runs" / name / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["args"]["block_size"] = ctx
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")


def test_report_budget_ladder_both_directions(tmp_path):
    """Лестница бюджетов: на каждом бюджете лучший lr (среднее по сидам), ускорение в обе стороны,
    прогоны с другим контекстом не смешиваются."""
    root = tmp_path / "all"

    def run(name, model, lr, seed, final, max_iters):
        _fake_run(root, name, model, lr, seed, final, [final + 0.5, final + 0.1, final])
        cfg_path = root / "runs" / name / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["args"]["max_iters"] = max_iters      # 100 токенов на шаг
        cfg["args"]["block_size"] = 1024
        cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    run("gpt_1K_lr0.003", "gpt", 3e-3, 42, 3.74, 10)
    run("gpt_1K_lr0.006", "gpt", 6e-3, 42, 3.78, 10)
    run("ngpt_1K_lr0.006", "ngpt", 6e-3, 42, 3.60, 10)
    run("ngpt_2K_lr0.0045", "ngpt", 4.5e-3, 42, 3.44, 20)
    run("ngpt_3K_lr0.0038", "ngpt", 3.8e-3, 42, 3.37, 30)
    run("gpt_5K_lr0.003", "gpt", 3e-3, 42, 3.36, 50)
    run("gpt_5K_lr0.003_s43", "gpt", 3e-3, 43, 3.38, 50)
    run("ngpt_5K_lr0.003", "ngpt", 3e-3, 42, 3.25, 50)
    run("gpt_10K_lr0.003", "gpt", 3e-3, 42, 3.22, 100)
    run("gpt_5K_lr0.003_s44", "gpt", 3e-3, 44, 3.10, 50)
    _fake_ctx(root, "gpt_5K_lr0.003_s44", 2048)      # другой контекст — в отчёт не попадает
    exp.report(root, quiet=True)
    text = (root / "report" / "report.md").read_text(encoding="utf-8")
    summary = json.loads((root / "report" / "summary.json").read_text())
    assert "пропущены: gpt_5K_lr0.003_s44" in text
    assert summary["final"]["gpt"] == [[1000, 3.74], [5000, pytest.approx(3.37)], [10000, 3.22]]
    assert [t for t, _ in summary["final"]["ngpt"]] == [1000, 2000, 3000, 5000]
    # GPT на 5K (3.37 в среднее по 2 сидам) — nGPT дошёл до этого на 3K -> ускорение 5/3
    rev = {d["tokens"]: d for d in summary["speedups_reverse"]}
    assert rev[5000]["sign"] == "=" and rev[5000]["ratio"] == pytest.approx(5000 / 3000)
    # nGPT на 5K (3.25): GPT — между 5K и 10K
    fwd = {d["tokens"]: d for d in summary["speedups"]}
    assert fwd[5000]["sign"] == "=" and 1.0 < fwd[5000]["ratio"] < 2.0
    assert summary["best_lr"] == {"gpt": 0.003, "ngpt": 0.003}
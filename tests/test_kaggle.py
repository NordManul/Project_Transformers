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
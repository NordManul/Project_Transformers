"""
Тесты compare.py: обучение всех трёх моделей с логированием, eval и графики.
Токенизатор подменяется заглушкой, сеть не нужна, всё на CPU.

    pytest tests/test_compare.py -v
"""
import json
import sys

import pytest
import torch

import compare


class DummyTokenizer:
    vocab_size, bos_token_id, eos_token_id = 200, 1, 2

    def encode(self, text, add_special_tokens=False):
        return [3 + ord(c) % 190 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return ""


@pytest.fixture
def data_file(tmp_path, monkeypatch):
    monkeypatch.setattr(compare.bg, "load_tokenizer", lambda: DummyTokenizer())
    path = tmp_path / "data.txt"
    path.write_text("the quick brown fox jumps over the lazy dog. " * 300, encoding="utf-8")
    return path


def run_cli(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["compare.py", *map(str, argv)])
    compare.main()


def train(monkeypatch, tmp_path, data_file, model, *extra):
    run_cli(monkeypatch, "train", "--model", model, "--data", data_file, "--out", tmp_path / "runs",
            "--n_layer", 2, "--n_head", 4, "--d_model", 32, "--block_size", 16, "--batch_size", 4,
            "--max_iters", 6, "--threads", 2, "--log_every", 2, "--stats_every", 3, "--eval_every", 3,
            "--eval_batches", 2, "--device", "cpu", *extra)
    run = tmp_path / "runs" / model
    records = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    return run, records


@pytest.mark.parametrize("model", compare.MODELS)
def test_train_writes_all_metrics(model, tmp_path, monkeypatch, data_file):
    run, records = train(monkeypatch, tmp_path, data_file, model)
    types = [r["type"] for r in records]
    assert types.count("final") == 1 and "train" in types and "val" in types and "stats" in types
    stats = [r for r in records if r["type"] == "stats"]
    assert [s["step"] for s in stats] == [1, 3, 6]
    for s in stats:
        for part in ("block", "attn", "mlp"):
            assert len(s["activations"][part]) == 2
            assert all(a["absmax"] > 0 and a["outlier"] >= 1 for a in s["activations"][part])
        assert len(s["grads"]["layers"]) == 2 and all(g > 0 for g in s["grads"]["layers"])
    special = stats[-1]["special"]
    assert ("lambda_" in special) == (model == "diff")
    assert ("s_z" in special) == (model == "ngpt")
    if model == "ngpt":   # выход блока nGPT лежит на единичной сфере
        assert all(abs(a["norm"] - 1) < 1e-3 for a in stats[-1]["activations"]["block"])
    final = records[-1]
    assert final["type"] == "final" and final["tokens_trained"] == 6 * 4 * 16
    assert (run / "model.pt").exists() and (run / "config.json").exists()


def test_same_batches_for_all_models(tmp_path, monkeypatch, data_file):
    """Обучающие батчи и val-набор задаются --data_seed и не зависят ни от модели, ни от --seed."""
    seen = {}
    original_build = compare.build_model

    def spy_build(name, *args, **kwargs):
        built = original_build(name, *args, **kwargs)
        seen[name] = []
        built[0].register_forward_pre_hook(
            lambda module, inputs: seen[name].append(inputs[0].clone()) if module.training else None)
        return built

    monkeypatch.setattr(compare, "build_model", spy_build)
    train(monkeypatch, tmp_path, data_file, "gpt", "--seed", 7)
    train(monkeypatch, tmp_path, data_file, "ngpt", "--seed", 99)
    assert len(seen["gpt"]) == len(seen["ngpt"]) == 6
    assert all(torch.equal(a, b) for a, b in zip(seen["gpt"], seen["ngpt"]))


def test_eval_on_held_out_file(tmp_path, monkeypatch, data_file, capsys):
    run, records = train(monkeypatch, tmp_path, data_file, "diff")
    val_file = tmp_path / "val.txt"
    val_file.write_text("the quick brown fox jumps over the lazy dog. " * 30, encoding="utf-8")
    run_cli(monkeypatch, "eval", run, "--data", val_file, "--device", "cpu")
    result = json.loads((run / "eval.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert result["tokens"] > 0 and result["val_loss"] > 0


def test_plot_creates_figures(tmp_path, monkeypatch, data_file):
    for model in compare.MODELS:
        train(monkeypatch, tmp_path, data_file, model)
    out = tmp_path / "plots"
    run_cli(monkeypatch, "plot", *(tmp_path / "runs" / m for m in compare.MODELS), "--out", out)
    for name in ("loss.png", "grad_norm.png", "layers_over_time.png", "layer_profile.png",
                 "architecture_params.png", "summary.md"):
        assert (out / name).stat().st_size > 0, name
    assert "| diff |" in (out / "summary.md").read_text(encoding="utf-8")
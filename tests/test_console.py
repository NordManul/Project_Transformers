"""Обучение с checkpoint, продолжение и пользовательский консольный сценарий."""
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from baseline_gpt import GPT, ModelConfig


ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="2")


def run(script, arguments, cwd, user_input=None):
    result = subprocess.run([sys.executable, str(ROOT / script), *arguments], cwd=cwd,
                            env=ENV, input=user_input, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_train_resume_and_interactive_console(tmp_path):
    train = tmp_path / "train.txt"
    val = tmp_path / "val.txt"
    train.write_text("Once upon a time, a little cat played with a ball.\n" * 100)
    val.write_text("One day, a little dog found a ball in the park.\n" * 50)
    checkpoint = tmp_path / "model.pt"
    arguments = ["--data", str(train), "--val_data", str(val), "--device", "cpu",
                 "--n_layer", "1", "--n_head", "2", "--d_model", "16", "--block_size", "16",
                 "--batch_size", "2", "--warmup", "0", "--eval_every", "3", "--eval_iters", "1",
                 "--save_every", "2", "--generate_tokens", "0", "--checkpoint", str(checkpoint),
                 "--log_file", str(tmp_path / "metrics.jsonl")]
    run("baseline_gpt.py", arguments + ["--max_iters", "3"], tmp_path)
    first = torch.load(checkpoint, weights_only=True)
    assert first["step"] == 3
    assert first["config"]["d_model"] == 16 and first["config"]["block_size"] == 16
    assert first["optimizer"]["state"]
    assert (tmp_path / "model.best.pt").is_file()
    resumed = run("baseline_gpt.py", arguments + ["--resume", str(checkpoint), "--max_iters", "6"], tmp_path)
    last = torch.load(checkpoint, weights_only=True)
    assert last["step"] == 6
    assert "Продолжаю с шага 3" in resumed.stdout
    assert not torch.equal(first["model"]["emb_in.weight"], last["model"]["emb_in.weight"])
    assert all(state["step"].item() == 6 for state in last["optimizer"]["state"].values())

    console_args = ["--checkpoint", str(checkpoint), "--device", "cpu", "--max_tokens", "2"]
    one_shot = run("interactive.py", console_args + ["--prompt", "Once upon a time", "--trace"], tmp_path)
    assert "token=" in one_shot.stdout and "варианты:" in one_shot.stdout
    assert "Модель> Once upon a time" in one_shot.stdout
    console = run("interactive.py", console_args, tmp_path,
                  "/trace on\nOne day\n/trace off\nOnce upon a time\n/exit\n")
    assert "Просмотр токенов включён." in console.stdout
    assert "Просмотр токенов выключен." in console.stdout
    assert console.stdout.count("Модель>") == 2


@pytest.mark.parametrize("options", [{"temperature": 0}, {"top_k": 0}, {"max_new_tokens": -1}])
def test_generation_rejects_invalid_settings(options):
    model = GPT(ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16, block_size=8)).eval()
    arguments = dict(max_new_tokens=3, temperature=0.8, top_k=5)
    arguments.update(options)
    with pytest.raises(ValueError):
        list(model.generate_tokens(torch.ones((1, 1), dtype=torch.long), **arguments))


def test_token_stream_stops_on_eos_and_has_no_gradient(monkeypatch):
    model = GPT(ModelConfig(vocab_size=32, n_layer=1, n_head=2, d_model=16, block_size=8)).eval()

    def forward(ids):
        assert not torch.is_grad_enabled()
        logits = torch.full((*ids.shape, 32), -100.0)
        logits[:, :, 2] = 100.0
        return logits, None

    monkeypatch.setattr(model, "forward", forward)
    tokens = list(model.generate_tokens(torch.ones((1, 1), dtype=torch.long), 10, eos_token_id=2))
    assert len(tokens) == 1 and tokens[0][0].item() == 2
    assert tokens[0][1].sum().item() == pytest.approx(1)


@pytest.mark.parametrize("options", [{"n_head": 0}, {"d_model": 15}, {"dropout": 1}, {"n_layer": -1}])
def test_model_config_validation(options):
    arguments = dict(vocab_size=32, n_layer=1, n_head=2, d_model=16, block_size=8)
    arguments.update(options)
    with pytest.raises(ValueError):
        ModelConfig(**arguments)

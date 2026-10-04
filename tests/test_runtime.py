"""Проверка baseline_gpt.py: обучение, разные конфигурации и запуск CLI без сети."""
import importlib
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch


@pytest.fixture(scope="module")
def implementation():
    return importlib.import_module("baseline_gpt")


@pytest.mark.parametrize("settings", [
    dict(n_layer=1, n_head=2, d_model=16, block_size=8),
    dict(n_layer=3, n_head=4, d_model=32, block_size=16, mlp_ratio=2,
         dropout=0.1, tie_embeddings=True),
    dict(n_layer=2, n_head=3, d_model=24, block_size=24, rope_base=5000),
])
def test_custom_model_learns_and_weights_can_be_restored(implementation, settings, tmp_path):
    torch.manual_seed(7)
    cfg = implementation.ModelConfig(vocab_size=64, **settings)
    model = implementation.GPT(cfg)
    optimizer = implementation.make_optimizer(model, lr=0.01, weight_decay=0)
    data = torch.randint(cfg.vocab_size, (2, cfg.block_size + 1))
    # Нарезка батча даёт targets с разрывами в памяти: обычный .view здесь не работает.
    x, y = data[:, :-1], data[:, 1:]
    assert not y.is_contiguous()
    model.eval()
    with torch.no_grad():
        _, initial_loss = model(x, y)
    model.train()
    for _ in range(80):
        _, loss = model(x, y)
        assert torch.isfinite(loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters())
        optimizer.step()

    model.eval()
    with torch.no_grad():
        logits, final_loss = model(x, y)
    assert logits.shape == (2, cfg.block_size, cfg.vocab_size)
    assert final_loss < initial_loss * 0.2
    if cfg.tie_embeddings:
        assert model.emb_in.weight is model.emb_out.weight

    checkpoint = tmp_path / "weights.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = implementation.GPT(cfg).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        restored_logits, _ = restored(x)
    torch.testing.assert_close(logits, restored_logits)


def test_attention_is_causal_with_dropout_configured(implementation):
    torch.manual_seed(1)
    cfg = implementation.ModelConfig(vocab_size=64, block_size=16, n_layer=2,
                                     n_head=4, d_model=32, dropout=0.2)
    model = implementation.GPT(cfg).eval()
    original = torch.randint(cfg.vocab_size, (2, cfg.block_size))
    changed = original.clone()
    changed[:, 8:] = (changed[:, 8:] + 1) % cfg.vocab_size
    with torch.no_grad():
        logits, _ = model(original)
        changed_logits, _ = model(changed)
    torch.testing.assert_close(logits[:, :8], changed_logits[:, :8])
    assert not torch.allclose(logits[:, 8:], changed_logits[:, 8:])


def test_generation_can_exceed_context_length(implementation):
    cfg = implementation.ModelConfig(vocab_size=64, block_size=8, n_layer=1,
                                     n_head=2, d_model=16)
    model = implementation.GPT(cfg).eval()
    output = model.generate(torch.ones((2, 3), dtype=torch.long),
                            max_new_tokens=12, temperature=0.8, top_k=5)
    assert output.shape == (2, 15)
    assert output.min() >= 0 and output.max() < cfg.vocab_size


def test_cli_trains_on_a_file_offline_and_reuses_token_cache(implementation, tmp_path):
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "training.txt"
    data.write_text("Привет, мир! Мы обучаем небольшой трансформер на тексте.\n" * 300,
                    encoding="utf-8")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
    command = [sys.executable, str(root / f"{implementation.__name__}.py"),
               "--preset", "tiny", "--data", str(data), "--max_iters", "8",
               "--warmup", "1", "--batch_size", "1", "--eval_every", "4",
               "--lr", "0.003"]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                            capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "не найден" not in result.stdout
    assert "Сгенерированный текст" in result.stdout
    losses = re.findall(r"train ([0-9.]+) \| val ([0-9.]+)", result.stdout)
    assert len(losses) == 3, result.stdout
    assert float(losses[-1][0]) < float(losses[0][0])
    assert float(losses[-1][1]) < float(losses[0][1])
    assert Path(f"{data}.mistral.pt").is_file()
    # Повторный запуск подтверждает, что кэш читается и пути не зависят от cwd.
    cached = subprocess.run(command, cwd=tmp_path, env=env, text=True,
                            capture_output=True, timeout=120)
    assert cached.returncode == 0, cached.stdout + cached.stderr
    assert "Токены загружены из кэша" in cached.stdout

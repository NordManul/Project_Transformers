"""
Тесты сохранения модели и интерактивной генерации. Сеть и данные не нужны.

    pytest tests/test_chat.py -v
"""
import io

import pytest
import torch

from diff_transformer import DiffConfig as ModelConfig, DiffGPT as GPT, load_checkpoint, save_checkpoint
from chat import complete, generate_stream


def small_model(**overrides):
    cfg = dict(vocab_size=64, block_size=16, n_layer=2, n_head=4, d_model=32)
    cfg.update(overrides)
    torch.manual_seed(0)
    return GPT(ModelConfig(**cfg)).eval()


def test_checkpoint_roundtrip(tmp_path):
    model = small_model()
    path = str(tmp_path / "sub" / "model.pt")
    save_checkpoint(path, model, step=7, val_loss=1.5, preset="test", data="x.txt")
    loaded, ckpt = load_checkpoint(path)
    assert ckpt["step"] == 7 and ckpt["val_loss"] == 1.5 and ckpt["preset"] == "test"
    assert loaded.cfg == model.cfg
    assert not loaded.training
    idx = torch.randint(64, (2, 10))
    assert torch.equal(model(idx)[0], loaded(idx)[0])


def test_stream_length_and_vocab():
    model = small_model()
    out = list(generate_stream(model, [1, 2, 3], max_new_tokens=10))
    assert len(out) == 10
    assert all(0 <= t < 64 for t in out)


def test_stream_crops_context_longer_than_block_size():
    model = small_model()
    out = list(generate_stream(model, list(range(30)), max_new_tokens=25))   # 30 + 25 > 16
    assert len(out) == 25


def test_greedy_is_deterministic_and_matches_argmax():
    model = small_model()
    ids = [5, 6, 7]
    a = list(generate_stream(model, ids, 5, temperature=0))
    b = list(generate_stream(model, ids, 5, temperature=0))
    assert a == b
    with torch.no_grad():
        first = int(model(torch.tensor([ids]))[0][0, -1].argmax())
    assert a[0] == first


def test_top_k_one_equals_greedy():
    model = small_model()
    torch.manual_seed(1)
    sampled = list(generate_stream(model, [3, 4], 6, temperature=1.0, top_k=1))
    assert sampled == list(generate_stream(model, [3, 4], 6, temperature=0))


class AlwaysToken(torch.nn.Module):
    """Модель-заглушка, которая всегда уверенно предсказывает один и тот же токен."""
    def __init__(self, token, vocab=64):
        super().__init__()
        self.cfg = ModelConfig(vocab_size=vocab, block_size=16, n_layer=1, n_head=1, d_model=4)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.token, self.vocab = token, vocab

    def forward(self, idx, targets=None):
        logits = torch.full((*idx.shape, self.vocab), -10.0)
        logits[..., self.token] = 10.0
        return logits, None


def test_stops_on_eos():
    assert list(generate_stream(AlwaysToken(2), [9], 10, temperature=0, eos_id=2)) == []
    assert list(generate_stream(AlwaysToken(5), [9], 4, temperature=0.8, top_k=40, eos_id=2)) == [5] * 4


class FakeTok:
    """Токенизатор-заглушка: один символ = один токен."""
    bos_token_id, eos_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 60 + 3 for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(ord("a") + (i - 3) % 26) for i in ids if i > 2)


def test_complete_prints_prompt_and_continuation():
    model = small_model()
    buf = io.StringIO()
    result = complete(model, FakeTok(), "abc", max_new_tokens=8, temperature=0, top_k=0, out=buf)
    printed = buf.getvalue()
    assert printed.endswith("\n")
    assert printed.rstrip("\n") == result
    assert len(result) >= len(FakeTok().decode(FakeTok().encode("abc")))

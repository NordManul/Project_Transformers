"""
Тесты Differential Transformer (Ye et al., ICLR 2025, arXiv:2410.05258). Без сети, на CPU.

    pytest tests/test_diff.py -v
"""
import math

import pytest
import torch

from diff_transformer import (
    DiffAttention, DiffConfig as ModelConfig, DiffGPT as GPT, apply_rope, lambda_init_fn, load_checkpoint,
    rope_cache, save_checkpoint,
)
from compare import ar_hit_mask


def cfg(**kw):
    base = dict(vocab_size=64, block_size=16, n_layer=3, n_head=4, d_model=32, arch="diff")
    base.update(kw)
    return ModelConfig(**base)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def test_lambda_init_schedule():
    assert lambda_init_fn(0) == pytest.approx(0.2)                    # l = 1: 0.8 - 0.6
    assert lambda_init_fn(1) == pytest.approx(0.8 - 0.6 * math.exp(-0.3))
    assert lambda_init_fn(100) == pytest.approx(0.8, abs=1e-6)
    model = GPT(cfg())
    assert [b.attn.lambda_init for b in model.blocks] == [lambda_init_fn(i) for i in range(3)]
    model = GPT(cfg(lambda_init=0.5))
    assert all(b.attn.lambda_init == 0.5 for b in model.blocks)


def test_shapes_and_heads():
    model = GPT(cfg())
    attn = model.blocks[0].attn
    assert isinstance(attn, DiffAttention)
    assert (attn.h, attn.d) == (2, 8)                 # h = n_head / 2, d = d_model / n_head
    logits, loss = model(torch.randint(64, (2, 16)), torch.randint(64, (2, 16)))
    assert logits.shape == (2, 16, 64) and torch.isfinite(loss)


def test_same_params_as_baseline_plus_lambda_and_norm():
    """Как в статье: параметры почти совпадают с обычным трансформером.
    Разница на слой — 4 вектора λ размерности d и веса RMSNorm размерности 2d."""
    c = cfg()
    n_diff = sum(p.numel() for p in GPT(c).parameters())
    n_base = sum(p.numel() for p in GPT(cfg(arch="baseline")).parameters())
    d = c.d_model // c.n_head
    assert n_diff - n_base == c.n_layer * (4 * d + 2 * d)


def test_causal():
    model = GPT(cfg()).eval()
    x = torch.randint(64, (1, 12))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 64
    a, _ = model(x)
    b, _ = model(y)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])


def reference_diff_attention(attn, x, cos, sin):
    """Прямая реализация ур. 1-3 статьи с явными матрицами внимания."""
    B, T, C = x.shape
    h, d = attn.h, attn.d
    q = apply_rope(attn.w_q(x).view(B, T, 2 * h, d).transpose(1, 2), cos, sin)
    k = apply_rope(attn.w_k(x).view(B, T, 2 * h, d).transpose(1, 2), cos, sin)
    v = attn.w_v(x).view(B, T, h, 2 * d).transpose(1, 2)
    mask = torch.full((T, T), float("-inf")).triu(1)
    lam = (torch.exp(attn.lambda_q1 @ attn.lambda_k1) - torch.exp(attn.lambda_q2 @ attn.lambda_k2)
           + attn.lambda_init)
    heads = []
    for i in range(h):
        q1, q2, k1, k2 = q[:, 2 * i], q[:, 2 * i + 1], k[:, 2 * i], k[:, 2 * i + 1]
        a1 = torch.softmax(q1 @ k1.transpose(-1, -2) / math.sqrt(d) + mask, dim=-1)
        a2 = torch.softmax(q2 @ k2.transpose(-1, -2) / math.sqrt(d) + mask, dim=-1)
        head = (a1 - lam * a2) @ v[:, i]
        if attn.head_norm is not None:
            head = attn.head_norm(head)
        heads.append(head * (1 - attn.lambda_init))
    return attn.w_o(torch.cat(heads, dim=-1))


@pytest.mark.parametrize("head_norm", [None, False])
def test_matches_paper_equations(head_norm):
    c = cfg(head_norm=head_norm)
    model = GPT(c).eval()
    attn = model.blocks[1].attn
    x = torch.randn(2, 10, c.d_model)
    cos, sin = rope_cache(10, c.d_model // c.n_head, c.rope_base)
    with torch.no_grad():
        ours = attn(x, cos, sin)
        ref = reference_diff_attention(attn, x, cos, sin)
    assert torch.allclose(ours, ref, atol=1e-5)


def test_lambda_starts_near_lambda_init():
    attn = GPT(cfg()).blocks[2].attn
    assert abs(attn.lam().item() - attn.lambda_init) < 0.1


def test_baseline_with_groupnorm_option():
    model = GPT(cfg(arch="baseline", head_norm=True))
    assert model.blocks[0].attn.head_norm is not None
    logits, _ = model(torch.randint(64, (1, 8)))
    assert logits.shape == (1, 8, 64)


def test_diff_learns(tmp_path):
    """Несколько шагов на одном батче: loss должен заметно упасть, λ — получать градиент."""
    model = GPT(cfg())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x = torch.randint(64, (4, 16))
    first = None
    for _ in range(60):
        _, loss = model(x[:, :-1], x[:, 1:])
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        assert model.blocks[0].attn.lambda_q1.grad is not None
        opt.step()
    assert loss.item() < first * 0.5
    path = str(tmp_path / "diff.pt")
    save_checkpoint(path, model, step=1)
    loaded, ckpt = load_checkpoint(path)
    assert ckpt["config"]["arch"] == "diff"
    assert torch.allclose(model.eval()(x)[0], loaded(x)[0], atol=1e-6)


def test_ar_hit_mask():
    # окно [1, 2, 3, 1, 2]: предсказываются 2, 3, 1, 2; биграмма (1, 2) повторилась на последнем
    assert ar_hit_mask([1, 2, 3, 1, 2], n=2) == [False, False, False, True]
    assert ar_hit_mask([5, 5, 5], n=1) == [False, True]

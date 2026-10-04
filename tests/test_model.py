"""
Тесты модели. Данные и сеть не нужны, всё работает на CPU за десятки секунд.

    pytest tests/test_model.py -v
"""
import math

import pytest
import torch
import torch.nn.functional as F

from baseline_gpt import (
    GPT, ModelConfig, PRESETS, RMSNorm, apply_rope, lr_at, make_optimizer, rope_cache,
)


def tiny_config(**overrides):
    cfg = dict(vocab_size=64, block_size=16, n_layer=2, n_head=4, d_model=32)
    cfg.update(overrides)
    return ModelConfig(**cfg)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


# ------------------------- Форма и базовые свойства -------------------------
def test_forward_shapes():
    cfg = tiny_config()
    model = GPT(cfg)
    idx = torch.randint(cfg.vocab_size, (3, cfg.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, cfg.block_size, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_shorter_sequence_than_block_size():
    cfg = tiny_config()
    model = GPT(cfg)
    logits, _ = model(torch.randint(cfg.vocab_size, (2, 5)))
    assert logits.shape == (2, 5, cfg.vocab_size)


def test_too_long_sequence_raises():
    cfg = tiny_config()
    model = GPT(cfg)
    with pytest.raises(AssertionError):
        model(torch.randint(cfg.vocab_size, (1, cfg.block_size + 1)))


def test_no_biases_anywhere():
    model = GPT(tiny_config())
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            assert module.bias is None, f"{name} не должен иметь bias"


def test_embeddings_are_untied_by_default():
    model = GPT(tiny_config())
    assert model.emb_in.weight.data_ptr() != model.emb_out.weight.data_ptr()


def test_initial_loss_close_to_uniform():
    """Сразу после инициализации предсказание почти равномерное: loss ≈ ln(V)."""
    cfg = tiny_config(vocab_size=32000, d_model=64)
    model = GPT(cfg).eval()
    idx = torch.randint(cfg.vocab_size, (4, cfg.block_size))
    with torch.no_grad():
        _, loss = model(idx, idx)
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 0.1


def test_output_projection_init_is_scaled():
    """A.6: std выходных проекций = 0.02 / sqrt(2 * n_layer)."""
    cfg = tiny_config(d_model=256, n_layer=8)
    model = GPT(cfg)
    expected = 0.02 / math.sqrt(2 * cfg.n_layer)
    std_attn_o = model.blocks[0].attn.w_o.weight.std().item()
    std_mlp_o = model.blocks[0].mlp.w_o.weight.std().item()
    std_q = model.blocks[0].attn.w_q.weight.std().item()
    assert abs(std_attn_o - expected) / expected < 0.05
    assert abs(std_mlp_o - expected) / expected < 0.05
    assert abs(std_q - 0.02) / 0.02 < 0.05


# ------------------------- Каузальность -------------------------
def test_causality():
    """Изменение токенов начиная с позиции p не влияет на логиты на позициях < p."""
    cfg = tiny_config()
    model = GPT(cfg).eval()
    p = 10
    a = torch.randint(cfg.vocab_size, (1, cfg.block_size))
    b = a.clone()
    b[0, p:] = (b[0, p:] + 1) % cfg.vocab_size
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    torch.testing.assert_close(la[:, :p], lb[:, :p])
    assert not torch.allclose(la[:, p], lb[:, p])


# ------------------------- Компоненты -------------------------
def test_rmsnorm_gives_unit_rms():
    norm = RMSNorm(64)
    x = torch.randn(5, 7, 64) * 13 + 2
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    torch.testing.assert_close(rms, torch.ones_like(rms), atol=1e-4, rtol=0)


def test_rope_depends_only_on_relative_position():
    """Скалярное произведение q·k после RoPE зависит только от (m - n)."""
    head_dim, T = 16, 32
    cos, sin = rope_cache(T, head_dim, 10000.0)
    q = torch.randn(head_dim)
    k = torch.randn(head_dim)

    def score(m, n):
        qm = apply_rope(q.expand(T, head_dim), cos, sin)[m]
        kn = apply_rope(k.expand(T, head_dim), cos, sin)[n]
        return (qm * kn).sum()

    torch.testing.assert_close(score(5, 2), score(20, 17), atol=1e-5, rtol=0)
    torch.testing.assert_close(score(9, 9), score(0, 0), atol=1e-5, rtol=0)


def test_rope_preserves_norm():
    head_dim, T = 16, 32
    cos, sin = rope_cache(T, head_dim, 10000.0)
    x = torch.randn(2, 3, T, head_dim)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1))


# ------------------------- Размер моделей из статьи -------------------------
@pytest.mark.parametrize("preset, expected_millions", [("0.5B", 468.2), ("1B", 1025.7)])
def test_parameter_count_matches_paper_table2(preset, expected_millions):
    """Создаём модель на meta-устройстве (без памяти) и сверяем с таблицей 2."""
    cfg = ModelConfig(vocab_size=32000, **PRESETS[preset])
    with torch.device("meta"):
        model = GPT(cfg)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    assert round(n, 1) == expected_millions


# ------------------------- Сверка с эталонной LLaMA из transformers -------------------------
def test_matches_huggingface_llama():
    """Копируем веса в LlamaForCausalLM с той же конфигурацией: логиты должны совпасть."""
    transformers = pytest.importorskip("transformers")
    cfg = tiny_config()
    ours = GPT(cfg).double().eval()

    hf_cfg = transformers.LlamaConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.d_model,
        intermediate_size=cfg.mlp_ratio * cfg.d_model,
        num_hidden_layers=cfg.n_layer,
        num_attention_heads=cfg.n_head,
        num_key_value_heads=cfg.n_head,
        max_position_embeddings=cfg.block_size,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
        hidden_act="silu",
    )  # база RoPE по умолчанию 10000, как в статье
    hf = transformers.LlamaForCausalLM(hf_cfg).double().eval()

    with torch.no_grad():
        hf.model.embed_tokens.weight.copy_(ours.emb_in.weight)
        hf.lm_head.weight.copy_(ours.emb_out.weight)
        hf.model.norm.weight.copy_(ours.norm_f.weight)
        for blk, layer in zip(ours.blocks, hf.model.layers):
            layer.input_layernorm.weight.copy_(blk.norm_attn.weight)
            layer.post_attention_layernorm.weight.copy_(blk.norm_mlp.weight)
            layer.self_attn.q_proj.weight.copy_(blk.attn.w_q.weight)
            layer.self_attn.k_proj.weight.copy_(blk.attn.w_k.weight)
            layer.self_attn.v_proj.weight.copy_(blk.attn.w_v.weight)
            layer.self_attn.o_proj.weight.copy_(blk.attn.w_o.weight)
            # HF: down(silu(gate(x)) * up(x)); у нас: w_o(w_u(x) * silu(w_v(x)))
            layer.mlp.gate_proj.weight.copy_(blk.mlp.w_v.weight)
            layer.mlp.up_proj.weight.copy_(blk.mlp.w_u.weight)
            layer.mlp.down_proj.weight.copy_(blk.mlp.w_o.weight)

    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in hf.parameters())

    idx = torch.randint(cfg.vocab_size, (2, cfg.block_size))
    with torch.no_grad():
        our_logits, _ = ours(idx)
        hf_logits = hf(idx).logits
    torch.testing.assert_close(our_logits, hf_logits, atol=1e-8, rtol=1e-8)


# ------------------------- Обучение -------------------------
def test_overfits_single_batch():
    """На одном фиксированном батче loss должен упасть почти до нуля."""
    cfg = tiny_config(d_model=64)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    data = torch.randint(cfg.vocab_size, (4, cfg.block_size + 1))
    x, y = data[:, :-1], data[:, 1:]
    first = None
    for _ in range(300):
        _, loss = model(x, y)
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, f"loss {first:.3f} -> {loss.item():.3f}"


def test_optimizer_groups():
    """Weight decay 0.1 на матрицы и эмбеддинги, 0 на веса RMSNorm."""
    model = GPT(tiny_config())
    opt = make_optimizer(model, lr=1e-3)
    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])
    n_norms = 2 * model.cfg.n_layer + 1
    assert len(no_decay["params"]) == n_norms


def test_lr_schedule_warmup_then_cosine_to_zero():
    base, warmup, total = 1e-3, 100, 1000
    lrs = [lr_at(s, base, warmup, total) for s in range(total + 1)]
    assert lrs[0] == pytest.approx(base / warmup)
    assert max(lrs) == pytest.approx(base)
    assert all(a <= b for a, b in zip(lrs[:warmup], lrs[1:warmup + 1]))   # рост
    assert all(a >= b for a, b in zip(lrs[warmup:], lrs[warmup + 1:]))     # спад
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)


def test_generate_appends_tokens():
    cfg = tiny_config()
    model = GPT(cfg).eval()
    start = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(start, max_new_tokens=40, top_k=5)   # длиннее block_size
    assert out.shape == (1, 41)
    assert out.max() < cfg.vocab_size
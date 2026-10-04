"""
Differential Transformer — «Differential Transformer» (Tianzhu Ye, Li Dong et al.,
Microsoft Research / Tsinghua University, ICLR 2025, arXiv:2410.05258).

Модуль использует базовый трансформер из корня репозитория (../baseline_gpt.py) без изменений:
оттуда берутся GPT, ModelConfig, RMSNorm, RoPE, SwiGLU, токенизатор и расписание lr.
Здесь добавлено только новое:

    DiffConfig     ModelConfig + поля arch / head_norm / lambda_init
    DiffAttention  дифференциальное внимание (ур. 1-3 статьи)
    BaselineAttention  обычное внимание из baseline_gpt + опциональный GroupNorm на голову
                       (вариант «Transformer + GroupNorm» из табл. 6) и запись логитов для анализа
    DiffGPT        GPT из baseline_gpt, в котором блоки используют эти виды внимания

    [Q1; Q2] = X W_Q,  [K1; K2] = X W_K,  V = X W_V
    DiffAttn(X) = (softmax(Q1 K1^T / sqrt(d)) - λ softmax(Q2 K2^T / sqrt(d))) V        (ур. 1)
    λ = exp(λq1·λk1) - exp(λq2·λk2) + λ_init,  λ_init = 0.8 - 0.6·exp(-0.3·(l-1))     (ур. 2)
    head_i = (1 - λ_init) · GroupNorm(DiffAttn_i(X)),  MultiHead = Concat(head_i) W_O  (ур. 3)

Проверка, что всё работает (формулы статьи, каузальность, параметры, обучение, генерация):
    python diff_transformer.py
"""
import math
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn

# базовый трансформер лежит в корне репозитория (на уровень выше этой папки)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)   # в конец: модули этой папки имеют приоритет

import torch.nn.functional as F  # noqa: E402

from baseline_gpt import (  # noqa: E402,F401  — реэкспорт для остальных скриптов папки
    GPT, PRESETS as BASE_PRESETS, Attention, ModelConfig, RMSNorm, SwiGLUMLP, apply_rope,
    encode_text, load_tokenizer, lr_at, make_optimizer, rope_cache,
)

PRESETS = {**BASE_PRESETS,
           "small": dict(n_layer=6, n_head=6, d_model=384, block_size=256)}   # ~39M, для одной GPU


@dataclass
class DiffConfig(ModelConfig):
    arch: str = "baseline"              # "baseline" — обычное внимание, "diff" — дифференциальное
    head_norm: Optional[bool] = None    # GroupNorm на голову; None: да для diff, нет для baseline
    lambda_init: Optional[float] = None  # None: λ_init по слоям (ур. 2), иначе константа


def lambda_init_fn(layer_idx):
    """λ_init = 0.8 - 0.6·exp(-0.3·(l-1)), l = 1..L (раздел 2.1 статьи); layer_idx = l - 1."""
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


def record_attention_logits(module, q, k):
    """Для анализа выбросов (табл. 5 статьи): сохраняет самые большие по модулю логиты внимания
    q·k/sqrt(d) и их медиану. Работает, только если module.record — список."""
    if getattr(module, "record", None) is None:
        return
    with torch.no_grad():
        T = q.shape[-2]
        scores = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        vals = scores[..., mask].abs().flatten()
        module.record.append((vals.topk(min(100, vals.numel())).values.cpu(), vals.median().item()))


class BaselineAttention(Attention):
    """Attention из baseline_gpt.py; head_norm=True — вариант «Transformer + GroupNorm» (табл. 6)."""
    def __init__(self, cfg, layer_idx=0):
        super().__init__(cfg)
        self.head_norm = RMSNorm(self.d_k, eps=1e-5) if cfg.head_norm else None
        self.record = None

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.w_q(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        record_attention_logits(self, q, k)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        if self.head_norm is not None:
            y = self.head_norm(y)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.w_o(y))


class DiffAttention(nn.Module):
    """Многоголовое дифференциальное внимание (Ye et al., 2025, ур. 1-3, рис. 2).

    Q и K каждой головы делятся на две половины (Q1, Q2), (K1, K2) размерности d, V имеет размерность 2d:
        DiffAttn(X) = (softmax(Q1 K1^T / sqrt(d)) - λ softmax(Q2 K2^T / sqrt(d))) V
        λ = exp(λq1·λk1) - exp(λq2·λk2) + λ_init                                     (ур. 2)
        head_i = (1 - λ_init) · RMSNorm(DiffAttn_i(X))                               (ур. 3)
    Число голов h = n_head / 2 при той же размерности d = d_model / n_head, что у baseline,
    поэтому параметры и FLOPs совпадают с обычным вниманием (h = d_model / 2d в статье).
    """
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        assert cfg.n_head % 2 == 0, "для diff-внимания n_head (число голов baseline) должно быть чётным"
        self.d = cfg.d_model // cfg.n_head        # размерность Q1, Q2, K1, K2
        self.h = cfg.n_head // 2                   # число дифференциальных голов
        D = cfg.d_model
        self.w_q = nn.Linear(D, D, bias=False)     # [Q1; Q2] для всех голов
        self.w_k = nn.Linear(D, D, bias=False)     # [K1; K2]
        self.w_v = nn.Linear(D, D, bias=False)     # V размерности 2d на голову
        self.w_o = nn.Linear(D, D, bias=False)
        self.lambda_init = cfg.lambda_init if cfg.lambda_init is not None else lambda_init_fn(layer_idx)
        # λ общий для всех голов слоя; векторы инициализируются N(0, 0.1), как в официальном коде
        self.lambda_q1 = nn.Parameter(torch.randn(self.d) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(self.d) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(self.d) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(self.d) * 0.1)
        use_norm = True if cfg.head_norm is None else cfg.head_norm
        self.head_norm = RMSNorm(2 * self.d, eps=1e-5) if use_norm else None
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.record = None

    def lam(self):
        l1 = torch.exp(torch.dot(self.lambda_q1, self.lambda_k1).float())
        l2 = torch.exp(torch.dot(self.lambda_q2, self.lambda_k2).float())
        return l1 - l2 + self.lambda_init

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        # Q, K: (B, 2h, T, d) — у головы i подголовы 2i (Q1/K1) и 2i+1 (Q2/K2); V: (B, h, T, 2d)
        q = self.w_q(x).view(B, T, 2 * self.h, self.d).transpose(1, 2)
        k = self.w_k(x).view(B, T, 2 * self.h, self.d).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.h, 2 * self.d).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        record_attention_logits(self, q, k)
        q = q.view(B, self.h, 2, T, self.d)
        k = k.view(B, self.h, 2, T, self.d)
        drop = self.dropout if self.training else 0.0
        # (A1 - λA2) V = A1 V - λ A2 V: две обычные causal-softmax, можно использовать SDPA/FlashAttention
        y1 = F.scaled_dot_product_attention(q[:, :, 0], k[:, :, 0], v, is_causal=True, dropout_p=drop)
        y2 = F.scaled_dot_product_attention(q[:, :, 1], k[:, :, 1], v, is_causal=True, dropout_p=drop)
        y = y1 - self.lam().to(y1.dtype) * y2                       # (B, h, T, 2d)
        if self.head_norm is not None:
            y = self.head_norm(y)                                   # GroupNorm на каждую голову
        y = y * (1.0 - self.lambda_init)                            # фиксированный множитель (ур. 3)
        y = y.transpose(1, 2).contiguous().view(B, T, C)            # Concat
        return self.resid_drop(self.w_o(y))


class DiffBlock(nn.Module):
    """Блок как в baseline_gpt (ур. 4-5 статьи), но с выбранным видом внимания."""
    def __init__(self, cfg, layer_idx=0):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        if cfg.arch == "diff":
            self.attn = DiffAttention(cfg, layer_idx)
        elif cfg.arch == "baseline":
            self.attn = BaselineAttention(cfg, layer_idx)
        else:
            raise ValueError(f"неизвестная архитектура {cfg.arch!r}")
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, h, cos, sin):
        h = h + self.attn(self.norm_attn(h), cos, sin)   # Y = MultiHead(LN(X)) + X
        h = h + self.mlp(self.norm_mlp(h))               # X' = SwiGLU(LN(Y)) + Y
        return h


class DiffGPT(GPT):
    """GPT из baseline_gpt.py (эмбеддинги, RoPE, финальная норма, forward, generate),
    у которого блоки заменены на DiffBlock. arch="baseline" — обычный трансформер."""
    def __init__(self, cfg: DiffConfig):
        super().__init__(cfg)
        self.blocks = nn.ModuleList([DiffBlock(cfg, i) for i in range(cfg.n_layer)])
        # та же инициализация, что в baseline_gpt (A.6): N(0, 0.02), выходные проекции / sqrt(2L)
        for block in self.blocks:
            block.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("attn.w_o.weight") or name.endswith("mlp.w_o.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))


# ============================ Чекпоинты ============================
def save_checkpoint(path, model, **meta):
    """Веса + конфигурация (только тензоры и простые типы — открывается с weights_only=True)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "config": asdict(model.cfg), **meta}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    """Возвращает (DiffGPT в режиме eval, словарь чекпоинта)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = DiffGPT(DiffConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


# ============================ Проверка работоспособности ============================
def reference_diff_attention(attn, x, cos, sin):
    """Прямая реализация ур. 1-3 статьи с явными матрицами внимания — эталон для сверки."""
    B, T, C = x.shape
    h, d = attn.h, attn.d
    q = apply_rope(attn.w_q(x).view(B, T, 2 * h, d).transpose(1, 2), cos, sin)
    k = apply_rope(attn.w_k(x).view(B, T, 2 * h, d).transpose(1, 2), cos, sin)
    v = attn.w_v(x).view(B, T, h, 2 * d).transpose(1, 2)
    mask = torch.full((T, T), float("-inf"), device=x.device).triu(1)
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


def _report(ok, text):
    print(f"  [{'OK' if ok else 'ОШИБКА'}] {text}", flush=True)
    return ok


def check_equations():
    torch.manual_seed(0)
    c = DiffConfig(vocab_size=64, block_size=16, n_layer=3, n_head=4, d_model=32, arch="diff")
    model = DiffGPT(c).eval()
    attn = model.blocks[1].attn
    x = torch.randn(2, 12, c.d_model)
    cos, sin = rope_cache(12, c.d_model // c.n_head, c.rope_base)
    with torch.no_grad():
        err = (attn(x, cos, sin) - reference_diff_attention(attn, x, cos, sin)).abs().max().item()
    return _report(err < 1e-4, f"совпадение с формулами статьи (ур. 1-3): макс. расхождение {err:.1e}")


def check_lambda():
    c = DiffConfig(vocab_size=64, block_size=16, n_layer=6, n_head=4, d_model=32, arch="diff")
    lams = [round(b.attn.lambda_init, 3) for b in DiffGPT(c).blocks]
    expected = [round(0.8 - 0.6 * math.exp(-0.3 * i), 3) for i in range(6)]
    return _report(lams == expected, f"λ_init по слоям = 0.8 - 0.6·exp(-0.3·(l-1)): {lams}")


def check_causal():
    torch.manual_seed(0)
    model = DiffGPT(DiffConfig(vocab_size=64, block_size=16, n_layer=2, n_head=4, d_model=32, arch="diff")).eval()
    x = torch.randint(64, (1, 12))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % 64
    with torch.no_grad():
        same = torch.allclose(model(x)[0][:, :-1], model(y)[0][:, :-1], atol=1e-5)
    return _report(same, "каузальность: будущие токены не влияют на прошлые предсказания")


def check_params():
    c = dict(vocab_size=32000, n_layer=6, n_head=6, d_model=384, block_size=256)
    nd = sum(p.numel() for p in DiffGPT(DiffConfig(arch="diff", **c)).parameters())
    nb = sum(p.numel() for p in DiffGPT(DiffConfig(arch="baseline", **c)).parameters())
    return _report(abs(nd - nb) / nb < 0.001,
                   f"параметров как у Transformer (пресет small): diff {nd / 1e6:.2f}M vs baseline {nb / 1e6:.2f}M")


def check_training(device, steps=150):
    """Короткое обучение маленькой diff-модели на реальных токенах (если есть) — loss должен падать."""
    torch.manual_seed(0)
    here = os.path.dirname(os.path.abspath(__file__))
    caches = [os.path.join(b, "data", "tinystories.txt.mistral.pt") for b in (here, ROOT)]
    cache = next((c for c in caches if os.path.exists(c)), None)
    if cache:
        data, src = torch.load(cache)[:2_000_000], "TinyStories"
    else:
        data, src = torch.arange(20_000) % 97 + 3, "синтетические данные"
    c = DiffConfig(vocab_size=32000, n_layer=2, n_head=4, d_model=128, block_size=64, arch="diff")
    model = DiffGPT(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    t0, losses = time.time(), []
    for _ in range(steps):
        ix = torch.randint(len(data) - 65, (16,))
        x = torch.stack([data[i:i + 64] for i in ix]).to(device)
        y = torch.stack([data[i + 1:i + 65] for i in ix]).to(device)
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    first, last = sum(losses[:10]) / 10, sum(losses[-10:]) / 10
    lam = model.blocks[0].attn.lam().item()
    _report(math.isfinite(last), f"обучение без nan/inf ({src}, {steps} шагов, {time.time() - t0:.0f} с)")
    return _report(last < first - 1.0, f"loss падает: {first:.2f} -> {last:.2f}; λ в 1-м слое {lam:.3f} (учится)")


def check_trained(device):
    """Если diff-модель уже обучена — показать её λ, val loss и пример генерации."""
    from chat import generate_stream
    from sampling import SamplingConfig
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(base, *rel) for base in (here, ROOT)
                  for rel in (("checkpoints", "ablation", "diff.pt"), ("checkpoints", "diff_small.pt"))]
    paths = [p for p in candidates if os.path.exists(p)]
    if not paths:
        print("  [—] обученной diff-модели пока нет: python run_experiments.py --quick")
        return True
    model, ckpt = load_checkpoint(paths[0], device)
    tok = load_tokenizer()
    lams = [round(b.attn.lam().item(), 3) for b in model.blocks]
    print(f"  Модель {paths[0]}: шаг {ckpt.get('step')}, val loss {ckpt.get('val_loss', float('nan')):.3f}")
    print(f"  Выученные λ по слоям: {lams}")
    torch.manual_seed(0)
    prompt = "Once upon a time"
    ids = tok.encode(prompt, add_special_tokens=False)
    new = list(generate_stream(model, ids, 80, eos_id=tok.eos_token_id, cfg=SamplingConfig()))
    print("  Пример: " + tok.decode(ids + new, skip_special_tokens=True).replace("\n", " "))
    return True


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Проверка Differential Transformer (arXiv:2410.05258), устройство {device}\n")
    print("1. Реализация")
    results = [check_equations(), check_lambda(), check_causal(), check_params()]
    print("\n2. Обучение")
    results.append(check_training(device))
    print("\n3. Обученная модель")
    results.append(check_trained(device))
    ok = all(results)
    print("\nИТОГ: " + ("Diff Transformer работает." if ok else "есть ошибки — см. выше."))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

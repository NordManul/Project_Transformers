"""
Базовый GPT из статьи «nGPT: Normalized Transformer with Representation Learning
on the Hypersphere» (Loshchilov et al., NVIDIA, arXiv:2410.01131).
Раздел 2 «Baseline Transformer», таблицы 1-3, приложение A.6.

Что взято из статьи:
  - pre-norm: h = h + ATTN(RMSNorm(h)); h = h + MLP(RMSNorm(h)); финальный RMSNorm  (ур. 4-5)
  - внимание: раздельные W_q, W_k, W_v, RoPE (база 10000) на q и k,
    масштаб 1/sqrt(d_k), каузальная маска, d_k = d_model / n_heads                 (ур. 12-14)
  - MLP: SwiGLU(u, v) = u * SiLU(v), d_MLP = 4 * d_model, без bias'ов               (ур. 17-19)
  - раздельные входная и выходная матрицы эмбеддингов, logits = E_output h          (ур. 1)
  - init: N(0, 0.02), у выходных проекций std / sqrt(2 * n_layer)                   (A.6)
  - AdamW, weight decay 0.1, warmup 2000 шагов, cosine annealing до 0              (табл. 3)

Чего в статье нет, и что выбрано по умолчанию (как в nanoGPT/LLaMA):
  betas (0.9, 0.95), grad clip 1.0, eps RMSNorm 1e-6, weight decay только на матрицы.

Токенизатор:
  В статье — LLaMA-2 (лицензия Meta). Здесь вместо него токенизатор Mistral 7B
  (mistralai/Mistral-7B-v0.1, лицензия Apache 2.0): тот же тип (SentencePiece BPE
  с byte-fallback), тот же словарь 32 000, BOS = 1, EOS = 2, поэтому архитектура
  и число параметров совпадают со статьёй. Сами токены отличаются, так что значения
  loss напрямую с таблицами статьи не сравниваются — сравнивайте baseline и nGPT
  между собой на одном токенизаторе.
  Репозиторий на Hugging Face требует согласиться поделиться контактами
  (кнопка на странице модели) и выполнить `hf auth login`. Это нужно один раз:
  при первом запуске токенизатор сохраняется в папку tokenizer/ (~2-3 МБ),
  и дальше грузится оттуда без сети и логина.

Запуск:
    pip install torch transformers sentencepiece
    python baseline_gpt.py --preset tiny --data input.txt
"""
import argparse
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================ Конфигурация ============================
@dataclass
class ModelConfig:
    vocab_size: int = 32000     # словарь 32k (в статье LLaMA-2, у нас Mistral 7B)
    block_size: int = 1024      # длина контекста (в статье 1k, 4k, 8k)
    n_layer: int = 24
    n_head: int = 16
    d_model: int = 1024
    mlp_ratio: int = 4          # d_MLP = 4 * d_model (табл. 2)
    rope_base: float = 10000.0
    tie_embeddings: bool = False  # в статье E_input и E_output раздельные
    dropout: float = 0.0        # в статье dropout не упоминается


# Таблица 2 статьи + маленький пресет для экспериментов на CPU/одной GPU
PRESETS = {
    "tiny": dict(n_layer=4,  n_head=4,  d_model=128,  block_size=128),
    "0.5B": dict(n_layer=24, n_head=16, d_model=1024, block_size=1024),
    "1B":   dict(n_layer=36, n_head=20, d_model=1280, block_size=1024),
}


# ============================ Компоненты ============================
class RMSNorm(nn.Module):
    """Нормирует вектор к RMS = 1 (т.е. к норме sqrt(d_model)) и масштабирует
    обучаемым вектором, инициализированным единицами."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def rope_cache(seq_len, head_dim, base, device=None):
    """Предвычисляет cos/sin для Rotary Position Embeddings."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, head_dim)
    return emb.cos(), emb.sin()


def apply_rope(x, cos, sin):
    """x: (B, H, T, head_dim). Поворачивает пары координат на угол, зависящий от позиции."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_k = cfg.d_model // cfg.n_head
        d = cfg.d_model
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        # (B, T, C) -> (B, H, T, d_k)
        q = self.w_q(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        # softmax(q k^T / sqrt(d_k) + M) v ; масштаб 1/sqrt(d_k) — значение по умолчанию
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)   # Concat(head_1..head_H)
        return self.resid_drop(self.w_o(y))


class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d, h = cfg.d_model, cfg.mlp_ratio * cfg.d_model
        self.w_u = nn.Linear(d, h, bias=False)
        self.w_v = nn.Linear(d, h, bias=False)
        self.w_o = nn.Linear(h, d, bias=False)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.resid_drop(self.w_o(self.w_u(x) * F.silu(self.w_v(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm_attn = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm_mlp = RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, h, cos, sin):
        h = h + self.attn(self.norm_attn(h), cos, sin)   # ур. 4
        h = h + self.mlp(self.norm_mlp(h))               # ур. 5
        return h


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb_in = nn.Embedding(cfg.vocab_size, cfg.d_model)            # E_input
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm_f = RMSNorm(cfg.d_model)                                 # финальная нормализация
        self.emb_out = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # E_output
        if cfg.tie_embeddings:
            self.emb_out.weight = self.emb_in.weight

        cos, sin = rope_cache(cfg.block_size, cfg.d_model // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Инициализация (A.6): N(0, 0.02); выходные проекции — std / sqrt(2 * n_layer)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("attn.w_o.weight") or name.endswith("mlp.w_o.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "последовательность длиннее block_size"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        h = self.drop(self.emb_in(idx))
        for block in self.blocks:
            h = block(h, cos, sin)
        h = self.norm_f(h)
        logits = self.emb_out(h)                                           # z = E_output h
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :].float() / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# ============================ Оптимизатор и расписание ============================
def make_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    """AdamW, weight decay 0.1 (табл. 3) — только для матриц и эмбеддингов,
    веса RMSNorm без decay (в статье не уточняется; так делают nanoGPT и LLaMA)."""
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at(step, base_lr, warmup, total):
    """Линейный warmup, затем cosine annealing до 0 (табл. 3)."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ============================ Токенизатор Mistral 7B ============================
TOKENIZER_REPO = "mistralai/Mistral-7B-v0.1"  # Apache 2.0, SentencePiece BPE, 32 000 токенов
# Локальная копия токенизатора (~2-3 МБ) в папке tokenizer/ рядом со скриптом
TOKENIZER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer")


def load_tokenizer():
    """Сначала ищет локальную копию в tokenizer/. Если её нет — один раз скачивает
    с Hugging Face и сохраняет туда, после чего сеть и логин больше не нужны."""
    from transformers import AutoTokenizer
    if os.path.isfile(os.path.join(TOKENIZER_DIR, "tokenizer_config.json")):
        tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        source = TOKENIZER_DIR
    else:
        try:
            tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
        except Exception as e:
            raise RuntimeError(
                f"Не удалось загрузить {TOKENIZER_REPO}. Откройте https://huggingface.co/{TOKENIZER_REPO}, "
                f"согласитесь на условия доступа и выполните `hf auth login`."
            ) from e
        tok.save_pretrained(TOKENIZER_DIR)
        source = TOKENIZER_REPO
        print(f"Токенизатор сохранён в {TOKENIZER_DIR} — дальше он загружается оттуда без сети")
    assert tok.vocab_size == 32000, f"ожидался словарь 32000, получен {tok.vocab_size}"
    print(f"Токенизатор: {source} (словарь {tok.vocab_size})")
    return tok


def encode_text(tok, text, chunk_chars=1_000_000):
    """Кодирует большой текст кусками по ~1M символов, разрезая по переводу строки.
    (SentencePiece добавляет пробел-префикс в начало каждого куска — на ~1 токен
    на мегабайт текста это не влияет.) BOS ставится в начало, как у LLaMA/Mistral."""
    ids = [tok.bos_token_id]
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_chars, len(text))
        if end < len(text):
            nl = text.rfind("\n", pos, end)
            if nl > pos:
                end = nl + 1
        ids.extend(tok.encode(text[pos:end], add_special_tokens=False))
        pos = end
    return ids


# ============================ Обучение ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="tiny", choices=PRESETS.keys())
    parser.add_argument("--data", default="input.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200, help="в статье 2000 (на длинных прогонах)")
    parser.add_argument("--lr", type=float, default=1e-3, help="в статье подбирался под задачу")
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    # --- данные (в статье — OpenWebText) ---
    if os.path.exists(args.data):
        text = open(args.data, encoding="utf-8").read()
    else:
        print(f"{args.data} не найден — использую короткий встроенный текст")
        text = "Привет, мир! Это базовый GPT из статьи про nGPT. " * 2000

    # --- токенизация: Mistral 7B, словарь 32 000 ---
    tok = load_tokenizer()
    vocab_size = tok.vocab_size
    cache = f"{args.data}.mistral.pt"
    if os.path.exists(args.data) and os.path.exists(cache) \
            and os.path.getmtime(cache) >= os.path.getmtime(args.data):
        data = torch.load(cache)
        print(f"Токены загружены из кэша {cache}")
    else:
        data = torch.tensor(encode_text(tok, text), dtype=torch.long)
        if os.path.exists(args.data):
            torch.save(data, cache)
    bos_id = tok.bos_token_id
    decode = lambda ids: tok.decode(ids, skip_special_tokens=True)

    print(f"Символов: {len(text):,} | токенов: {len(data):,} | "
          f"символов на токен: {len(text) / len(data):.2f}")

    cfg = ModelConfig(vocab_size=vocab_size, **PRESETS[args.preset])
    T = cfg.block_size
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    if len(splits["val"]) <= T + 1:
        raise ValueError(f"Слишком мало данных: в val-части {len(splits['val'])} токенов, "
                         f"нужно больше block_size={T}. Возьмите текст побольше.")

    def get_batch(split):
        d = splits[split]
        ix = torch.randint(len(d) - T - 1, (args.batch_size,))
        x = torch.stack([d[i:i + T] for i in ix])
        y = torch.stack([d[i + 1:i + T + 1] for i in ix])
        return x.to(device), y.to(device)

    @torch.no_grad()
    def estimate_loss(model, iters=50):
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(iters)
            for i in range(iters):
                x, y = get_batch(split)
                with autocast:
                    _, loss = model(x, y)
                losses[i] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Пресет {args.preset}: {n_params / 1e6:.2f}M параметров, устройство {device}, bf16={use_bf16}")
    optimizer = make_optimizer(model, args.lr)

    for step in range(args.max_iters + 1):
        lr = lr_at(step, args.lr, args.warmup, args.max_iters)
        for g in optimizer.param_groups:
            g["lr"] = lr
        if step % args.eval_every == 0:
            l = estimate_loss(model)
            print(f"шаг {step:5d} | lr {lr:.2e} | train {l['train']:.3f} | val {l['val']:.3f}")
        x, y = get_batch("train")
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    start = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    with autocast:
        out = model.generate(start, max_new_tokens=300, temperature=0.8, top_k=20)
    print("\n--- Сгенерированный текст ---\n" + decode(out[0].tolist()))


if __name__ == "__main__":
    main()
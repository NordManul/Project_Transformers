"""
nGPT — нормализованный трансформер из статьи «nGPT: Normalized Transformer with
Representation Learning on the Hypersphere» (Loshchilov et al., arXiv:2410.01131).
Один самодостаточный файл: модель, токенизатор, обучение и генерация.

Рецепт из раздела 2.6 статьи и где он реализован:
  1. Убрать RMSNorm/LayerNorm                                  -> в NGPT их нет
  2. После каждого шага обучения нормировать E_input, E_output,
     W_q, W_k, W_v, W_o, W_u, W_v, W_oMLP вдоль embedding-измерения -> NGPT.normalize_weights()
  3. h <- Norm(h + alpha_A (h_A - h)), h <- Norm(h + alpha_M (h_M - h)),
     alpha_init = 0.05, alpha_scale = 1/sqrt(d_model)          -> NormalizedBlock (ур. 10-11)
  4. Масштаб softmax sqrt(d_k) вместо 1/sqrt(d_k);
     q <- Norm(q) s_qk, k <- Norm(k) s_qk,
     s_qk_init = 1, s_qk_scale = 1/sqrt(d_model)               -> NormalizedAttention (ур. 15-16)
  5. u <- u s_u, v <- v s_v sqrt(d_model), init = 1, scale = 1  -> NormalizedMLP (ур. 20-21)
  6. z <- z s_z, s_z_init = 1, s_z_scale = 1/sqrt(d_model)      -> NGPT.forward (ур. 3)
  7. Adam без weight decay и без warmup, cosine до 0           -> make_ngpt_optimizer, main()

Остальное как у базовой модели из статьи: RoPE (база 10000), SwiGLU с d_MLP = 4 d_model,
без bias'ов, раздельные E_input и E_output. Пресеты 0.5B и 1B — таблица 2.

Обучаемые масштабы (раздел 2.5): параметр хранится со значением s_scale, а в forward
умножается на s_init / s_scale. Так s_scale управляет эффективной скоростью обучения
этого параметра в Adam, не меняя глобальный learning rate.
Eigen learning rates берутся по модулю (alpha <- |alpha|), как в основном тексте (A.2).

Запуск:
    python ngpt.py --preset tiny --data input.txt
    python ngpt.py --preset tiny --data input.txt --scale_ref_dim 1024   # для маленьких моделей
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
class NGPTConfig:
    vocab_size: int = 32000    # токенизатор Mistral 7B (в статье LLaMA-2, тоже 32k)
    block_size: int = 1024     # длина контекста (в статье 1k, 4k, 8k)
    n_layer: int = 24
    n_head: int = 16
    d_model: int = 1024
    mlp_ratio: int = 4         # d_MLP = 4 * d_model (табл. 2)
    rope_base: float = 10000.0
    tie_embeddings: bool = False
    alpha_init: float = 0.05   # eigen learning rates alpha_A и alpha_M (≈ 1/n_layers)
    sqk_init: float = 1.0      # масштаб q и k
    suv_init: float = 1.0      # масштабы s_u и s_v в MLP
    sz_init: float = 1.0       # масштаб логитов
    qk_norm: bool = True       # нормировать q и k (ур. 15-16); без неё хуже экстраполяция (A.8)
    # Размерность, от которой считается s_scale = 1/sqrt(.) для alpha, s_qk и s_z.
    # 0 — d_model, как в статье. У маленьких моделей масштабы тогда учатся медленно
    # (эффективный шаг ∝ sqrt(d_model)); scale_ref_dim=1024 даёт им скорость как у 0.5B из статьи.
    scale_ref_dim: int = 0

    def base_scale(self):
        return 1 / math.sqrt(self.scale_ref_dim or self.d_model)


# Таблица 2 статьи + маленький пресет для экспериментов
PRESETS = {
    "tiny": dict(n_layer=4,  n_head=4,  d_model=128,  block_size=128),
    "0.5B": dict(n_layer=24, n_head=16, d_model=1024, block_size=1024),
    "1B":   dict(n_layer=36, n_head=20, d_model=1280, block_size=1024),
}


# ============================ Модель ============================
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


def unit(x, dim=-1):
    """Norm(x): приводит вектор к единичной норме, без обучаемых весов."""
    return F.normalize(x.float(), dim=dim).type_as(x)


class Scale(nn.Module):
    """Обучаемый вектор масштабов с параметризацией s_init / s_scale (раздел 2.5)."""
    def __init__(self, dim, init, scale):
        super().__init__()
        self.ratio = init / scale
        self.s = nn.Parameter(torch.full((dim,), float(scale)))

    def forward(self):
        return self.s * self.ratio


class NormalizedAttention(nn.Module):
    def __init__(self, cfg: NGPTConfig):
        super().__init__()
        d = cfg.d_model
        self.n_head = cfg.n_head
        self.d_k = d // cfg.n_head
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, d, bias=False)
        self.w_v = nn.Linear(d, d, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)
        self.s_qk = Scale(d, cfg.sqk_init, cfg.base_scale())   # по вектору d_k на каждую голову
        self.qk_norm = cfg.qk_norm

    def forward(self, h, cos, sin):
        B, T, C = h.shape
        q = self.w_q(h).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(h).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(h).view(B, T, self.n_head, self.d_k).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if self.qk_norm:                                         # ур. 15-16
            q, k = unit(q), unit(k)
        s_qk = self.s_qk().view(self.n_head, 1, self.d_k)
        q, k = (q * s_qk).type_as(v), (k * s_qk).type_as(v)
        # масштаб softmax sqrt(d_k) вместо 1/sqrt(d_k) (шаг 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=math.sqrt(self.d_k))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.w_o(y)


class NormalizedMLP(nn.Module):
    def __init__(self, cfg: NGPTConfig):
        super().__init__()
        d, hidden = cfg.d_model, cfg.mlp_ratio * cfg.d_model
        self.w_u = nn.Linear(d, hidden, bias=False)
        self.w_v = nn.Linear(d, hidden, bias=False)
        self.w_o = nn.Linear(hidden, d, bias=False)
        self.s_u = Scale(hidden, cfg.suv_init, 1.0)
        self.s_v = Scale(hidden, cfg.suv_init, 1.0)
        self.sqrt_d = math.sqrt(d)

    def forward(self, h):
        u = self.w_u(h) * self.s_u()                    # ур. 20
        v = self.w_v(h) * (self.s_v() * self.sqrt_d)    # ур. 21: sqrt(d) — чтобы SiLU был нелинейным
        return self.w_o(u * F.silu(v))


class NormalizedBlock(nn.Module):
    def __init__(self, cfg: NGPTConfig):
        super().__init__()
        d = cfg.d_model
        self.attn = NormalizedAttention(cfg)
        self.mlp = NormalizedMLP(cfg)
        self.alpha_a = Scale(d, cfg.alpha_init, cfg.base_scale())
        self.alpha_m = Scale(d, cfg.alpha_init, cfg.base_scale())

    def forward(self, h, cos, sin):
        h_a = unit(self.attn(h, cos, sin))
        h = unit(h + self.alpha_a().abs() * (h_a - h))   # ур. 10
        h_m = unit(self.mlp(h))
        h = unit(h + self.alpha_m().abs() * (h_m - h))   # ур. 11
        return h


class NGPT(nn.Module):
    def __init__(self, cfg: NGPTConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.emb_in = nn.Embedding(cfg.vocab_size, d)                 # E_input
        self.blocks = nn.ModuleList([NormalizedBlock(cfg) for _ in range(cfg.n_layer)])
        self.emb_out = nn.Linear(d, cfg.vocab_size, bias=False)       # E_output
        if cfg.tie_embeddings:
            self.emb_out.weight = self.emb_in.weight
        self.s_z = Scale(cfg.vocab_size, cfg.sz_init, cfg.base_scale())

        cos, sin = rope_cache(cfg.block_size, d // cfg.n_head, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # A.6: N(0, 1/sqrt(d_model)); после нормировки инициализация уже не важна
        std = 1 / math.sqrt(d)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=std)
        self.normalize_weights()

    @torch.no_grad()
    def normalize_weights(self):
        """Шаг 2: каждый вектор вдоль embedding-измерения (размера d_model) -> единичная норма.
        Веса nn.Linear хранятся как (out, in): если d_model — вход, нормируем строки (dim=1),
        если d_model — выход (W_o, W_oMLP), нормируем столбцы (dim=0).
        Вызывать после каждого optimizer.step()."""
        def norm_(w, dim):
            w.copy_(F.normalize(w.float(), dim=dim).to(w.dtype))

        norm_(self.emb_in.weight, 1)          # (V, d)
        norm_(self.emb_out.weight, 1)         # (V, d)
        for b in self.blocks:
            norm_(b.attn.w_q.weight, 1)       # (d, d_model)
            norm_(b.attn.w_k.weight, 1)
            norm_(b.attn.w_v.weight, 1)
            norm_(b.attn.w_o.weight, 0)       # (d_model, d)
            norm_(b.mlp.w_u.weight, 1)        # (d_MLP, d_model)
            norm_(b.mlp.w_v.weight, 1)
            norm_(b.mlp.w_o.weight, 0)        # (d_model, d_MLP)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "последовательность длиннее block_size"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        h = self.emb_in(idx)                  # уже на единичной сфере
        for block in self.blocks:
            h = block(h, cos, sin)
        # финальной нормализации нет: h и так единичной нормы
        logits = self.emb_out(h) * self.s_z()  # ур. 1 и 3: косинусы в [-1, 1], затем масштаб
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
def make_ngpt_optimizer(model, lr, betas=(0.9, 0.95)):
    """Шаг 7 / табл. 3: Adam, то есть AdamW с weight decay 0, для всех параметров."""
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=0.0)


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
    parser = argparse.ArgumentParser(description="Обучение nGPT")
    parser.add_argument("--preset", default="tiny", choices=PRESETS.keys())
    parser.add_argument("--data", default="input.txt")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-3, help="в статье подбирался под задачу")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="0 — без клиппинга")
    parser.add_argument("--eval_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale_ref_dim", type=int, default=0,
                        help="0 — как в статье; 1024 — ускорить обучение масштабов у маленьких моделей")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

    # --- данные (в статье — OpenWebText) ---
    if os.path.exists(args.data):
        text = open(args.data, encoding="utf-8").read()
    else:
        print(f"{args.data} не найден — использую короткий встроенный текст")
        text = "Привет, мир! Это nGPT — нормализованный трансформер. " * 2000

    # --- токенизация: Mistral 7B, словарь 32 000 ---
    tok = load_tokenizer()
    cache = f"{args.data}.mistral.pt"
    if os.path.exists(args.data) and os.path.exists(cache) \
            and os.path.getmtime(cache) >= os.path.getmtime(args.data):
        data = torch.load(cache)
        print(f"Токены загружены из кэша {cache}")
    else:
        data = torch.tensor(encode_text(tok, text), dtype=torch.long)
        if os.path.exists(args.data):
            torch.save(data, cache)
    print(f"Символов: {len(text):,} | токенов: {len(data):,} | "
          f"символов на токен: {len(text) / len(data):.2f}")

    # --- модель и оптимизатор (табл. 3: Adam, без weight decay и warmup) ---
    cfg = NGPTConfig(vocab_size=tok.vocab_size, scale_ref_dim=args.scale_ref_dim, **PRESETS[args.preset])
    model = NGPT(cfg).to(device)
    optimizer = make_ngpt_optimizer(model, args.lr)
    T = cfg.block_size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"NGPT, пресет {args.preset}: {n_params / 1e6:.2f}M параметров | "
          f"устройство {device}, bf16={use_bf16} | lr {args.lr}")

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
    def estimate_loss(iters=50):
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

    # --- обучение ---
    for step in range(args.max_iters + 1):
        lr = lr_at(step, args.lr, 0, args.max_iters)   # без warmup
        for g in optimizer.param_groups:
            g["lr"] = lr
        if step % args.eval_every == 0:
            l = estimate_loss()
            print(f"шаг {step:5d} | lr {lr:.2e} | train {l['train']:.3f} | val {l['val']:.3f}")
        if step == args.max_iters:
            break
        x, y = get_batch("train")
        with autocast:
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        model.normalize_weights()   # шаг 2: возвращаем веса на гиперсферу

    # --- генерация ---
    model.eval()
    start = torch.tensor([[tok.bos_token_id]], dtype=torch.long, device=device)
    with autocast:
        out = model.generate(start, max_new_tokens=300, temperature=0.8, top_k=20)
    print("\n--- Сгенерированный текст ---\n" + tok.decode(out[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
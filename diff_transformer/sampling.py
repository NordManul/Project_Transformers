"""
Эвристики генерации текста: как из распределения вероятностей модели выбрать следующий токен,
чтобы текст был связнее и меньше повторялся. Модель при этом не меняется.

  temperature         делит логиты: < 1 — увереннее и скучнее, > 1 — разнообразнее и безумнее
  top_k               выбираем только из k самых вероятных токенов
  top_p (nucleus)     выбираем из минимального набора токенов с суммарной вероятностью >= p
  repetition_penalty  штраф токенам, которые уже встречались в последних `window` токенах
                      (как в CTRL, Keskar et al. 2019): логит > 0 делится на штраф, < 0 умножается
  no_repeat_ngram     запрет повторять n-грамму токенов, которая уже была в контексте
"""
from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F


@dataclass
class SamplingConfig:
    temperature: float = 0.8
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    no_repeat_ngram: int = 4
    window: int = 128

    def plain(self):
        """Те же temperature и top_k, но без остальных эвристик — для сравнения."""
        return replace(self, top_p=1.0, repetition_penalty=1.0, no_repeat_ngram=0)

    def describe(self):
        return (f"temp={self.temperature} top_k={self.top_k} top_p={self.top_p} "
                f"rep={self.repetition_penalty} ngram={self.no_repeat_ngram}")


def apply_repetition_penalty(logits, context, penalty, window=128):
    """Штрафует токены, которые встречались в последних `window` токенах контекста."""
    if penalty == 1.0 or not context:
        return logits
    seen = torch.tensor(sorted(set(context[-window:])), dtype=torch.long, device=logits.device)
    vals = logits[seen]
    logits[seen] = torch.where(vals > 0, vals / penalty, vals * penalty)
    return logits


def banned_ngram_tokens(context, n):
    """Токены, после которых последние n токенов повторили бы уже встречавшуюся n-грамму.
    Пример (n=2): контекст [5, 7, 9, 5] -> запрещён 7, потому что пара (5, 7) уже была."""
    if n <= 0 or len(context) < n:
        return set()
    prefix = tuple(context[len(context) - n + 1:])
    banned = set()
    for i in range(len(context) - n + 1):
        if tuple(context[i:i + n - 1]) == prefix:
            banned.add(context[i + n - 1])
    return banned


def top_k_filter(logits, k):
    if k and k < logits.numel():
        kth = torch.topk(logits, k).values[-1]
        logits[logits < kth] = -float("inf")
    return logits


def top_p_filter(logits, p):
    """Оставляет минимальный набор самых вероятных токенов с суммарной вероятностью >= p."""
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    # убираем токен, если до него вероятность уже набрала p (первый токен остаётся всегда)
    remove = (probs.cumsum(-1) - probs) >= p
    logits[order[remove]] = -float("inf")
    return logits


def sample_next(logits, context, cfg: SamplingConfig):
    """Выбирает следующий токен. logits — вектор (vocab,), context — список id контекста."""
    logits = logits.float().clone()
    logits = apply_repetition_penalty(logits, context, cfg.repetition_penalty, cfg.window)
    banned = banned_ngram_tokens(context, cfg.no_repeat_ngram)
    if banned:
        masked = logits.clone()
        masked[list(banned)] = -float("inf")
        if torch.isfinite(masked).any():      # если запрещено всё — запрет игнорируем
            logits = masked
    if cfg.temperature <= 0:                  # жадный выбор
        return int(torch.argmax(logits))
    logits = logits / cfg.temperature
    logits = top_k_filter(logits, cfg.top_k)
    logits = top_p_filter(logits, cfg.top_p)
    return int(torch.multinomial(F.softmax(logits, dim=-1), 1))

"""
Тесты эвристик генерации (sampling.py) и остановки по предложению/абзацу (chat.py).

    pytest tests/test_sampling.py -v
"""
import io

import torch

from chat import complete, stop_position
from sampling import (
    SamplingConfig, apply_repetition_penalty, banned_ngram_tokens, sample_next, top_p_filter,
)


def test_repetition_penalty_lowers_seen_tokens():
    logits = torch.tensor([2.0, -2.0, 1.0, 3.0])
    out = apply_repetition_penalty(logits.clone(), [0, 1], penalty=2.0)
    assert out[0] == 1.0          # положительный логит делится
    assert out[1] == -4.0         # отрицательный умножается
    assert out[2] == 1.0 and out[3] == 3.0   # невстречавшиеся не трогаем


def test_repetition_penalty_window():
    logits = torch.zeros(5) + 1.0
    out = apply_repetition_penalty(logits.clone(), [0, 1, 2, 3], penalty=2.0, window=2)
    assert out.tolist() == [1.0, 1.0, 0.5, 0.5, 1.0]


def test_banned_ngrams():
    assert banned_ngram_tokens([5, 7, 9, 5], 2) == {7}
    assert banned_ngram_tokens([1, 2, 3, 1, 2], 3) == {3}
    assert banned_ngram_tokens([1, 2, 3], 3) == set()
    assert banned_ngram_tokens([1, 2], 0) == set()
    assert banned_ngram_tokens([4, 4, 4], 1) == {4}


def test_top_p_keeps_minimal_nucleus():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = top_p_filter(logits.clone(), 0.75)
    assert torch.isfinite(out).tolist() == [True, True, False, False]
    out = top_p_filter(logits.clone(), 0.1)          # самый вероятный остаётся всегда
    assert torch.isfinite(out).tolist() == [True, False, False, False]
    assert torch.isfinite(top_p_filter(logits.clone(), 1.0)).all()


def test_no_repeat_ngram_changes_greedy_choice():
    logits = torch.tensor([0.0, 5.0, 4.0, 0.0])     # жадно выбрали бы 1
    cfg = SamplingConfig(temperature=0, no_repeat_ngram=2, repetition_penalty=1.0, top_p=1.0)
    assert sample_next(logits, [3, 1, 3], cfg) == 2  # пара (3, 1) уже была
    assert sample_next(logits, [0, 0, 3], cfg) == 1


def test_all_banned_falls_back():
    logits = torch.tensor([1.0, 2.0])
    cfg = SamplingConfig(temperature=0, no_repeat_ngram=1, repetition_penalty=1.0)
    assert sample_next(logits, [0, 1], cfg) == 1


def test_sample_next_does_not_modify_input():
    logits = torch.tensor([1.0, 2.0, 3.0])
    sample_next(logits, [2], SamplingConfig())
    assert logits.tolist() == [1.0, 2.0, 3.0]


def test_plain_disables_heuristics():
    p = SamplingConfig(temperature=0.5, top_k=10).plain()
    assert (p.temperature, p.top_k, p.top_p, p.repetition_penalty, p.no_repeat_ngram) == (0.5, 10, 1.0, 1.0, 0)


def test_stop_position():
    assert stop_position(" was happy. She ran", "sentence") == len(" was happy.")
    assert stop_position(" was happy.", "sentence") is None         # ждём, что будет дальше
    assert stop_position(" end.\n\nNext", "paragraph") == len(" end.")
    assert stop_position("\n\nhello", "paragraph") is None          # пустые строки в начале не считаются
    assert stop_position("anything. at all\n\n", "none") is None


class ScriptModel(torch.nn.Module):
    """Заглушка: по очереди выдаёт заранее заданные токены."""
    def __init__(self, script, vocab=64):
        super().__init__()
        from diff_transformer import DiffConfig as ModelConfig
        self.cfg = ModelConfig(vocab_size=vocab, block_size=32, n_layer=1, n_head=1, d_model=4)
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.script, self.vocab, self.start = script, vocab, None

    def forward(self, idx, targets=None):
        if self.start is None:
            self.start = idx.shape[1]
        step = idx.shape[1] - self.start
        logits = torch.full((*idx.shape, self.vocab), -10.0)
        logits[..., self.script[min(step, len(self.script) - 1)]] = 10.0
        return logits, None


class CharTok:
    """id = 10 + индекс символа в алфавите."""
    alphabet = "abcdefghijklmnopqrstuvwxyz .!\n"
    bos_token_id, eos_token_id = 1, 2

    def encode(self, text, add_special_tokens=False):
        return [10 + self.alphabet.index(c) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.alphabet[i - 10] for i in ids if i >= 10)


def test_complete_stops_at_sentence_and_paragraph():
    tok = CharTok()
    script = tok.encode(" hi. yo\n\nzz")
    cfg = SamplingConfig(temperature=0).plain()
    out = complete(ScriptModel(script), tok, "a", 50, cfg=cfg, stop="sentence", out=io.StringIO())
    assert out == "a hi."
    out = complete(ScriptModel(script), tok, "a", 50, cfg=cfg, stop="paragraph", out=io.StringIO())
    assert out == "a hi. yo"

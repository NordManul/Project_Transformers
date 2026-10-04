"""
Тесты эвристических метрик (metrics.py). Чистый Python, torch не нужен.

    pytest tests/test_metrics.py -v
"""
import pytest

from metrics import (
    avg_sentence_len, build_vocab, clean_end_ratio, distinct_n, format_table, known_word_ratio,
    repeat_rate, summarize, words,
)


def test_words():
    assert words("Lily's dog, Max — ran!") == ["lily's", "dog", "max", "ran"]


def test_distinct():
    assert distinct_n(["a b c d"], 1) == 1.0
    assert distinct_n(["a a a a"], 1) == 0.25
    assert distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
    assert distinct_n([""], 2) == 0.0


def test_repeat_rate():
    assert repeat_rate("one two three four five") == 0.0
    loop = "the cat sat down " * 6
    assert repeat_rate(loop) > 0.7


def test_known_words():
    vocab = build_vocab("the cat sat on the mat")
    assert known_word_ratio(["the cat flew"], vocab) == pytest.approx(2 / 3)


def test_sentence_len_and_clean_end():
    assert avg_sentence_len(["One two. Three four five!"]) == 2.5
    assert clean_end_ratio(["Done.", "not done", 'She said "hi."  ']) == pytest.approx(2 / 3)


def test_summarize_and_table():
    res = summarize(["The cat sat. The dog ran."], build_vocab("the cat sat the dog"))
    assert set(res) >= {"distinct-1", "repeat-4", "known_words", "clean_end"}
    table = format_table({"a": res, "b": res})
    assert "distinct-1" in table and "known_words" in table

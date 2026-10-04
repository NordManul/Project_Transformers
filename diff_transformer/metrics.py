"""
Эвристические метрики качества сгенерированного текста. Чистый Python, без torch.
Они не заменяют оценку человеком, но позволяют сравнивать модели и настройки генерации числами.

  distinct-n        доля уникальных n-грамм слов среди всех (выше — разнообразнее; < 0.3 — много повторов)
  repeat-4          доля 4-грамм слов, которые уже встречались раньше в том же тексте (ниже — лучше)
  known_words       доля слов, которые встречаются в обучающем тексте (ниже — модель выдумывает слова)
  sentence_len      средняя длина предложения в словах (слишком длинные — «бред без точек»)
  clean_end         доля ответов, которые заканчиваются концом предложения
"""
import re
from collections import Counter

_WORD = re.compile(r"[a-zA-Zа-яА-ЯёЁ]+(?:'[a-zA-Z]+)?")
_SENTENCE_SPLIT = re.compile(r"[.!?]+")


def words(text):
    return [w.lower() for w in _WORD.findall(text)]


def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(texts, n):
    """Уникальные n-граммы слов / все n-граммы по всем текстам вместе."""
    grams = [g for t in texts for g in ngrams(words(t), n)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeat_rate(text, n=4):
    """Доля n-грамм слов, которые уже встречались раньше в этом же тексте."""
    seen, repeats, grams = set(), 0, ngrams(words(text), n)
    for g in grams:
        if g in seen:
            repeats += 1
        seen.add(g)
    return repeats / len(grams) if grams else 0.0


def known_word_ratio(texts, vocab):
    ws = [w for t in texts for w in words(t)]
    return sum(w in vocab for w in ws) / len(ws) if ws else 0.0


def avg_sentence_len(texts):
    lens = [len(words(s)) for t in texts for s in _SENTENCE_SPLIT.split(t)]
    lens = [n for n in lens if n > 0]
    return sum(lens) / len(lens) if lens else 0.0


def clean_end_ratio(texts):
    ends = [bool(re.search(r"[.!?][\"')]*\s*$", t)) for t in texts if t.strip()]
    return sum(ends) / len(ends) if ends else 0.0


def build_vocab(text, min_count=1):
    counts = Counter(words(text))
    return {w for w, c in counts.items() if c >= min_count}


def summarize(texts, vocab=None):
    """Все метрики для набора сгенерированных текстов (только сгенерированная часть, без промпта)."""
    result = {
        "distinct-1": distinct_n(texts, 1),
        "distinct-2": distinct_n(texts, 2),
        "distinct-3": distinct_n(texts, 3),
        "repeat-4": sum(repeat_rate(t) for t in texts) / len(texts) if texts else 0.0,
        "sentence_len": avg_sentence_len(texts),
        "clean_end": clean_end_ratio(texts),
        "avg_words": sum(len(words(t)) for t in texts) / len(texts) if texts else 0.0,
    }
    if vocab is not None:
        result["known_words"] = known_word_ratio(texts, vocab)
    return result


# Что считается лучше: +1 — больше лучше, -1 — меньше лучше, 0 — просто информация
DIRECTION = {"distinct-1": 1, "distinct-2": 1, "distinct-3": 1, "repeat-4": -1,
             "known_words": 1, "clean_end": 1, "sentence_len": 0, "avg_words": 0}


def format_table(results):
    """results: {название настройки: summarize(...)} -> текстовая таблица."""
    names = list(results)
    keys = list(next(iter(results.values())))
    width = max(14, *(len(n) + 2 for n in names))
    lines = ["метрика".ljust(14) + "".join(n.rjust(width) for n in names) + "   лучше"]
    for k in keys:
        arrow = {1: "↑", -1: "↓", 0: ""}[DIRECTION.get(k, 0)]
        lines.append(k.ljust(14) + "".join(f"{results[n][k]:.3f}".rjust(width) for n in names)
                     + f"   {arrow}")
    return "\n".join(lines)

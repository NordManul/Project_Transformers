"""
Тесты токенизатора Mistral 7B (mistralai/Mistral-7B-v0.1, Apache 2.0).
Нужно согласиться на условия доступа на странице модели и выполнить `hf auth login`.
Без доступа тесты пропускаются.

    pytest tests/test_tokenizer.py -v
"""
import pytest

from baseline_gpt import encode_text, load_tokenizer

ENGLISH = (
    "The Transformer architecture is the foundation for most modern language models.\n"
    "We train both the baseline Transformer and the normalized Transformer on OpenWebText.\n"
)


@pytest.fixture(scope="module")
def tok():
    pytest.importorskip("transformers")
    try:
        return load_tokenizer()
    except RuntimeError as e:
        pytest.skip(f"нет доступа к токенизатору Mistral 7B: {e}")


def test_vocab_and_special_tokens(tok):
    assert tok.vocab_size == 32000
    assert tok.bos_token_id == 1
    assert tok.eos_token_id == 2


def test_encoded_data_starts_with_single_bos(tok):
    ids = encode_text(tok, ENGLISH)
    assert ids[0] == tok.bos_token_id
    assert tok.bos_token_id not in ids[1:]
    assert tok.eos_token_id not in ids
    assert max(ids) < 32000


def test_roundtrip(tok):
    ids = encode_text(tok, ENGLISH)
    assert tok.decode(ids, skip_special_tokens=True) == ENGLISH


def test_chunked_encoding_matches_whole(tok):
    """Кодирование кусками почти не отличается от кодирования целиком:
    текст тот же (с точностью до пробелов на стыках), токенов — не больше чем +1 на кусок."""
    text = ENGLISH * 50
    whole = encode_text(tok, text, chunk_chars=10**9)
    chunked = encode_text(tok, text, chunk_chars=500)
    n_chunks = len(text) // 500 + 1
    assert abs(len(chunked) - len(whole)) <= n_chunks
    norm = lambda s: " ".join(s.split())
    assert norm(tok.decode(chunked, skip_special_tokens=True)) == norm(text)


def test_english_compression(tok):
    """Для английского BPE даёт порядка 3–6 символов на токен."""
    ids = encode_text(tok, ENGLISH * 20)
    chars_per_token = len(ENGLISH * 20) / (len(ids) - 1)
    assert 3.0 < chars_per_token < 6.0
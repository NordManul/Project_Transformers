# Baseline GPT из статьи nGPT

Реализация базового трансформера, с которым сравнивается nGPT в статье
[«nGPT: Normalized Transformer with Representation Learning on the Hypersphere»](https://arxiv.org/abs/2410.01131)
(Loshchilov et al., NVIDIA, 2024).

Архитектура в стиле LLaMA, как в разделе 2 статьи: pre-norm RMSNorm, RoPE, SwiGLU,
без bias'ов, раздельные входные и выходные эмбеддинги. Пресеты `0.5B` и `1B`
точно совпадают с таблицей 2 статьи по числу параметров.

## Структура

```
baseline_gpt.py          модель, токенизатор, обучение
tests/test_model.py      тесты модели (без сети, CPU)
tests/test_tokenizer.py  тесты токенизатора
tokenizer/               токенизатор Mistral 7B (~2 МБ)
requirements.txt         зависимости
```

## Установка

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash); на Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Тесты

```bash
pytest -v
```

Среди прочего проверяются каузальность, RoPE, инициализация, число параметров
из статьи и совпадение логитов с `LlamaForCausalLM` из `transformers`.

## Обучение

```bash
python baseline_gpt.py --preset tiny --data input.txt
```

`input.txt` — любой текстовый файл. Пресеты: `tiny`, `0.5B`, `1B`.

## Токенизатор

В статье использован токенизатор LLaMA-2. Здесь вместо него токенизатор Mistral 7B
(`mistralai/Mistral-7B-v0.1`): тот же тип (SentencePiece BPE), тот же словарь
на 32 000 токенов, но свободная лицензия. Архитектура и число параметров совпадают
со статьёй, а значения loss напрямую с её таблицами не сравниваются.

Если папки `tokenizer/` нет, она создаётся при первом запуске, или её можно скачать вручную:

```bash
hf download mistralai/Mistral-7B-v0.1 tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json --local-dir tokenizer
```

Файлы токенизатора принадлежат Mistral AI и распространяются по лицензии
Apache 2.0 (текст лицензии — `tokenizer/LICENSE`).
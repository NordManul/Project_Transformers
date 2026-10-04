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
prepare_dataset.py       загрузка небольшой выборки TinyStories
interactive.py           ввод текста, потоковая генерация, просмотр токенов
tests/test_model.py      тесты модели (без сети, CPU)
tests/test_tokenizer.py  тесты токенизатора
tests/test_runtime.py    обучение с разными конфигурациями и запуск CLI
tests/test_console.py    сохранение, продолжение обучения и консольный режим
tokenizer/               токенизатор Mistral 7B (~2 МБ)
data/tinystories/         тексты и метаданные датасета (не отправляются в Git)
checkpoints/             веса модели и оптимизатора (не отправляются в Git)
runs/                    история train/validation loss (не отправляется в Git)
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

Первый эксперимент использует [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories):
короткие английские истории с простым словарём. Скрипт потоково читает первые 5 000 историй
из обучающего файла и 200 историй из отдельного проверочного файла, затем закрывает
соединение; весь датасет на несколько гигабайт скачивать не нужно.
Источник закреплён на ревизии `f54c09f`, лицензия датасета — CDLA-Sharing-1.0.
URL, размеры выборок и SHA-256 сохраняются в `data/tinystories/metadata.json`.

```bash
source .venv/bin/activate
python prepare_dataset.py
python baseline_gpt.py --preset tiny \
  --data data/tinystories/train.txt --val_data data/tinystories/val.txt \
  --device auto --batch_size 4 --max_iters 5000 --warmup 50 --lr 0.001 \
  --eval_every 500 --eval_iters 10 --save_every 500 \
  --checkpoint checkpoints/tinystories.pt --log_file runs/tinystories.jsonl
```

`auto` выбирает CUDA, затем GPU Apple (MPS), затем CPU. Для CPU можно уменьшить
`--batch_size` до 1–2. Первый эксперимент — модель `tiny` с 9,24 млн параметров;
это обучение с нуля продолжать текст, без готовых весов Mistral и без диалогового обучения.
После короткого обучения возможны повторения и ошибки; для этой выборки вводите английский текст.

Последние веса сохраняются в `checkpoints/tinystories.pt`, лучшие по проверочной ошибке —
в `checkpoints/tinystories.best.pt`. Сохранение включает конфигурацию модели, оптимизатор,
номер шага и состояния генераторов случайных чисел. Ctrl+C сохраняет выполненные шаги.
Пример продолжения до 10 000 суммарных шагов (параметры модели берутся из checkpoint):

```bash
python baseline_gpt.py --resume checkpoints/tinystories.pt \
  --data data/tinystories/train.txt --val_data data/tinystories/val.txt \
  --device auto --batch_size 4 --max_iters 10000 --warmup 50 --lr 0.0003 \
  --eval_every 500 --eval_iters 10 --save_every 500 \
  --checkpoint checkpoints/tinystories.pt --log_file runs/tinystories.jsonl
```

При увеличении `--max_iters` расписание learning rate перестраивается под новую цель.
Также можно использовать свой UTF-8 файл через `--data input.txt`:
если `--val_data` отсутствует, последние 10% токенов выделяются для проверки.
Отсутствующий или слишком короткий файл вызывает понятную ошибку.

Пресеты: `tiny`, `0.5B`, `1B`. Архитектуру можно менять параметрами
`--n_layer`, `--n_head`, `--d_model`, `--block_size`, `--mlp_ratio`, `--dropout`,
`--rope_base`, `--tie_embeddings`. Размер модели должен делиться на количество голов,
а размер одной головы должен быть чётным для RoPE. Данные кэшируются в `*.mistral.pt`.

## Интерактивная генерация

```bash
python interactive.py
```

Введите, например, `Once upon a time, a little cat`. Продолжение печатается постепенно.
Каждый ввод — отдельный запрос; история предыдущих запросов не добавляется.
Команды `/trace on` и `/trace off` переключают подробный просмотр и потоковый текст;
`/exit` завершает режим. Ctrl+C прерывает текущую генерацию, сохраняя возможность нового ввода.

В режиме trace для каждого шага видны ID выбранного токена, его представление,
вероятность выбора и пять наиболее вероятных кандидатов после temperature/top-k.
Модель вычисляет вероятности следующего токена по последним токенам контекста,
выбирает один токен, дописывает его и повторяет процесс. Ограничение для `tiny` —
128 токенов контекста, что меньше полного текста длинной истории.

Один запрос с просмотром процесса:

```bash
python interactive.py --prompt "Once upon a time, a little cat" --trace --max_tokens 20
```

Параметры: `--checkpoint`, `--device`, `--max_tokens`, `--temperature`, `--top_k`, `--seed`.
По умолчанию загружается `checkpoints/tinystories.best.pt`.

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

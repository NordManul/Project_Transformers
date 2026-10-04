# Differential Transformer

Реализация **Differential Transformer** из статьи
[«Differential Transformer»](https://arxiv.org/abs/2410.05258) (Ye et al., Microsoft Research /
Tsinghua University, ICLR 2025) и сравнение с обычным трансформером на TinyStories по методике статьи.

Папка опирается на то, что уже есть в репозитории, и не меняет его:
базовый трансформер, токенизатор и обучение берутся из `../baseline_gpt.py`,
токенизатор — из `../tokenizer/`, зависимости — из `../requirements.txt`.
Здесь только новое: дифференциальное внимание, обучение с абляциями, чат с эвристиками,
сравнение со статьёй и тесты.

## Что такое Diff Transformer

Отличается от обычного трансформера только вниманием (остальное как в LLaMA: pre-norm RMSNorm,
RoPE, SwiGLU):

- дифференциальное внимание `(softmax(Q1K1ᵀ/√d) − λ·softmax(Q2K2ᵀ/√d))·V` (ур. 1) —
  разность двух карт внимания вычитает «шум» на нерелевантных токенах;
- λ = exp(λq1·λk1) − exp(λq2·λk2) + λ_init, λ_init = 0.8 − 0.6·exp(−0.3·(l−1)) (ур. 2);
- **GroupNorm (RMSNorm) на каждую голову** и множитель (1 − λ_init) (ур. 3);
- h = n_head / 2 голов той же размерности — число параметров как у обычного трансформера.

## Файлы

```
diff_transformer.py   DiffConfig, DiffAttention, DiffGPT (наследует GPT из ../baseline_gpt.py),
                      сохранение/загрузка; самопроверка: python diff_transformer.py
train.py              обучение (--arch diff|baseline, --head_norm, --lambda_init) + эвристики обучения
run_experiments.py    обучает модели абляции из табл. 6 статьи и запускает compare.py
compare.py            loss / AR-Hit / Others (табл. 6), выбросы активаций (табл. 5), метрики генерации,
                      таблицы рядом с числами статьи -> results/comparison.md
chat.py               интерактивный режим: модель продолжает введённую фразу
sampling.py           эвристики генерации: top-p, штраф за повторы, запрет повтора n-грамм
metrics.py            эвристические метрики текста (distinct-n, повторы, известные слова)
evaluate.py           perplexity + генерация жадно / без эвристик / с эвристиками
prepare_data.py       скачивает часть TinyStories в data/
tests/                тесты (сверка с формулами статьи, генерация, эвристики, метрики)
results/              отчёт сравнения со статьёй
```

## Запуск

Окружение — общее для репозитория (`pip install -r requirements.txt` в корне; для видеокарты —
torch с https://pytorch.org). Все команды — из этой папки:

```powershell
.\.venv\Scripts\Activate.ps1             # в корне репозитория
cd diff_transformer
pytest -v                                # тесты папки
python diff_transformer.py               # проверка, что Diff Transformer работает
python prepare_data.py --mb 50           # TinyStories, ~12 млн токенов -> data/
python run_experiments.py --quick        # Transformer, Diff, Diff без GroupNorm + compare.py
python chat.py --ckpt checkpoints/ablation/diff.pt
```

Одна модель: `python train.py --arch diff --preset small --data data/tinystories.txt --max_iters 3000 --lr 6e-4`.
Все 7 моделей абляции (+ Transformer с вдвое меньшим числом голов, λ_init = 0.8 и 0.5):
`python run_experiments.py`. Обученные модели пропускаются, Ctrl+C — продолжить позже.

## Результаты (TinyStories, 38.7M параметров, 3000 шагов, 1 seed)

| | Transformer | Diff (GN) | Diff без GN | Статья (1.4B): Transformer → Diff |
|---|---|---|---|---|
| Valid loss | 1.761 | **1.748** (−0.75%) | 1.750 | 3.087 → 3.062 (−0.81%) |
| AR-Hit loss | 0.962 | **0.921** (−4.3%) | 0.935 | 0.898 → 0.880 (−2.0%) |
| Others loss | 1.931 | 1.924 (−0.4%) | 1.923 | 3.272 → 3.247 (−0.8%) |
| Логиты внимания, top-1 | 22.0 | 21.6 | — | 318 → 38.8 |
| Логиты внимания, медиана | 1.82 | 1.18 (×0.65) | — | 5.4 → 3.3 (×0.61) |

- **Подтвердилось:** Diff лучше Transformer при том же числе параметров, с почти тем же
  относительным выигрышем, что в статье; выигрыш сосредоточен на AR-Hit — токенах,
  которые модель «вспоминает» из контекста.
- **Не подтвердилось на нашем масштабе:** заметный вклад GroupNorm (+0.002 против +0.060 в статье)
  и подавление выбросов активаций. В статье это эффекты больших и долго обучаемых моделей
  (нестабильность обучения, «массивные» активации); у модели 39M на 25M токенах их почти нет.
- Абсолютные loss со статьёй несравнимы (1.4B параметров, 40K шагов, другие данные и токенизатор),
  сравниваются направление и относительная величина эффектов. Полный отчёт — `results/comparison.md`.

## Эвристики

**Обучение** (`train.py`): ранняя остановка по val (`--patience`), автоподбор batch под видеопамять
с накоплением градиентов, пропуск шагов с nan/inf, сохранение лучшей по val модели.

**Генерация** (`sampling.py`, `chat.py`): temperature 0.8, top-k 40, top-p 0.9, штраф за повторы 1.15,
запрет повтора 4-грамм, остановка на конце абзаца. В чате: `/topp`, `/rep`, `/ngram`, `/stop`,
`/plain` / `/heur` — выключить/включить эвристики, `/compare` — сравнить обе генерации.

**Метрики** (`metrics.py`, `evaluate.py`): distinct-1/2/3, repeat-4, known_words, clean_end.

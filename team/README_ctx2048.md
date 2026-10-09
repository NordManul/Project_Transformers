# nGPT против GPT на контексте 2048 — инструкция по запуску на Kaggle

## Что это за эксперимент

Мы уже сравнили GPT и nGPT при контексте 1024 токена: nGPT лучше (val loss 3.254 против 3.359), по токенам обучается быстрее в ≈1.41×. Теперь проверяем, как выигрыш меняется с длиной контекста: этот прогон — **контекст 2048**, параллельно на другом аккаунте идёт 4096. Три точки (1024, 2048, 4096) покажут, растёт ли выигрыш nGPT, как в статье.

Отличия от главного прогона:

| | Главный прогон | Этот запуск |
|---|---|---|
| Контекст | 1024 | **2048** |
| Окон в батче | 64 | **32** |
| Окон за один проход | 8 | **4** |
| Токенов на шаг | 65 536 | 65 536 (не меняется) |
| Шагов / токенов всего | 11 826 / 775M | столько же |
| lr | 0.003 у обеих моделей | 0.003 у обеих моделей |
| Сид | 42 | 42 |

Токенов на шаг и расписание lr те же, поэтому сравнение честное. GPT и nGPT обучаются одновременно,
каждая на своей видеокарте T4. Время: GPT ~5.5–6.5 ч, nGPT ~9–10 ч.
Расход квоты GPU: примерно 11–12 из 30 часов в неделю.

**Порядок:** 1) подготовка (5 минут) → 2) запуск (браузер можно закрыть) → 3) проверка и результат
→ 4) только если обучение не успело — вторая сессия.

---

## 1. Подготовка (один раз)

1. Аккаунт Kaggle с **подтверждённым телефоном**: аватар → Settings → Phone Verification = Verified.
   Без этого не будет ни GPU, ни интернета.
2. На kaggle.com: **Create → New Notebook**.
3. Переименуй ноутбук (клик по названию сверху) в **`ngpt-ctx2048`**.
4. Справа, в панели **Session options**:
   - **Accelerator → GPU T4 x2** (именно x2, не P100 и не одна T4);
   - **Internet → On**;
   - раздел **Input** — **пустой**.
5. Сделай в ноутбуке **четыре ячейки с кодом** (новая — кнопка **+ Code** под последней ячейкой)
   и скопируй в них текст ниже **целиком, ничего не меняя**.

### Ячейка 1 — настройки

```python
# ===== Настройки (ничего не менять) =====
CTX = 2048       # длина контекста
SEED = 42       # сид: начальные веса и порядок батчей
LR = "3e-3"       # lr обеих моделей
BUDGET = "775M"   # токенов обучения
BATCH = 64 * 1024 // CTX          # окон в батче: токенов на шаг всегда 65 536
MICRO = max(1, 8 * 1024 // CTX)   # окон за один проход по видеокарте

SESSION_HOURS = 11.5
TRAIN_TOKENS = "9e8"
REPO = "https://github.com/NordManul/Project_Transformers.git"
PLAN = (f"--n_layer 6 --n_head 6 --d_model 384 --block_size {CTX} --batch_size {BATCH} --micro_batch {MICRO} "
        f"--budgets {BUDGET} --lrs {LR} --seed {SEED} --data_seed {1234 + SEED - 42}")
NAME = f"ctx{CTX}_{BUDGET}_lr{LR}_s{SEED}"

import glob, json, os, shutil, subprocess, time
T0 = time.time()
WORK = "/kaggle/working"
CODE = f"{WORK}/Project_Transformers"
DATA = f"{WORK}/data/owt"
if not os.path.exists(CODE):
    subprocess.run(["git", "clone", "--depth", "1", REPO, CODE], check=True)
os.chdir(CODE)
print("код:", subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip())
import torch
print("torch", torch.__version__, "| GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

OUT = f"{WORK}/runs/{NAME}"
print("PLAN:", PLAN)
print("OUT:", OUT)
```

### Ячейка 2 — данные (900M токенов OpenWebText, ~14 минут) и продолжение прошлой сессии

```python
found = sorted(glob.glob("/kaggle/input/**/data/owt/train.bin", recursive=True))
if os.path.exists(f"{DATA}/train.bin"):
    print("Данные уже на месте")
elif found and json.load(open(os.path.join(os.path.dirname(found[0]), "meta.json")))["train_tokens"] >= 9e8:
    print("Копирую данные из входа:", os.path.dirname(found[0]))
    shutil.copytree(os.path.dirname(found[0]), DATA, dirs_exist_ok=True)
else:
    !python tokenize_dataset.py --dataset openwebtext --train_tokens {TRAIN_TOKENS} --val_tokens 1e7 --out {DATA}
print(open(f"{DATA}/meta.json").read())

# Если во входе есть прошлая сессия этого же запуска — продолжаем её, а не начинаем заново
prev = sorted(glob.glob(f"/kaggle/input/**/runs/{NAME}/plan.json", recursive=True))
if prev and not os.path.exists(f"{OUT}/plan.json"):
    print("Продолжаю прошлый запуск:", os.path.dirname(prev[0]))
    shutil.copytree(os.path.dirname(prev[0]), OUT, dirs_exist_ok=True)
else:
    print("Новый запуск" if not prev else "Прошлый запуск уже скопирован")
```

### Ячейка 3 — обучение (GPT и nGPT одновременно, каждая на своей T4)

```python
HOURS = f"{SESSION_HOURS - (time.time() - T0) / 3600 - 0.3:.2f}"
print("Часов на обучение:", HOURS)
!python ngpt_vs_gpt.py run --data {DATA} --out {OUT} {PLAN} --hours {HOURS}
```

### Ячейка 4 — отчёт и PDF

```python
!python ngpt_vs_gpt.py report --out {OUT} > /dev/null
!python make_pdf_report.py {OUT}/report --out /kaggle/working/{NAME}.pdf
from IPython.display import Markdown, display
display(Markdown(open(f"{OUT}/report/report.md", encoding="utf-8").read()))
```

Сохрани ноутбук (**Ctrl+S**). Ячейки вручную запускать не нужно.

---

## 2. Запуск

1. Проверь справа: **GPU T4 x2**, **Internet On**, **Input пустой**.
2. **Save Version** (кнопка вверху справа) → **Save & Run All (Commit)** → **Save**.
3. Если вверху справа горит включённая интерактивная сессия — выключи её кнопкой **⏻**
   (иначе она тратит квоту впустую). Браузер и компьютер можно закрыть: всё идёт на серверах Kaggle.

---

## 3. Проверка и результат

### Через ~20 минут после запуска

**Your Work → `ngpt-ctx2048` → версия, которая сейчас идёт → Logs.** Должно быть:

```
код: <короткий хэш>
torch ... | GPU: ['Tesla T4', 'Tesla T4']                 ← обязательно две T4
PLAN: ... --block_size 2048 --batch_size 32 --micro_batch 4 ... --seed 42 ...
Новый запуск
Часов на обучение: 10.9...
План: 2 прогонов, устройства ['0', '1'], лимит 10.9... ч
[0] gpt_775M_lr0.003: старт (lr 0.003, 11826 шагов, 775M токенов)
[1] ngpt_775M_lr0.003: старт (lr 0.003, 11826 шагов, 775M токенов)
```

Дальше лог надолго замолкает — **это нормально**: ход обучения пишется в файлы, а не в лог.

Что-то не так, если:
- GPU одна или их нет → отмени запуск (**Active Events → ⋯ → Cancel**), поставь GPU T4 x2, запусти заново;
- вместо «Новый запуск» написано «Продолжаю прошлый запуск» → во входе что-то лишнее: отмени, очисти Input,
  запусти заново;
- «ОШИБКА» или «CUDA out of memory» → пришли в чат весь текст ошибки из Logs.

### В конце

Через ~5.5–6.5 ч в логе появится строка GPT, позже — строка nGPT:

```
[0] gpt_775M_lr0.003: готово, val loss X.XXXX
[1] ngpt_775M_lr0.003: готово, val loss Y.YYYY
Все прогоны закончены.
```

Тогда:
1. Ноутбук → **Share → Add collaborators** → `vladimir337` → права **Can view**.
2. Напиши в чат две строки `готово, val loss ...`.
3. PDF-отчёт лежит во вкладке **Output** этой версии ноутбука: `ctx2048_775M_lr3e-3_s42.pdf`. Скачай и пришли в чат.

Если вместо «готово» у какой-то модели написано **«пауза (время сессии)»** и в конце
«Не закончено: 1 прогонов» — модель не успела за 12 часов. Это не ошибка, переходи к разделу 4.

---

## 4. Только если была «пауза»: вторая сессия

Обучение сохранилось в выводе первой сессии и продолжится с того же места.

1. Открой ноутбук `ngpt-ctx2048` → **Edit**.
2. Справа **Add Input** → вкладка **Your Work** (или **Notebooks**) → **`ngpt-ctx2048`** → **Add**.
   Это вывод твоей первой сессии: в нём данные и сохранённое обучение.
3. В ячейках **ничего не меняй**. Проверь: GPU T4 x2, Internet On.
4. **Save Version → Save & Run All (Commit) → Save**, выключи интерактивную сессию **⏻**.
5. Через ~10 минут в Logs должно быть:
   ```
   Копирую данные из входа: /kaggle/input/...
   Продолжаю прошлый запуск: /kaggle/input/.../runs/ctx2048_775M_lr3e-3_s42
   [0] ngpt_775M_lr0.003: продолжаю (lr 0.003, 11826 шагов, 775M токенов)
   ```
   Модель, которая уже закончила, заново не запускается.
6. В конце — «Все прогоны закончены», дальше как в разделе 3: Share, строки «готово», PDF
   (из Output **второй** сессии — в нём полный отчёт).

---

## Коротко

| Шаг | Что делаешь | Время |
|---|---|---|
| 1 | Новый ноутбук `ngpt-ctx2048`, 4 ячейки, GPU T4 x2, Internet On, Input пустой | 5 мин |
| 2 | Save & Run All (Commit), выключить ⏻ | 1 мин |
| 3 | Через 20 мин проверить Logs; в конце Share `vladimir337`, строки «готово», PDF | ~9–10 ч ожидания |
| 4 | Только при «паузе»: Add Input = свой ноутбук, снова Save & Run All | 1–3 ч |
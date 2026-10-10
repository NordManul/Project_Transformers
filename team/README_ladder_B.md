# Лестница бюджетов nGPT против GPT — аккаунт B

## Зачем

Мы хотим увидеть главный результат статьи nGPT: **nGPT доходит до того же качества, что GPT,
за в несколько раз меньшее число токенов, и этот выигрыш растёт с длиной обучения.**
Статья считает это «лестницей бюджетов»: каждую модель учат несколько раз с разным числом токенов
(у каждого бюджета своё полное расписание lr) и смотрят, на каком бюджете nGPT получает тот же
итоговый loss, что GPT на своём. Три аккаунта делят лестницу между собой; уже готовые прогоны
(150M и 775M) тоже входят в неё.

**Твоя часть (аккаунт B):** **Ступени nGPT: 1.4 млрд токенов (два lr), 450M и 300M**, плюс короткая ступень GPT 250M. Это вся лестница nGPT, которой у нас ещё нет.

| Прогон | Модель | Токенов | lr | Время на T4 |
|---|---|---|---|---|
| `ngpt_1.4B_lr0.003` | NGPT | 1.4B | 0.003 | ~15.4 ч |
| `ngpt_1.4B_lr0.002` | NGPT | 1.4B | 0.002 | ~15.4 ч |
| `ngpt_450M_lr0.0038` | NGPT | 450M | 0.0038 | ~5.0 ч |
| `ngpt_300M_lr0.0045` | NGPT | 300M | 0.0045 | ~3.3 ч |
| `gpt_250M_lr0.003` | GPT | 250M | 0.003 | ~1.8 ч |

Модель как во всех прошлых прогонах: 6 слоёв, d = 384, контекст 1024, батч 64 × 1024, сид 42.
Обе видеокарты T4 заняты все сессии. Нужно две сессии по ~11.5 ч. Квота: ~23 из 30 часов в неделю.
Вторая сессия — та же кнопка «Save & Run All», обучение продолжается с того же места.

---

## 1. Подготовка (один раз)

1. **Create → New Notebook**, переименуй в **`ngpt-ladder-b`**.
2. Справа: **Accelerator → GPU T4 x2**, **Internet → On**, **Input — пустой**.
3. Сделай **четыре ячейки с кодом** и скопируй в них текст ниже **целиком, ничего не меняя**.

### Ячейка 1 — настройки

```python
# ===== Настройки аккаунта B (ничего не менять) =====
JOBS = "ngpt:1.4B:3e-3,ngpt:1.4B:2e-3,ngpt:450M:3.8e-3,ngpt:300M:4.5e-3,gpt:250M:3e-3"   # модель:бюджет:lr
TRAIN_TOKENS = "1.5e9"   # данных больше, чем самый длинный прогон: без повторов
NAME = "ladder_B"

SESSION_HOURS = 11.5
REPO = "https://github.com/NordManul/Project_Transformers.git"
PLAN = ("--n_layer 6 --n_head 6 --d_model 384 --block_size 1024 --batch_size 64 --micro_batch 8 "
        f"--jobs {JOBS} --seed 42 --data_seed 1234")

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

### Ячейка 2 — данные (~22 минуты в первой сессии) и продолжение прошлой сессии

```python
found = sorted(glob.glob("/kaggle/input/**/data/owt/train.bin", recursive=True))
if os.path.exists(f"{DATA}/train.bin"):
    print("Данные уже на месте")
elif found and json.load(open(os.path.join(os.path.dirname(found[0]), "meta.json")))["train_tokens"] >= float(TRAIN_TOKENS):
    print("Копирую данные из входа:", os.path.dirname(found[0]))
    shutil.copytree(os.path.dirname(found[0]), DATA, dirs_exist_ok=True)
else:
    !python tokenize_dataset.py --dataset openwebtext --train_tokens {TRAIN_TOKENS} --val_tokens 1e7 --out {DATA}
print(open(f"{DATA}/meta.json").read())

prev = sorted(glob.glob(f"/kaggle/input/**/runs/{NAME}/plan.json", recursive=True))
if prev and not os.path.exists(f"{OUT}/plan.json"):
    print("Продолжаю прошлый запуск:", os.path.dirname(prev[0]))
    shutil.copytree(os.path.dirname(prev[0]), OUT, dirs_exist_ok=True)
else:
    print("Новый запуск" if not prev else "Прошлый запуск уже скопирован")
```

### Ячейка 3 — обучение

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

Сохрани (**Ctrl+S**). Вручную ячейки не запускай.

---

## 2. Сессия 1

1. Проверь: **GPU T4 x2**, **Internet On**, **Input пустой**.
2. **Save Version → Save & Run All (Commit) → Save**, затем выключи интерактивную сессию **⏻**.
   Браузер можно закрыть.
3. Через ~1 час: **Your Work → `ngpt-ladder-b` → Logs**. Должно быть:
   ```
   torch ... | GPU: ['Tesla T4', 'Tesla T4']        ← обязательно две
   PLAN: ... --jobs ngpt:1.4B:3e-3,ngpt:1.4B:2e-3,ngpt:450M:3.8e-3,ngpt:300M:4.5e-3,gpt:250M:3e-3 ...
   Новый запуск
   План: 5 прогонов, устройства ['0', '1'], лимит ... ч
   [0] ...: старт (...)
   [1] ...: старт (...)
   ```
   Если «ОШИБКА» — пришли весь текст ошибки в чат.
4. Через ~11.5 ч сессия закончится. В конце лога примерно так (это **нормально**):
   ```
[.] ngpt_1.4B_lr0.003: пауза
[.] ngpt_1.4B_lr0.002: пауза
   Не закончено: ... прогонов. Запустите ту же команду ещё раз, чтобы продолжить.
   ```

## 3. Сессия 2 — сразу после первой

1. Открой `ngpt-ladder-b` → **Edit**.
2. Справа **Add Input → Your Work → `ngpt-ladder-b` → Add** (это вывод первой сессии: данные и сохранённое обучение).
3. В ячейках **ничего не меняй**. GPU T4 x2, Internet On.
4. **Save Version → Save & Run All (Commit) → Save**, выключи **⏻**.
5. Через ~10 минут в Logs должно быть:
   ```
   Копирую данные из входа: /kaggle/input/...
   Продолжаю прошлый запуск: /kaggle/input/.../runs/ladder_B
   [0] ...: продолжаю (...)
   ```
6. В конце второй сессии:
   ```
[.] ngpt_1.4B_lr0.003: готово, val loss ...
[.] ngpt_1.4B_lr0.002: готово, val loss ...
[.] ngpt_450M_lr0.0038: готово, val loss ...
[.] ngpt_300M_lr0.0045: готово, val loss ...
[.] gpt_250M_lr0.003: готово, val loss ...
   Все прогоны закончены.
   ```
   Если вместо «Все прогоны закончены» снова «Не закончено» — запусти **третью сессию** так же, как вторую
   (во Input убери старую версию и добавь `ngpt-ladder-b` заново — подтянется последняя).

## 4. Сдать результат

1. **Share → Add collaborators → `vladimir337` → Can view** (нужно для общего отчёта).
2. Пришли в чат все строки `готово, val loss ...` из конца лога последней сессии.

---

## Коротко

| Шаг | Что делаешь | Время |
|---|---|---|
| 1 | Ноутбук `ngpt-ladder-b`, 4 ячейки, GPU T4 x2, Internet On, Input пустой | 5 мин |
| 2 | Сессия 1: Save & Run All, ⏻, через час проверить Logs | ~11.5 ч |
| 3 | Сессия 2: Add Input = `ngpt-ladder-b`, Save & Run All, ⏻ | ~11.5 ч |
| 4 | Share `vladimir337`, прислать строки «готово, val loss» | 2 мин |
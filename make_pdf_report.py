"""
Собирает отчёт эксперимента nGPT vs GPT в один PDF: текст отчёта (таблицы, ускорение)
и все графики с подписями, что на них смотреть.

Нужен только matplotlib. Берёт папку report/, которую создаёт ngpt_vs_gpt.py:
    report/report.md, report/ngpt_vs_gpt.png, report/details/*.png

    python make_pdf_report.py runs/ngpt_vs_gpt/report
    python make_pdf_report.py runs/ngpt_vs_gpt/report --out мой_отчёт.pdf
"""
import argparse
import re
import sys
import textwrap
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

A4 = (8.27, 11.69)   # дюймы
MARGIN_X, TOP, BOTTOM = 0.07, 0.95, 0.05
WRAP = 92            # символов в строке обычного текста

# Графики в порядке важности и что на каждом смотреть
FIGURES = [
    ("ngpt_vs_gpt.png", "Главный график",
     "Слева — итоговый val loss в зависимости от числа токенов обучения (на каждом бюджете лучший lr, "
     "у каждого бюджета своё расписание lr). Ниже — лучше. Если оранжевая линия nGPT ниже синей "
     "линии GPT, nGPT на этом бюджете лучше. В центре — подбор learning rate. Справа — ускорение nGPT "
     "по лестнице бюджетов, как в статье: во сколько раз меньше токенов нужно nGPT для того же loss "
     "(полые значки — только граница, бюджет вне диапазона)."),
    ("details/loss.png", "Кривые обучения",
     "Слева — train loss по ходу обучения, справа — val loss на фиксированном наборе окон. "
     "Ускорение видно так: проведите горизонтальную линию на уровне финального loss GPT и посмотрите, "
     "на скольких токенах до него дошёл nGPT."),
    ("details/grad_norm.png", "Градиенты и learning rate",
     "Слева — общая норма градиента (логарифмическая шкала), справа — расписание lr: "
     "у GPT разгон (warmup) и косинус до нуля, у nGPT сразу косинус без разгона."),
    ("details/architecture_params.png", "Обучаемые масштабы nGPT",
     "α (eigen learning rates) по слоям и масштаб логитов s_z. В начале обучения nGPT должен их «раскачать» — "
     "поэтому на коротких бюджетах он отстаёт."),
    ("details/layer_profile.png", "Профиль по глубине в конце обучения",
     "Норма выхода блоков, выбросы активаций (max/rms) и норма градиента по слоям. "
     "У nGPT норма выхода блока всегда 1 — все векторы на единичной сфере."),
    ("details/layers_over_time.png", "Активации и градиенты по слоям во времени",
     "Строки — величины, столбцы — прогоны; цвет линии — номер слоя (тёмный — первый, жёлтый — последний)."),
]


class Page:
    """Страница A4, на которую текст пишется сверху вниз; сама переходит на новую страницу."""

    def __init__(self, pdf):
        self.pdf, self.fig, self.y = pdf, None, TOP

    def new(self):
        self.close()
        self.fig = plt.figure(figsize=A4)
        self.y = TOP

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None

    def line(self, text, size=10, weight="normal", family="DejaVu Sans", gap=1.45, color="black"):
        step = size * gap / 72 / A4[1]
        if self.fig is None or self.y - step < BOTTOM:
            self.new()
        self.fig.text(MARGIN_X, self.y, text, fontsize=size, fontweight=weight, family=family,
                      color=color, va="top")
        self.y -= step

    def space(self, points=6):
        self.y -= points / 72 / A4[1]


def clean(text):
    """Убирает разметку markdown внутри строки: **жирный**, `код`."""
    return re.sub(r"\*\*(.+?)\*\*", r"\1", text).replace("`", "")


def table_lines(rows):
    """Строки markdown-таблицы -> выровненные столбцы моноширинным шрифтом."""
    cells = [[clean(c.strip()) for c in r.strip().strip("|").split("|")] for r in rows
             if not re.fullmatch(r"\|?[\s|:-]+\|?", r.strip())]
    widths = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(max(map(len, cells)))]
    out = []
    for n, r in enumerate(cells):
        out.append("  ".join(c.ljust(w) for c, w in zip(r, widths)))
        if n == 0:
            out.append("  ".join("─" * w for w in widths))
    return out


def render_markdown(page, text):
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i].rstrip()
        if raw.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            for k, row in enumerate(table_lines(block)):
                page.line(row, size=8.5, family="DejaVu Sans Mono", weight="bold" if k == 0 else "normal")
            page.space(6)
            continue
        i += 1
        if not raw.strip():
            page.space(5)
        elif raw.startswith("# "):
            continue   # заголовок документа пишется на титуле
        elif raw.startswith("## "):
            page.space(8)
            page.line(clean(raw[3:]), size=13, weight="bold")
            page.space(2)
        elif raw.startswith("- "):
            for k, part in enumerate(textwrap.wrap(clean(raw[2:]), WRAP - 4)):
                page.line(("•  " if k == 0 else "   ") + part)
        else:
            for part in textwrap.wrap(clean(raw), WRAP):
                page.line(part)


def figure_page(pdf, path, title, caption):
    img = plt.imread(path)
    h, w = img.shape[:2]
    size = (A4[1], A4[0]) if w > 2500 else A4      # широкие таблицы графиков — на альбомный лист
    fig = plt.figure(figsize=size)
    fig.text(MARGIN_X, TOP, title, fontsize=14, fontweight="bold", va="top")
    wrapped = textwrap.wrap(caption, WRAP if size == A4 else int(WRAP * 1.4))
    line = 0.018 * A4[1] / size[1]
    for k, part in enumerate(wrapped):
        fig.text(MARGIN_X, TOP - 1.6 * line - k * line, part, fontsize=9.5, va="top", color="#333333")
    top = TOP - 2.5 * line - len(wrapped) * line
    ratio = size[0] / size[1]
    width = 1 - 2 * MARGIN_X
    height = width * (h / w) * ratio                    # высота в долях страницы при полной ширине
    if height > top - BOTTOM:                           # высокая картинка — вписываем по высоте
        height = top - BOTTOM
        width = height / (h / w) / ratio
    ax = fig.add_axes([(1 - width) / 2, top - height, width, height])
    ax.imshow(img)
    ax.axis("off")
    fig.text(0.5, 0.025, Path(path).name, fontsize=7, color="#888888", ha="center")
    pdf.savefig(fig, dpi=200)
    plt.close(fig)


def build(report_dir, out):
    report_dir = Path(report_dir)
    md = report_dir / "report.md"
    if not md.exists():
        sys.exit(f"Нет {md}. Укажите папку report/ из runs/ngpt_vs_gpt (её создаёт ngpt_vs_gpt.py report).")
    with PdfPages(out) as pdf:
        page = Page(pdf)
        page.new()
        page.line("nGPT против GPT", size=22, weight="bold", gap=1.6)
        page.line("Нормализованный трансформер (arXiv:2410.01131) против обычного трансформера", size=11,
                  color="#444444")
        page.line(f"Отчёт собран {date.today():%d.%m.%Y} из {report_dir}", size=8, color="#888888")
        page.space(10)
        render_markdown(page, md.read_text(encoding="utf-8"))
        page.close()
        added = 0
        for name, title, caption in FIGURES:
            if (report_dir / name).exists():
                figure_page(pdf, report_dir / name, title, caption)
                added += 1
        known = {n for n, _, _ in FIGURES}
        for extra in sorted(report_dir.rglob("*.png")):   # всё, чего нет в списке выше
            rel = extra.relative_to(report_dir).as_posix()
            if rel not in known:
                figure_page(pdf, extra, rel, "")
                added += 1
        info = pdf.infodict()
        info["Title"] = "nGPT против GPT"
    print(f"Готово: {out} (текст отчёта + {added} графиков)")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser(description="Отчёт ngpt_vs_gpt.py -> один PDF")
    p.add_argument("report_dir", nargs="?", default="runs/ngpt_vs_gpt/report")
    p.add_argument("--out", default="ngpt_vs_gpt_report.pdf")
    args = p.parse_args()
    build(args.report_dir, args.out)


if __name__ == "__main__":
    main()
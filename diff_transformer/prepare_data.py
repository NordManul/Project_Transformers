"""
Скачивает часть датасета TinyStories (короткие простые рассказы на английском,
roneneldan/TinyStories на Hugging Face, лицензия CDLA-Sharing-1.0) и сохраняет
её в data/tinystories.txt.

На TinyStories даже маленькие модели (десятки миллионов параметров) учатся писать
связный текст — в отличие от Шекспира, где словарь и язык слишком сложные.
Весь файл весит ~2 ГБ, поэтому качаем только первые --mb мегабайт.

    python prepare_data.py              # 50 МБ (~12 млн токенов)
    python prepare_data.py --mb 200     # больше данных — лучше качество, дольше обучение
"""
import argparse
import os
import sys
import urllib.request

URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
       "TinyStoriesV2-GPT4-train.txt")
SEPARATOR = "<|endoftext|>"   # разделитель рассказов в исходном файле


def main():
    parser = argparse.ArgumentParser(description="Скачать часть TinyStories")
    parser.add_argument("--mb", type=int, default=50, help="сколько мегабайт текста скачать")
    parser.add_argument("--out", default=os.path.join("data", "tinystories.txt"))
    args = parser.parse_args()

    limit = args.mb * 1024 * 1024
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"Скачиваю первые {args.mb} МБ из {URL}")

    buf = bytearray()
    with urllib.request.urlopen(URL, timeout=60) as resp:
        while len(buf) < limit:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            buf.extend(chunk)
            print(f"\r  {len(buf) / 2**20:6.1f} / {args.mb} МБ", end="", flush=True)
    print()

    text = buf.decode("utf-8", errors="ignore")
    # отрезаем недокачанный последний рассказ
    cut = text.rfind(SEPARATOR)
    if cut > 0:
        text = text[:cut]
    # разделитель заменяем пустой строкой, чтобы токенизатор не резал его на мусорные токены
    stories = [s.strip() for s in text.split(SEPARATOR) if s.strip()]
    text = "\n\n".join(stories) + "\n"

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(f"Готово: {args.out} — {len(stories):,} рассказов, {len(text) / 2**20:.1f} МБ")
    print("\nДальше обучение:\n"
          f"  python train.py --arch diff --preset small --data {args.out} "
          "--max_iters 5000 --eval_every 500 --lr 6e-4 --dropout 0.1")


if __name__ == "__main__":
    sys.exit(main())

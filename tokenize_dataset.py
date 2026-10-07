"""
Токенизация датасета в бинарные файлы для compare.py и ngpt_vs_gpt.py.

Тексты скачиваются с Hugging Face потоком (весь датасет на диск не сохраняется),
кодируются токенизатором Mistral 7B (тем же, что у моделей) и пишутся подряд как uint16:
    <out>/val.bin     первые --val_tokens токенов (отложенная выборка)
    <out>/train.bin   следующие --train_tokens токенов
    <out>/meta.json   откуда данные и сколько токенов
Каждый документ начинается с BOS (id 1), как у LLaMA/Mistral.
Словарь 32 000 < 65 536, поэтому на токен уходит 2 байта: 1 млрд токенов = 2 ГБ.

    python tokenize_dataset.py --dataset openwebtext --train_tokens 2e9 --out data/owt
    python tokenize_dataset.py --dataset tinystories --train_tokens 5e8 --out data/tinystories
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DATASETS = {
    # статья nGPT обучала на OpenWebText; ~9 млрд токенов GPT-2, у Mistral примерно столько же
    "openwebtext": dict(path="Skylion007/openwebtext", split="train", field="text"),
    "tinystories": dict(path="roneneldan/TinyStories", split="train", field="text"),
}


def iter_texts(name):
    from datasets import load_dataset
    spec = DATASETS[name]
    ds = load_dataset(spec["path"], split=spec["split"], streaming=True)
    for row in ds:
        text = row[spec["field"]]
        if text and text.strip():
            yield text


def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def tokenize_to_bin(texts, tok, out_dir, train_tokens, val_tokens, docs_per_batch=512, log_every=50_000_000):
    """Пишет val.bin, затем train.bin, пока не наберётся нужное число токенов.
    Документ, на котором кончилась выборка, обрезается."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [("val", int(val_tokens)), ("train", int(train_tokens))]
    counts, docs = {"val": 0, "train": 0}, 0
    part = 0
    f = (out_dir / "val.bin.tmp").open("wb")
    began, next_log = time.time(), log_every
    for batch in batched(texts, docs_per_batch):
        encoded = tok(batch, add_special_tokens=False)["input_ids"]
        for ids in encoded:
            arr = np.asarray([tok.bos_token_id] + ids, dtype=np.uint16)
            docs += 1
            while len(arr):
                name, target = targets[part]
                take = arr[:target - counts[name]]
                f.write(take.tobytes())
                counts[name] += len(take)
                arr = arr[len(take):]
                if counts[name] == target:
                    f.close()
                    os.replace(out_dir / f"{name}.bin.tmp", out_dir / f"{name}.bin")
                    part += 1
                    if part == len(targets):
                        return counts, docs, time.time() - began
                    f = (out_dir / f"{targets[part][0]}.bin.tmp").open("wb")
                    arr = arr[:0]   # остаток документа не переносим в другую выборку
        total = counts["val"] + counts["train"]
        if total >= next_log:
            speed = total / (time.time() - began)
            left = (sum(t for _, t in targets) - total) / speed
            print(f"  {total / 1e6:,.0f}M токенов, {docs:,} документов | {speed / 1e6:.2f}M ток/с | "
                  f"осталось ~{left / 60:.0f} мин", flush=True)
            next_log += log_every
    f.close()
    name = targets[part][0]
    os.replace(out_dir / f"{name}.bin.tmp", out_dir / f"{name}.bin")
    print(f"Датасет кончился раньше: {counts}", flush=True)
    return counts, docs, time.time() - began


def parse_count(text):
    return int(float(text))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    parser = argparse.ArgumentParser(description="Датасет с Hugging Face -> train.bin / val.bin (uint16)")
    parser.add_argument("--dataset", default="openwebtext", choices=DATASETS)
    parser.add_argument("--train_tokens", type=parse_count, default=2_000_000_000)
    parser.add_argument("--val_tokens", type=parse_count, default=10_000_000)
    parser.add_argument("--out", default="data/owt")
    args = parser.parse_args()

    out = Path(args.out)
    if (out / "train.bin").exists() and (out / "meta.json").exists():
        print(f"{out} уже готов: {json.loads((out / 'meta.json').read_text(encoding='utf-8'))}")
        return

    import baseline_gpt as bg
    tok = bg.load_tokenizer()
    if not getattr(tok, "is_fast", False):
        print("Внимание: медленный (не fast) токенизатор — будет долго", flush=True)
    print(f"{args.dataset}: val {args.val_tokens:,} + train {args.train_tokens:,} токенов -> {out}", flush=True)
    counts, docs, seconds = tokenize_to_bin(iter_texts(args.dataset), tok, out, args.train_tokens, args.val_tokens)
    meta = dict(dataset=DATASETS[args.dataset]["path"], tokenizer=bg.TOKENIZER_REPO, vocab_size=tok.vocab_size,
                dtype="uint16", train_tokens=counts["train"], val_tokens=counts["val"], documents=docs,
                seconds=round(seconds))
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Готово за {seconds / 60:.1f} мин: {meta}", flush=True)


if __name__ == "__main__":
    main()
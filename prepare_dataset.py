"""Скачивает небольшую воспроизводимую выборку TinyStories потоково."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


REPO = "https://huggingface.co/datasets/roneneldan/TinyStories"
REVISION = "f54c09f"


def download_stories(filename, count, destination):
    url = f"{REPO}/resolve/{REVISION}/{filename}"
    request = urllib.request.Request(url, headers={"User-Agent": "Project-Transformers/1.0"})
    stories, lines = [], []
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw in response:
            line = raw.decode("utf-8")
            if line.strip() == "<|endoftext|>":
                story = "".join(lines).strip()
                lines = []
                if story:
                    stories.append(story)
                if len(stories) >= count:
                    break
            else:
                lines.append(line)
    if len(stories) != count:
        raise RuntimeError(f"{filename}: получено {len(stories)} историй вместо {count}")
    text = "\n\n".join(stories) + "\n"
    destination.write_text(text, encoding="utf-8")
    print(f"{destination}: {len(stories)} историй, {len(text):,} символов", flush=True)
    return dict(source=url, stories=count, sha256=hashlib.sha256(text.encode()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_stories", type=int, default=5000)
    parser.add_argument("--val_stories", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("data/tinystories"))
    args = parser.parse_args()
    if min(args.train_stories, args.val_stories) <= 0:
        parser.error("Количество историй должно быть положительным")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(dataset=REPO, revision=REVISION, language="English",
                    license="CDLA-Sharing-1.0", separator="two newlines")
    metadata["train"] = download_stories("TinyStories-train.txt", args.train_stories,
                                          args.output / "train.txt")
    metadata["validation"] = download_stories("TinyStories-valid.txt", args.val_stories,
                                               args.output / "val.txt")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()

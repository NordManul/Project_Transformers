"""
Обучает набор моделей как в абляции статьи «Differential Transformer» (табл. 6) на одних и тех же
данных с одинаковыми гиперпараметрами, затем запускает compare.py.

    python run_experiments.py                  # все 7 моделей, по 3000 шагов
    python run_experiments.py --quick          # только transformer, diff, diff_no_gn
    python run_experiments.py --steps 5000     # дольше — точнее
    python run_experiments.py --only diff      # одна модель

Уже обученные модели (checkpoints/ablation/<имя>.pt) пропускаются; --force — переобучить.
Обучение можно прервать Ctrl+C и запустить снова — продолжится со следующей модели.

Соответствие статье (пресет small: d_model 384, 6 «обычных» голов по 64):
  transformer        Transformer, 6 голов x 64            (статья: 16 голов x 128)
  transformer_8h     Transformer, 3 головы x 128          (статья: 8 x 256 — вдвое меньше голов)
  transformer_8h_gn  то же + GroupNorm на голову
  diff               DIFF, 3 головы, d = 64 (V = 128), GroupNorm, λ_init по слоям
  diff_no_gn         DIFF без GroupNorm
  diff_lambda08      DIFF с постоянным λ_init = 0.8
  diff_lambda05      DIFF с постоянным λ_init = 0.5
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

EXPERIMENTS = {
    "transformer":       ["--arch", "baseline"],
    "transformer_8h":    ["--arch", "baseline", "--n_head", "3"],
    "transformer_8h_gn": ["--arch", "baseline", "--n_head", "3", "--head_norm", "on"],
    "diff":              ["--arch", "diff"],
    "diff_no_gn":        ["--arch", "diff", "--head_norm", "off"],
    "diff_lambda08":     ["--arch", "diff", "--lambda_init", "0.8"],
    "diff_lambda05":     ["--arch", "diff", "--lambda_init", "0.5"],
}
QUICK = ["transformer", "diff", "diff_no_gn"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "tinystories.txt"))
    parser.add_argument("--preset", default="small")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--quick", action="store_true", help="только transformer, diff, diff_no_gn")
    parser.add_argument("--only", nargs="+", choices=EXPERIMENTS, default=None)
    parser.add_argument("--force", action="store_true", help="переобучить уже готовые модели")
    parser.add_argument("--no_compare", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        sys.exit(f"Нет {args.data}. Сначала: python prepare_data.py --mb 50")
    names = args.only or (QUICK if args.quick else list(EXPERIMENTS))
    out_dir = os.path.join("checkpoints", "ablation")
    os.makedirs(out_dir, exist_ok=True)
    common = ["--preset", args.preset, "--data", args.data, "--max_iters", str(args.steps),
              "--eval_every", "500", "--lr", str(args.lr), "--batch_size", str(args.batch_size),
              "--dropout", "0", "--patience", "0", "--warmup", "200", "--seed", "42"]

    for i, name in enumerate(names, 1):
        out = os.path.join(out_dir, f"{name}.pt")
        done = out + ".done"           # метка, что обучение дошло до конца (не прервано)
        if os.path.exists(out) and os.path.exists(done) and not args.force:
            print(f"[{i}/{len(names)}] {name}: уже обучена ({out}), пропускаю")
            continue
        cmd = [sys.executable, os.path.join(HERE, "train.py"), *common, *EXPERIMENTS[name], "--name", name, "--out", out]
        print(f"\n[{i}/{len(names)}] {name}\n  " + " ".join(cmd), flush=True)
        try:
            subprocess.run(cmd, check=True)
            open(done, "w").close()
        except KeyboardInterrupt:
            print("\nОстановлено. Запустите снова — готовые модели будут пропущены.")
            return
        except subprocess.CalledProcessError as e:
            if e.returncode == 130:
                print("\nОстановлено. Запустите снова — готовые модели будут пропущены.")
                return
            sys.exit(f"Обучение {name} завершилось с ошибкой (код {e.returncode}).")

    if not args.no_compare:
        paths = [os.path.join(out_dir, f"{n}.pt") for n in EXPERIMENTS if os.path.exists(os.path.join(out_dir, f"{n}.pt"))]
        subprocess.run([sys.executable, os.path.join(HERE, "compare.py"), *paths], check=False)


if __name__ == "__main__":
    main()

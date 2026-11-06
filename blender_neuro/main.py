#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
blender_neuro — домашний симулятор самообучающихся существ.

Окно:  python main.py                       (задача walk, гуманоид)
       python main.py --task target --creature fox
       python main.py --task balance --window 1000x700

Турбо-обучение без окна (быстрее всего):
       python main.py --headless --task walk --steps 10000000 --nenvs 6

Просмотр обученной модели:
       python main.py --task walk --load outputs/ckpt_walk_human.zip --watch
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_window(s):
    try:
        w, h = s.lower().split("x")
        return max(320, int(w)), max(240, int(h))
    except Exception:
        return (800, 600)


def main():
    ap = argparse.ArgumentParser(description="blender_neuro: RL-симулятор существ")
    ap.add_argument("--task", default="walk", choices=["walk", "balance", "target"],
                    help="задача обучения (default: walk)")
    ap.add_argument("--creature", default="human",
                    help="пресет существа: human, fox, wolf, bear, hare, badger, "
                         "cat, hedgehog или путь к .json")
    ap.add_argument("--window", default="800x600", help="размер окна, напр. 800x600")
    ap.add_argument("--load", default=None, help="путь к чекпоинту .zip")
    ap.add_argument("--device", default="auto", help="auto|cuda|cpu")
    ap.add_argument("--speed", type=int, default=2, help="шагов просмотра на кадр")
    ap.add_argument("--watch", action="store_true",
                    help="сразу показывать политику в окне")
    # headless
    ap.add_argument("--headless", action="store_true", help="обучение без окна")
    ap.add_argument("--steps", type=int, default=5_000_000, help="всего шагов обучения")
    ap.add_argument("--nenvs", type=int, default=4, help="параллельных сред")
    ap.add_argument("--save", default=None, help="путь сохранения модели (headless)")
    args = ap.parse_args()

    if args.headless:
        from neurosim.train import train_headless
        base = args.save or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "outputs", f"ckpt_{args.task}_{os.path.splitext(os.path.basename(args.creature))[0]}")
        os.makedirs(os.path.dirname(base), exist_ok=True)

        def progress(done, total):
            print(f"\rобучение: {done}/{total} шагов "
                  f"({100 * done // max(1, total):d}%)", end="", flush=True)

        print(f"Турбо-обучение: задача={args.task}, существо={args.creature}, "
              f"сред={args.nenvs}")
        train_headless(args.task, _creature_path(args.creature), args.steps,
                       base, nenvs=args.nenvs, load_path=args.load,
                       device=args.device, progress_cb=progress)
        print(f"\nГотово. Модель: {base}.zip")
        print("Просмотр:  python main.py --task %s --load %s.zip --watch"
              % (args.task, base))
        return

    from neurosim.gui import App
    app = App(task=args.task, creature=args.creature,
              window=parse_window(args.window), load=args.load,
              device=args.device, speed=args.speed, watch=args.watch)
    if args.watch and app.model is None:
        print("Чекпоинт не найден — показ случайной политики. "
              "Сначала обучите (T или --headless).")
    try:
        app.run()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()


def _creature_path(creature):
    from neurosim.gui import resolve_creature
    return resolve_creature(creature)


if __name__ == "__main__":
    main()

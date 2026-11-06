# -*- coding: utf-8 -*-
"""
neurosim.gui — Окно-симулятор (PyBullet GUI, разворачивается, старт 800x600).

Кнопки прямо в окне + горячие клавиши:
  1/2/3 — задача (ходьба / баланс / цель)
  W     — Смотреть (вкл/выкл показ обученной политики)
  T     — Турбо-обучение (фоном, без графики) на выбранные миллионы шагов
  S     — Стоп турбо
  R     — Сброс существа
  K     — Сохранить чекпоинт
  E     — Экспорт BVH + JSON для Blender
  C     — Камера-слежение вкл/выкл
  SPACE — пауза/шаг
"""
import os
import threading
import time

import numpy as np
import pybullet as p

from . import export as EX
from . import spec as S
from .env import NeuroEnv
from .train import make_model, run_greedy_episode

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRESETS = os.path.join(ROOT, "presets")
OUTPUTS = os.path.join(ROOT, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

TASKS = ["walk", "balance", "target"]
TASK_RU = {"walk": "Ходьба", "balance": "Баланс", "target": "Цель"}
TURBO_STEPS = [1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000]


def resolve_creature(creature):
    """'human'/'fox'/... -> путь к пресету; путь .json -> как есть."""
    if creature.endswith(".json"):
        return creature
    path = os.path.join(PRESETS, f"{creature}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"нет пресета: {creature} ({path})")
    return path


class App:
    def __init__(self, task="walk", creature="human", window=(800, 600),
                 load=None, device="auto", speed=2, watch=False):
        self.TASKS = TASKS
        self.task_idx = TASKS.index(task) if task in TASKS else 0
        self.creature_path = resolve_creature(creature)
        self.window = window
        self.load_arg = load
        self.device = device
        self.speed = speed
        self.model = None
        self.training = False
        self.stop_flag = False
        self.watch = False
        self.follow = True
        self.paused = False
        self.record_buf = []
        self.stats = {"steps": 0, "target": 0, "episodes": 0, "reward": 0.0}
        self._btn_prev = {}
        self._ui_ids = {}
        self._hud_id = None
        self._target = [0.0, 0.0, 1.0]
        self.watch = watch

        w, h = window
        self.cid = p.connect(p.GUI, options=f"--width={w} --height={h}")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
        self._build_task()
        if load:
            self.model = make_model(None, device=device, load_path=load)

    # ------------------------------------------------------------- построение

    def _build_task(self):
        task = TASKS[self.task_idx]
        self.env = NeuroEnv(task=task, creature_path=self.creature_path,
                            client_id=self.cid, connect_gui=False,
                            record_buf=self.record_buf,
                            window=self.window)
        self.spec = self.env.spec
        self.model = None if self._dims_changed() else self.model
        p.resetDebugVisualizerCamera(3.2, -50, -25,
                                     [self.env.spawn[0], 0, 1.0])
        self._make_ui()

    def _dims_changed(self):
        return False  # модель и так пересоздаётся при смене задачи

    # ------------------------------------------------------------------- UI

    def _make_ui(self):
        self._ui_ids = {}
        self._btn_prev = {}
        items = [
            ("btn_watch", "1: Смотреть (W)"),
            ("btn_turbo", "2: Турбо-обучение (T)"),
            ("btn_stop", "3: Стоп турбо (S)"),
            ("btn_reset", "4: Сброс существа (R)"),
            ("btn_save", "5: Сохранить модель (K)"),
            ("btn_export", "6: Экспорт BVH+JSON (E)"),
            ("btn_cam", "7: Камера-слежение (C)"),
        ]
        for key, label in items:
            self._ui_ids[key] = p.addUserDebugParameter(label, 0.0, 1.0, 0.0)
        self._ui_ids["task"] = p.addUserDebugParameter(
            "Задача: 0=ходьба 1=баланс 2=цель", 0.0, 2.0, float(self.task_idx))
        self._ui_ids["speed"] = p.addUserDebugParameter(
            "Скорость просмотра (шагов/кадр)", 1.0, 20.0, float(self.speed))
        self._ui_ids["mill"] = p.addUserDebugParameter(
            "Турбо, млн шагов (1/2/5/10/20/50)", 0.0, 5.0, 2.0)

    def _read_ui(self):
        def pressed(key):
            v = p.readUserDebugParameter(self._ui_ids[key])
            was = self._btn_prev.get(key, 0.0)
            self._btn_prev[key] = v
            return v > 0.5 and was <= 0.5

        if pressed("btn_watch"):
            self.watch = not self.watch
        if pressed("btn_turbo") and not self.training:
            self.start_turbo()
        if pressed("btn_stop"):
            self.stop_flag = True
        if pressed("btn_reset"):
            self.env.reset()
        if pressed("btn_save"):
            self.save_model()
        if pressed("btn_export"):
            self.export_all()
        if pressed("btn_cam"):
            self.follow = not self.follow

        t = int(round(p.readUserDebugParameter(self._ui_ids["task"])))
        if t != self.task_idx:
            self.task_idx = t
            self._rebuild()
        self.speed = int(p.readUserDebugParameter(self._ui_ids["speed"]))

    def _rebuild(self):
        self.env.close()
        self.record_buf.clear()
        if hasattr(self, "_last_obs"):
            del self._last_obs
        self.model = None
        self.watch = False
        self._build_task()

    # -------------------------------------------------------------- обучение

    def start_turbo(self):
        idx = int(round(p.readUserDebugParameter(self._ui_ids["mill"])))
        mill = TURBO_STEPS[idx]
        self.stop_flag = False
        self.training = True
        self._turbo_thread = threading.Thread(
            target=self._turbo_worker, args=(mill,), daemon=True)
        self._turbo_thread.start()

    def _ckpt_path(self):
        base = os.path.splitext(os.path.basename(self.creature_path))[0]
        return os.path.join(OUTPUTS, f"ckpt_{TASKS[self.task_idx]}_{base}")

    def _turbo_worker(self, total_steps):
        try:
            env2 = NeuroEnv(task=TASKS[self.task_idx],
                            creature_path=self.creature_path, seed=777)
            ckpt = self._ckpt_path()
            load = self.load_arg or (ckpt + ".zip" if os.path.exists(ckpt + ".zip") else None)
            if self.model is None:
                self.model = make_model(env2, device=self.device, load_path=load)
            per = max(50_000, total_steps // 40)
            while self.model.num_timesteps < total_steps and not self.stop_flag:
                self.model.learn(min(per, total_steps - self.model.num_timesteps),
                                 reset_num_timesteps=False)
                self.stats["steps"] = self.model.num_timesteps
                self.stats["target"] = total_steps
                if self.model.num_timesteps % 1_000_000 < per:
                    self.model.save(ckpt)  # периодический чекпоинт
            self.model.save(ckpt)
            self.watch = True  # показать результат
        except Exception as e:  # чтобы GUI не молча умирал
            self.stats["error"] = str(e)
        finally:
            self.training = False

    def save_model(self):
        if self.model is not None:
            self.model.save(self._ckpt_path())

    # ---------------------------------------------------------------- экспорт

    def export_all(self):
        if len(self.record_buf) < 20:
            if self.model is None:
                self.stats["error"] = "экспорт: нет модели (сначала обучите)"
                return False
            try:
                run_greedy_episode(self.model, self.env, max_steps=400)
            except Exception as e:
                self.stats["error"] = f"экспорт: {e}"
                return False
        if len(self.record_buf) < 2:
            return False
        stamp = time.strftime("%Y%m%d_%H%M%S")
        name = self.spec["name"]
        base = os.path.join(OUTPUTS, f"{TASKS[self.task_idx]}_{name}_{stamp}")
        EX.export_bvh(self.record_buf, self.spec, base + ".bvh", fps=48.0)
        EX.export_motion_json(self.record_buf, self.spec, base + ".json",
                              world_def=self.env.world_def,
                              task=TASKS[self.task_idx])
        self.record_buf.clear()
        self.last_export = base
        return True

    # ------------------------------------------------------------------- кадр

    def _step_watch(self):
        obs = self._last_obs if hasattr(self, "_last_obs") else None
        if obs is None:
            obs, _ = self.env.reset()
        if self.model is not None and not self.paused:
            action, _ = self.model.predict(obs, deterministic=True)
        else:
            action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        obs, r, term, trunc, _ = self.env.step(action)
        self._last_obs = obs
        self.stats["reward"] = self.stats["reward"] * 0.95 + r * 0.05
        if term or trunc:
            obs, _ = self.env.reset()
            self._last_obs = obs
            self.stats["episodes"] += 1

    def _camera_follow(self):
        if not self.follow:
            return
        info = p.getDebugVisualizerCamera()
        dist, yaw, pitch = info[2], info[3], info[4]
        pos, _ = self.env.creature.root_state()
        self._target = [self._target[k] * 0.9 + pos[k] * 0.1 for k in range(3)]
        p.resetDebugVisualizerCamera(dist, yaw, pitch, self._target)

    def _hud(self):
        txt = (f"Задача: {TASK_RU[TASKS[self.task_idx]]} | "
               f"Существо: {self.spec['name']} | "
               f"{'ТУРБО-ОБУЧЕНИЕ ' + str(self.stats.get('steps', 0)) + ' шагов' if self.training else ('СМОТРИМ' if self.watch else 'ПАУЗА')}")
        if self.training:
            txt = (f"[ТУРБО] {self.stats.get('steps', 0)} / "
                   f"{self.stats.get('target', '?')} шагов")
        elif self.stats.get("error"):
            txt = f"ОШИБКА ОБУЧЕНИЯ: {self.stats['error'][:60]}"
        else:
            r = self.stats.get("reward", 0.0)
            txt += f" | reward~{r:.1f}"
        pos = list(self._target) if self._target else [0, 0, 1]
        pos[2] += 0.6
        if self._hud_id is None:
            self._hud_id = p.addUserDebugTextItem(txt, pos,
                                                  textSize=1.3,
                                                  textColorRGB=[1, 1, 0.4])
        else:
            p.addUserDebugTextItem(txt, pos, textSize=1.3,
                                   textColorRGB=[1, 1, 0.4],
                                   replaceItemUniqueId=self._hud_id)

    # ------------------------------------------------------------- главный цикл

    def run(self):
        keys = {"1": 0, "2": 1, "3": 2}
        while True:
            try:
                kb = p.getKeyboardEvents()
            except Exception:
                kb = {}
            for key, state in kb.items():
                if not (state & p.KEY_WAS_TRIGGERED):
                    continue
                ch = None
                if isinstance(key, int):
                    ch = chr(key) if 32 <= key < 127 else None
                if key in (p.B3G_SPACE,):
                    self.paused = not self.paused
                elif ch == "W":
                    self.watch = not self.watch
                elif ch == "T" and not self.training:
                    self.start_turbo()
                elif ch == "S":
                    self.stop_flag = True
                elif ch == "R":
                    self.env.reset()
                elif ch == "K":
                    self.save_model()
                elif ch == "E":
                    self.export_all()
                elif ch == "C":
                    self.follow = not self.follow
                elif ch in keys:
                    self.task_idx = keys[ch]
                    self._rebuild()

            self._read_ui()

            if self.watch and not self.training and not self.paused:
                for _ in range(max(1, self.speed)):
                    self._step_watch()
            self._camera_follow()
            self._hud()
            time.sleep(0.01)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
        try:
            p.disconnect(self.cid)
        except Exception:
            pass

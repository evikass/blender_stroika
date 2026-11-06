# -*- coding: utf-8 -*-
"""
neurosim.env — Gymnasium-окружение: существо + мир + задача.

Три задачи:
  walk    — идти вперёд по полосе препятствий (награда за скорость и стойку)
  balance — стоять и выдерживать случайные толчки
  target  — дойти до жёлтого маркера в лесном мире

Наблюдение: кватернион корня, высота, локальная скорость, угловая скорость,
углы и скорости всех суставов, контакты стоп, вектор на цель.
Действие: [-1,1] по каждому суставу -> целевой угол вокруг позы покоя.
"""
import numpy as np
import gymnasium as gym
import pybullet as p

from . import spec as S
from .build import Creature
from . import world as W

CONTROL_DT = 5 / 240.0          # 48 Гц
MAX_EP_STEPS = 800              # ~16.7 c


class NeuroEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, task="walk", creature_path=None, creature_spec=None,
                 client_id=None, connect_gui=False, spawn=(0.0, 0.0, 0.04),
                 record_buf=None, window=(800, 600)):
        super().__init__()
        self.task_name = task
        if client_id is None:
            if connect_gui:
                w, h = window
                client_id = p.connect(p.GUI, options=f"--width={w} --height={h}")
            else:
                client_id = p.connect(p.DIRECT)
        self.client = p
        self.cid = client_id
        self.spawn = list(spawn)
        self.record_buf = record_buf  # внешний список кадров или None

        self.spec = creature_spec or S.load(creature_path)
        self.links = self.spec["links"]
        self.n_joints = len(S.actuated(self.spec))
        self.root_rest_z = self.spec["links"][0]["com"][2]
        self.total_mass = sum(l["mass"] for l in self.spec["links"])

        n_obs = 16 + 2 * self.n_joints
        self.observation_space = gym.spaces.Box(-15.0, 15.0, shape=(n_obs,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(self.n_joints,),
                                           dtype=np.float32)

        self._build()
        self._steps = 0
        self._goal = np.zeros(3)
        self._d_prev = None
        self._push_step = -1
        self._push_force = np.zeros(3)

    # ------------------------------------------------------------------ мир

    def _build(self):
        wname = {"walk": "walk", "balance": "empty", "target": "forest"}[self.task_name]
        wd = W.WORLDS[wname]() if isinstance(wname, str) else wname
        self.world_def = wd
        self.ground, self.props = W.build_world(self.client, wd)
        self.creature = Creature(self.spec, self.client,
                                 base_pos=[self.spawn[0], self.spawn[1],
                                           self.spawn[2]])
        self.creature.hold_rest()
        self.goal_id = None
        if self.task_name == "target":
            self._new_goal()
        for _ in range(60):
            self.client.stepSimulation()

    def _new_goal(self):
        ang = self.np_random.uniform(-0.9, 0.9)
        r = self.np_random.uniform(2.5, 5.0)
        self._goal = np.array([self.spawn[0] + r * np.cos(ang) + 2.0,
                               r * np.sin(ang), 0.0])
        if self.goal_id is not None:
            self.client.removeBody(self.goal_id)
        self.goal_id = W.add_goal_marker(self.client,
                                         [self._goal[0], self._goal[1], 0.3])

    # ------------------------------------------------------------------ gym

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.client.removeBody(self.creature.body)
        self.creature = Creature(self.spec, self.client,
                                 base_pos=[self.spawn[0], self.spawn[1],
                                           self.spawn[2]])
        self.creature.hold_rest()
        if self.goal_id is not None:
            self.client.removeBody(self.goal_id)
            self.goal_id = W.add_goal_marker(self.client,
                                             [self._goal[0], self._goal[1], 0.3])
        self._steps = 0
        self._d_prev = None
        self._push_step = -1
        if self.task_name == "target":
            self._new_goal()
        for _ in range(60):
            self.client.stepSimulation()
        return self._obs(), {}

    def _obs(self):
        pos, quat = self.creature.root_state()
        q, qd = self.creature.joint_states()
        vel, ang = self.client.getBaseVelocity(self.creature.body)
        # скорость в локальной рамке корня
        v_local = S.quat_rotate(S.quat_conj(quat), vel)
        contacts = self.creature.contacts_with(self.ground)
        if len(contacts) < 2:
            contacts = contacts + [0.0] * (2 - len(contacts))
        goal_local = [0.0, 0.0, 0.0]
        if self.task_name == "target":
            d = np.array(self._goal) - np.array(pos)
            goal_local = S.quat_rotate(S.quat_conj(quat), d.tolist())
        obs = np.array(list(quat) + [pos[2]] + list(v_local) + list(ang)
                       + q + qd + contacts + list(goal_local), dtype=np.float32)
        return obs

    def _pushes(self):
        """Случайные толчки для задачи баланса (масштаб по массе существа)."""
        if self.task_name != "balance":
            return
        if self._push_step < 0:
            self._push_step = self._steps + int(self.np_random.integers(60, 160))
            ang = self.np_random.uniform(0, 2 * np.pi)
            # дельта-v 0.25–0.75 м/с независимо от массы существа
            mag = self.total_mass * self.np_random.uniform(0.25, 0.75)
            self._push_force = mag * np.array([np.cos(ang), np.sin(ang), 0])
            self._push_active = True

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        self.creature.apply_action(action)
        chest = self.creature.name2idx.get("chest", 0)
        push_now = (self.task_name == "balance" and self._push_step == self._steps)
        for _ in range(5):
            if push_now:
                self.client.applyExternalForce(
                    self.creature.body, chest,
                    forceObj=self._push_force.tolist(),
                    posObj=self.creature.root_state()[0],
                    flags=self.client.WORLD_FRAME)
            self.client.stepSimulation()

        self._steps += 1
        if self.record_buf is not None and len(self.record_buf) < 20000:
            self.record_buf.append(self._frame())

        pos, quat = self.creature.root_state()
        alive = not self.creature.is_fallen()
        reward = 0.0
        terminated = False

        if self.task_name == "walk":
            vel, _ = self.client.getBaseVelocity(self.creature.body)
            v_local = S.quat_rotate(S.quat_conj(quat), vel)
            reward = 3.0 * v_local[0] + 0.5 * alive - 0.001 * float(np.sum(action ** 2))
        elif self.task_name == "balance":
            reward = (0.2 * alive + 0.8 * max(0.0, self.creature.root_up_z())
                      - 0.001 * float(np.sum(action ** 2)))
            self._pushes()
        elif self.task_name == "target":
            d = float(np.linalg.norm(np.array(self._goal[:2]) - np.array(pos[:2])))
            if self._d_prev is None:
                self._d_prev = d
            reward = 2.0 * (self._d_prev - d) + 0.3 * alive - 0.02
            self._d_prev = d
            if d < 0.7:
                reward += 15.0
                terminated = True

        if not alive:
            terminated = True
        truncated = self._steps >= MAX_EP_STEPS
        return self._obs(), float(reward), bool(terminated), bool(truncated), {}

    def _frame(self):
        pos, quat = self.creature.root_state()
        q, _ = self.creature.joint_states()
        return {"root_pos": [round(v, 5) for v in pos],
                "root_quat": [round(v, 6) for v in quat],
                "joints": [round(v, 5) for v in q]}

    def close(self):
        try:
            self.client.disconnect(self.cid)
        except Exception:
            pass

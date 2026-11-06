# -*- coding: utf-8 -*-
"""
Самопроверка установки blender_neuro (headless, без окна):
  1) гуманоид собирается и стоит на ногах
  2) кинематика аддона/BVH совпадает с физикой
  3) среда всех трёх задач работает
  4) PPO делает шаг обучения и сохраняется
  5) BVH и JSON экспортируются
Запуск:  python tests/test_headless.py
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np


def main():
    from neurosim import spec as S
    from neurosim.build import Creature
    from neurosim import world as W
    from neurosim.env import NeuroEnv
    from neurosim.fk import fk_com_positions
    from neurosim import export as EX
    import pybullet as p

    human = os.path.join(ROOT, "presets", "human.json")
    print("1) сборка и стойка гуманоида...")
    cid = p.connect(p.DIRECT)
    spec = S.load(human)
    W.build_world(p, W.make_world_empty())
    c = Creature(spec, p, base_pos=[0, 0, 0.04])
    c.hold_rest()
    for _ in range(480):
        p.stepSimulation()
    pos, _ = c.root_state()
    assert abs(pos[2] - 1.0) < 0.15, f"гуманоид не стоит: z={pos[2]:.2f}"
    print(f"   OK, стоит (z={pos[2]:.2f} м)")

    print("2) сверка FK с физикой в случайной позе...")
    rng = np.random.default_rng(0)
    angles = rng.uniform(-0.5, 0.5, c.n_joints)
    c.motors_off()
    for lid, a in zip(c.joint_ids, angles):
        p.resetJointState(c.body, lid, a, 0.0)
    p.performCollisionDetection()
    states = c.link_states_com()
    jd = dict(zip(c.joint_names, angles))
    fk = fk_com_positions(spec, states[0][0], states[0][1], jd)
    err = max(float(np.linalg.norm(np.array(a) - np.array(b)))
              for a, (b, _) in zip(fk, states))
    assert err < 1e-4, f"FK расходится: {err}"
    print(f"   OK, макс. расхождение {err * 1000:.3f} мм")
    p.disconnect()

    print("3) среды трёх задач...")
    for task in ("walk", "balance", "target"):
        env = NeuroEnv(task=task, creature_path=human)
        obs, _ = env.reset(seed=1)
        for _ in range(30):
            obs, r, term, trunc, _ = env.step(
                np.zeros(env.action_space.shape))
            if term or trunc:
                env.reset()
        env.close()
    print("   OK")

    print("4) PPO: шаг обучения + сохранение...")
    from neurosim.train import make_model
    env = NeuroEnv(task="walk", creature_path=human)
    model = make_model(env, device="cpu", seed=0)
    model.learn(total_timesteps=2048)
    tmp = os.path.join(tempfile.gettempdir(), "neuro_selftest_ckpt")
    model.save(tmp)
    from stable_baselines3 import PPO
    PPO.load(tmp)
    env.close()
    print("   OK")

    print("5) экспорт BVH + JSON...")
    buf = []
    env = NeuroEnv(task="walk", creature_path=human, record_buf=buf)
    obs, _ = env.reset()
    for _ in range(120):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    out = os.path.join(tempfile.gettempdir(), "neuro_selftest_out")
    bvh = EX.export_bvh(buf, spec, os.path.join(out, "selftest.bvh"))
    js = EX.export_motion_json(buf, spec, os.path.join(out, "selftest.json"),
                               world_def=env.world_def, task="walk")
    lines = open(bvh).read().splitlines()
    assert f"Frames: {len(buf)}" in lines
    data = json.load(open(js))
    assert data["format"] == "neurosim-motion" and len(data["frames"]) == len(buf)
    env.close()
    print(f"   OK ({len(buf)} кадров): {bvh}")
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ — установка работает!")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
neurosim.train — Обучение (PPO из Stable-Baselines3).

Турбо-режим: DIRECT-клиент PyBullet (без графики), тысячи шагов/с.
На RTX 5060 Ti поставьте torch с CUDA 12.8:
    pip install torch --index-url https://download.pytorch.org/whl/cu128
Физика считается на CPU, сеть PPO — на GPU.
"""
import numpy as np

from .env import NeuroEnv

DEFAULT_HYPER = dict(
    n_steps=2048,
    batch_size=2048,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    # старт близко к "спокойной стойке" (std≈0.14 рад): существо сначала
    # стоит, потом учится двигаться — иначе падает раньше, чем учится
    policy_kwargs=dict(net_arch=[256, 256], log_std_init=-2.0),
)


def make_model(env, device="auto", load_path=None, seed=0):
    from stable_baselines3 import PPO
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if load_path:
        model = PPO.load(load_path, env=env, device=device)
    else:
        model = PPO("MlpPolicy", env, device=device, verbose=0, seed=seed,
                    **DEFAULT_HYPER)
    return model


def make_vec_env(task, creature_path, nenvs=4, seed=0):
    from stable_baselines3.common.vec_env import (DummyVecEnv,
                                                  SubprocVecEnv)

    def _fn(rank):
        def _init():
            env = NeuroEnv(task=task, creature_path=creature_path)
            env.reset(seed=seed + rank)
            return env
        return _init

    if nenvs > 1:
        try:
            return SubprocVecEnv([_fn(i) for i in range(nenvs)])
        except Exception:
            pass
    return DummyVecEnv([_fn(0)])


def train_headless(task, creature_path, total_steps, save_path,
                   nenvs=4, load_path=None, device="auto", seed=0,
                   progress_cb=None, chunk=100_000):
    """Турбо-обучение без графики. Возвращает модель."""
    venv = make_vec_env(task, creature_path, nenvs=nenvs, seed=seed)
    model = make_model(venv, device=device, load_path=load_path, seed=seed)
    last = model.num_timesteps
    while model.num_timesteps < total_steps:
        n = min(chunk, total_steps - model.num_timesteps)
        model.learn(n, reset_num_timesteps=False)
        if progress_cb:
            progress_cb(model.num_timesteps, total_steps)
        last = model.num_timesteps
    model.save(save_path)
    venv.close()
    return model


def run_greedy_episode(model, env, record_buf=None, max_steps=800):
    """Прогнать обученную политику (deterministic) один эпизод, писать кадры."""
    obs, _ = env.reset()
    total = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, _ = env.step(action)
        total += r
        if term or trunc:
            break
    return total

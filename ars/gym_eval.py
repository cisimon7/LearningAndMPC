import gym
import numpy as np
import torch as th
from typing import Callable, Any
from .Normalizer import Normalizer


def ars_policy_eval(
        eval_env: gym.Env,
        eval_policy: th.nn.Module,
        policy_params_path: str = None,
        eval_steps=10,
        on_step: Callable[[float, int], Any] = lambda goodness, step: None,
):
    if policy_params_path is not None:
        eval_policy.load_state_dict(th.load(policy_params_path))

    reward_sequence = []

    for i in range(eval_steps):
        x0, _ = eval_env.reset()
        done, fitness = False, 0
        while not done:
            action = eval_policy(th.from_numpy(x0)).detach().numpy()
            x0, reward, terminated, truncated, _ = eval_env.step(action)
            fitness += reward
            done = terminated or truncated

        on_step(fitness, i)
        reward_sequence.append(np.asarray(fitness))

    avg_reward = np.mean(reward_sequence)
    print(f"Final {avg_reward=}")

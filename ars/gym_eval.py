import gym
import numpy as np
import torch as th
from typing import Callable, Any
from .Normalizer import Normalizer


def ars_policy_eval(
        eval_env: gym.Env,
        eval_policy: th.nn.Module,
        policy_post_process: Callable[[th.Tensor], np.ndarray] = lambda tensor: tensor.detach().numpy(),
        eval_normalizer: Normalizer = None,
        policy_params_path: str = None,
        normalizer_params_path: str = None,
        eval_steps=10,
        on_step: Callable[[float, int], Any] = lambda goodness, step: None,
):
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = 1 if isinstance(eval_env.action_space, gym.spaces.Discrete) else eval_env.action_space.shape[0]

    if policy_params_path is not None:
        eval_policy.load_state_dict(th.load(policy_params_path))

    eval_normalizer = eval_normalizer if eval_normalizer else Normalizer(obs_dim)
    if normalizer_params_path is not None:
        eval_normalizer.load_state(normalizer_params_path)

    reward_sequence = []

    def policy(state):
        state = eval_normalizer.normalize(th.from_numpy(state))
        state = th.FloatTensor(state)
        return eval_policy(state)

    for i in range(eval_steps):

        x0, _ = eval_env.reset()
        done, fitness = False, 0
        while not done:
            action = policy(x0)
            action = policy_post_process(action)
            x0, reward, terminated, truncated, _ = eval_env.step(action)
            fitness += reward
            done = terminated or truncated

        on_step(fitness, i)
        reward_sequence.append(np.asarray(fitness))

    avg_reward = np.mean(reward_sequence)
    print(f"Final {avg_reward=}")

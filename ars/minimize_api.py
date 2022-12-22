import gym
import numpy as np
import torch as th
from abc import ABC
from tqdm import tqdm
from torch import Tensor
from gym.core import ObsType, ActType
from .ARSOptimizer import ARSOptimizer
from typing import Callable, Any, Tuple


def ars_minimize(obj_func: Callable, n_vars, n_steps=1_000, on_step: Callable[[float], Any] = lambda goodness: None,
                 **ars_opti_kwargs):
    class MinimizationEnv(gym.Env, ABC):
        def __init__(self):
            self.observation_space = gym.spaces.Discrete(1)
            self.action_space = gym.spaces.Box(
                low=np.array([-np.inf for _ in range(n_vars)]),
                high=np.array([np.inf for _ in range(n_vars)])
            )

        def reset(self, seed=None, options=None) -> Tuple[ObsType, dict]:
            return np.array([0]), {}

        def step(self, min_args: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
            reward = -obj_func(th.from_numpy(min_args))
            return np.array([0]), float(reward.numpy()), True, False, {}

    class MinimizationPolicy(th.nn.Module):
        def __init__(self):
            super(MinimizationPolicy, self).__init__()
            self.min_args = th.nn.Parameter(th.Tensor([0 for _ in range(n_vars)]))

        def forward(self, nothing: Tensor = None):
            return self.min_args

    policy = MinimizationPolicy()
    optimizer = ARSOptimizer(
        env=MinimizationEnv(),
        policy=policy,
        sdv=1,
        **ars_opti_kwargs
    )

    with tqdm(total=n_steps, postfix={"loss": th.inf}) as tqdm_updater:
        for t in range(1, n_steps + 1):
            optimizer.step()

            tqdm_updater.update()
            goodness = optimizer.loss
            tqdm_updater.set_postfix({"loss": goodness})
            on_step(goodness)

    optimizer.load()
    return policy.min_args.detach()

# https://arxiv.org/abs/1803.07055
# page 6

from collections.abc import Iterator
from typing import Optional, List, Callable, Any

import numpy as np
import torch as th
from torch import Tensor
from .Normalizer import Normalizer
from torch.optim import Optimizer


class ARSOptimizer(Optimizer):
    def __init__(self, parameters: Tensor, get_env, action_sz: int, get_policy: Callable[[Tensor, Normalizer], Any],
                 step_sz=1E-2, sdv=1E-3, n_directions=50, n_choice=None, hrz=1, normalizer=None, alive_bonus=0):
        self.goodness = -np.inf
        assert (n_directions % 2) == 0
        n_choice = int(n_directions / 2) if n_choice is None else n_choice
        assert n_choice <= n_directions

        # Initialize parameters, mean and covariance matrices to zero
        parameters.data.mul_(0)
        super().__init__(
            [parameters],
            dict(step_sz=step_sz, sdv=sdv, n_directions=n_directions, n_choice=n_choice, hrz=hrz, action_sz=action_sz,
                 alive_bonus=alive_bonus)
        )
        self.gen_env = get_env
        self.get_policy = get_policy
        self.normalizer = normalizer if normalizer else Normalizer(action_sz)

    def step(self, closure=None) -> Optional[float]:
        param_args = np.random.choice(self.param_groups)
        alive_bonus: float = float(param_args["alive_bonus"])
        n_directions: int = int(param_args["n_directions"])
        step_sz: float = float(param_args["step_sz"])
        parameters: Tensor = param_args["params"][0]
        n_choice: int = int(param_args["n_choice"])
        Horizon: int = int(param_args["hrz"])
        sdv: float = float(param_args["sdv"])

        if not self.state["step"]:
            self.state["step"] = 1
        else:
            self.state["step"] += 1

        with th.no_grad():
            changes, rwd_minus, rwd_plus = self.ars_step(Horizon, n_choice, n_directions, parameters, sdv, step_sz,
                                                         alive_bonus)

        parameters.data.add_(changes)
        step_goodness = np.mean([*rwd_plus, *rwd_minus])
        if step_goodness > self.goodness:
            self.goodness = step_goodness

        return None

    def ars_step(self, Horizon: int, n_choice: int, n_directions: int, parameters: Tensor, sdv: float, step_sz: float,
                 alive_bonus: float):
        # Line 4 of Algorithm
        # i.i.d standard normal distribution
        deltas: List[Tensor] = [th.normal(mean=0.0, std=1.0, size=parameters.shape) for _ in range(n_directions)]

        # Line 5 of Algorithm
        params_plus: List[Tensor] = [parameters + (sdv * delta) for delta in deltas]
        params_minus: List[Tensor] = [parameters - (sdv * delta) for delta in deltas]

        rollouts_plus: List[List[tuple]] = [self.query_oracle(param, Horizon, alive_bonus) for param in params_plus]
        rollouts_minus: List[List[tuple]] = [self.query_oracle(param, Horizon, alive_bonus) for param in params_minus]
        # print(rollouts_plus[0][0][2])

        rwd_plus = [np.sum([tup[2] for tup in rollouts]) for rollouts in rollouts_plus]
        rwd_minus = [np.sum([tup[2] for tup in rollouts]) for rollouts in rollouts_minus]

        # Line 6 of Algorithm
        rwd_max = [np.max([rwd_p, rwd_m]) for (rwd_p, rwd_m) in zip(rwd_plus, rwd_minus)]
        rwd_max_sorted = np.argsort(rwd_max)[::-1]

        deviations_sorted = [deltas[idx] for idx in rwd_max_sorted]
        rwd_plus_sorted = [rwd_plus[idx] for idx in rwd_max_sorted]
        rwd_minus_sorted = [rwd_minus[idx] for idx in rwd_max_sorted]

        # Line 7 of Algorithm
        sdv_rwd = np.std([*rwd_plus_sorted[:n_choice], *rwd_minus_sorted[:n_choice]])
        changes = (step_sz / (sdv_rwd * n_choice + 1e-6)) * th.stack([
            (rwd_plus_sorted[j] - rwd_minus_sorted[j]) * deviations_sorted[j]
            for j in range(n_choice)
        ]).sum(dim=0)

        return changes, rwd_minus, rwd_plus

    def query_oracle(self, params: Tensor, horizon: int, alive_bonus) -> List[tuple]:
        get_policy = self.get_policy
        get_env = self.gen_env

        env = get_env(params)
        x_prev = env.reset()
        policy = get_policy(params, self.normalizer)

        triple, done, h = [], False, 0
        while (not done) and (h < int(horizon)):
            action = policy(x_prev)
            x_prev, rwd, done = env.step(action)
            # Plus 1 to account for pow(0,0)==1
            triple.append((x_prev, action, rwd + pow(alive_bonus, h + 1)))
            h += 1

        return triple

# https://arxiv.org/abs/1803.07055
# page 6

from typing import Optional, List, Callable, Any

import functorch
import numpy as np
import torch as th
from torch import Tensor
from functools import partial
from .Normalizer import Normalizer
from torch.optim import Optimizer


class ARSOptimizer(Optimizer):
    def __init__(self, parameters: Tensor, get_env, action_sz: int, get_policy: Callable[[Tensor, Normalizer], Any],
                 step_sz=1E-2, sdv=1E-3, n_directions=50, n_choice=None, hrz=1, normalizer=None, alive_bonus=0):
        self.goodness = -th.inf
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

        self.param_args = np.random.choice(self.param_groups)
        self.alive_bonus = alive_bonus
        self.n_directions = n_directions
        self.step_sz = step_sz
        self.parameters = parameters
        self.n_choice = n_choice
        self.Horizon = hrz
        self.sdv = sdv

    def step(self, closure=None) -> Optional[float]:

        # param_args = np.random.choice(self.param_groups)
        parameters: Tensor = self.param_args["params"][0]

        alive_bonus = self.alive_bonus
        n_directions = self.n_directions
        step_sz = self.step_sz
        n_choice = self.n_choice
        Horizon = self.Horizon
        sdv = self.sdv

        if not self.state["step"]:
            self.state["step"] = 1
        else:
            self.state["step"] += 1

        with th.no_grad():
            grad, mean_rwd, sdv_rwd = self.ars_step(Horizon, n_choice, n_directions, parameters, sdv, step_sz,
                                                    alive_bonus)

        parameters.data.add_(grad)
        if mean_rwd > self.goodness:
            self.goodness = mean_rwd

        return None

    def ars_step(self, horizon: int, n_choice: int, n_directions: int, parameters: Tensor, sdv: float, step_sz: float,
                 alive_bonus: float):
        # Line 4 of Algorithm: i.i.d standard normal distribution
        # deltas: List[Tensor] = [th.normal(mean=0.0, std=1.0, size=parameters.shape) for _ in range(n_directions)]
        # deltas = th.vstack([th.normal(mean=0.0, std=1.0, size=parameters.shape) for _ in range(n_directions)])
        deltas = th.normal(mean=0.0, std=1.0, size=(n_directions, *parameters.shape))

        # Line 5 of Algorithm
        # params_plus: List[Tensor] = [parameters + (sdv * delta) for delta in deltas]
        # params_minus: List[Tensor] = [parameters - (sdv * delta) for delta in deltas]
        delta_plus = parameters + (sdv * deltas)
        delta_minus = parameters - (sdv * deltas)

        # rollouts_plus: List[List[tuple]] = [self.query_oracle(param, Horizon, alive_bonus) for param in params_plus]
        # rollouts_minus: List[List[tuple]] = [self.query_oracle(param, Horizon, alive_bonus) for param in params_minus]
        v_query = partial(self.query_oracle, horizon=horizon, alive_bonus=alive_bonus)
        rwd_plus = functorch.vmap(v_query)(delta_plus)
        rwd_minus = functorch.vmap(v_query)(delta_minus)
        rwd_plus, rwd_minus = rwd_plus.squeeze(dim=1).sum(dim=1), rwd_minus.squeeze(dim=1).sum(dim=1)

        # rwd_plus = [np.sum([tup[2] for tup in rollouts]) for rollouts in rollouts_plus]
        # rwd_minus = [np.sum([tup[2] for tup in rollouts]) for rollouts in rollouts_minus]

        # Line 6 of Algorithm
        # rwd_max = [np.max([rwd_p, rwd_m]) for (rwd_p, rwd_m) in zip(rwd_plus, rwd_minus)]
        # rwd_max_sorted = np.argsort(rwd_max)[::-1]
        rwd_max = th.maximum(rwd_plus, rwd_minus)
        rwd_max_sorted = rwd_max.argsort(dim=0, descending=True)

        # deviations_sorted = [deltas[idx] for idx in rwd_max_sorted]
        # rwd_plus_sorted = [rwd_plus[idx] for idx in rwd_max_sorted]
        # rwd_minus_sorted = [rwd_minus[idx] for idx in rwd_max_sorted]

        deviations_sorted = deltas[rwd_max_sorted][:n_choice]
        rwd_plus_sorted = rwd_plus[rwd_max_sorted][:n_choice]
        rwd_minus_sorted = rwd_minus[rwd_max_sorted][:n_choice]

        # Line 7 of Algorithm
        # sdv_rwd = np.std([*rwd_plus_sorted[:n_choice], *rwd_minus_sorted[:n_choice]])
        # changes = (step_sz / (sdv_rwd * n_choice + 1e-6)) * th.stack([
        #     (rwd_plus_sorted[j] - rwd_minus_sorted[j]) * deviations_sorted[j]
        #     for j in range(n_choice)
        # ]).sum(dim=0)
        sdv_rwd = th.std(th.vstack([rwd_plus_sorted, rwd_minus_sorted])).clip(1e-6)
        mean_rwd = th.mean(th.vstack([rwd_plus_sorted, rwd_minus_sorted]))

        grad = (step_sz / (n_choice * sdv_rwd)) * (rwd_plus_sorted - rwd_minus_sorted) @ deviations_sorted

        return grad, mean_rwd, sdv_rwd

    # def query_oracle(self, params: Tensor, horizon: int, alive_bonus) -> List[tuple]:
    #     get_policy = self.get_policy
    #     get_env = self.gen_env
    #
    #     env = get_env(params)
    #     x_prev = env.reset()
    #     policy = get_policy(params, self.normalizer)
    #
    #     triple, done, h = [], False, 0
    #     while (not done) and (h < int(horizon)):
    #         action = policy(x_prev)
    #         x_prev, rwd, done = env.step(action)
    #         # Plus 1 to account for pow(0,0)==1
    #         triple.append((x_prev, action, rwd + pow(alive_bonus, h + 1)))
    #         h += 1
    #
    #     return triple

    def query_oracle(self, deltas: Tensor, horizon: int, alive_bonus) -> Tensor:
        get_policy = self.get_policy
        get_env = self.gen_env

        env = get_env(deltas)
        x_prev = env.reset()
        policy = get_policy(deltas, self.normalizer)

        # A recursive function
        def inner_rec(x=x_prev, rewards=th.zeros(horizon), h=0):
            h += 1
            action = policy(x)
            x_next, rwd, done = env.step(action)
            rwd = rwd.reshape(1, 1)
            rewards = rewards + th.Tensor([1 if i == h - 1 else 0 for i in range(horizon)]) * rwd

            done = done or (h < int(horizon))
            if done:
                return rewards
            else:
                inner_rec(x_next, rewards, h)

        return inner_rec(x_prev)

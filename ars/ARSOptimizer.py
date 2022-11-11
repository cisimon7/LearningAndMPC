# https://arxiv.org/abs/1803.07055
# page 6
import copy
import math
from typing import Optional

import gym
import numpy as np
import torch as th
import functorch as fth
from torch import Tensor
from torch.optim import Optimizer

from .Normalizer import Normalizer


# Is it good practise to use @dataclass for the purpose of avoiding having to do all these self. things?
class ARSOptimizer(Optimizer):
    def __init__(self, parameters: Tensor, env: gym.Env, action_sz: int, policy: th.nn.Module, step_sz=1E-2, sdv=1E-3,
                 n_directions=50, n_choice=None, hrz=1, normalizer=None, alive_bonus=0):
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
        self.gen_env = env
        self.get_policy = policy
        self.normalizer = normalizer if normalizer else Normalizer(action_sz)

        self.param_args = np.random.choice(self.param_groups)
        self.alive_bonus = alive_bonus
        self.n_directions = n_directions
        self.step_sz = step_sz
        self.parameters = parameters
        self.n_choice = n_choice
        self.Horizon = hrz
        self.sdv = sdv

        self.vec_envs = gym.vector.SyncVectorEnv([lambda: env for _ in range(n_directions)])
        # Wrap environment to return tensor observations and receive tensor actions

        self.vec_models, self.vec_params, self.vec_buffers = fth.combine_state_for_ensemble(
            [copy.deepcopy(policy) for _ in range(n_directions)]
        )

        self.shapes = []
        for param in self.vec_params:
            self.shapes.append(param.shape[1:])

    def reshape_tensor(self, tensor):
        parameters, start = (), 0
        for layer in self.shapes:
            length = start + math.prod(layer)
            parameters += (tensor.index_select(0, th.arange(start=start, end=length)).view(layer),)
            start += length

        return parameters

    def vec_query_oracle(self, parameters, horizon: int) -> Tensor:
        observations, _ = self.vec_envs.reset()
        rewards, done, step = None, False, 0

        while step < horizon:
            step += 1
            if done:
                rewards = np.vstack([rewards, np.zeros(self.n_directions)])
            else:
                actions: Tensor = fth.vmap(self.vec_models)(
                    parameters, self.vec_buffers, th.from_numpy(observations)
                )
                observations, rwds, terminated, truncated, infos = self.vec_envs.step(actions.numpy())

                rewards = np.vstack([rwds]) if rewards is None else np.vstack([rewards, rwds])
                done = np.all(terminated) or np.all(truncated)

        return th.from_numpy(rewards)

    def step(self, closure=None) -> Optional[float]:

        parameters: Tensor = self.param_args["params"][0]

        with th.no_grad():
            # Line 4 of Algorithm: i.i.d standard normal distribution
            deltas = th.normal(mean=0.0, std=1.0, size=(self.n_directions, *parameters.shape))

            # Line 5 of Algorithm TODO(Look into using vmap)
            deltas_plus = parameters + (self.sdv * deltas)
            deltas_minus = parameters - (self.sdv * deltas)

            vec_deltas_plus = fth.vmap(self.reshape_tensor)(deltas_plus)
            vec_deltas_minus = fth.vmap(self.reshape_tensor)(deltas_minus)

            vec_rwd_roll_p = self.vec_query_oracle(vec_deltas_plus, horizon=self.Horizon)
            vec_rwd_roll_m = self.vec_query_oracle(vec_deltas_minus, horizon=self.Horizon)

            rwd_p, rwd_m = vec_rwd_roll_p.sum(dim=0, dtype=th.float32), vec_rwd_roll_m.sum(dim=0, dtype=th.float32)

            # Line 6 of Algorithm
            rwd_max = th.maximum(rwd_p, rwd_m)
            rwd_max_sorted = rwd_max.argsort(dim=0, descending=True)

            deviations_sorted = deltas[rwd_max_sorted][:self.n_choice]
            rwd_plus_sorted = rwd_p[rwd_max_sorted][:self.n_choice]
            rwd_minus_sorted = rwd_m[rwd_max_sorted][:self.n_choice]

            # Line 7 of Algorithm
            sdv_rwd = th.std(th.vstack([rwd_plus_sorted, rwd_minus_sorted])).clip(1e-6)
            mean_rwd = th.mean(th.vstack([rwd_plus_sorted, rwd_minus_sorted]))

            grad = (self.step_sz / (self.n_choice * sdv_rwd)) * (rwd_plus_sorted - rwd_minus_sorted) @ deviations_sorted

            parameters.data.add_(grad)
            if mean_rwd > self.goodness:
                self.goodness = mean_rwd

        if not self.state["step"]:
            self.state["step"] = 1
        else:
            self.state["step"] += 1

        return None

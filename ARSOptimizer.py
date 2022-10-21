# https://math.stackexchange.com/questions/1147084/dynamic-update-of-co-variance-matrix-upon-new-sample
# https://arxiv.org/abs/1803.07055
# page 6

from collections.abc import Iterator
from typing import Optional, List

import numpy as np
import torch as th
from torch import Tensor
import multiprocessing as mp
from torch.optim import Optimizer


class ARSOptimizer(Optimizer):
    def __init__(self, parameters: Iterator[Tensor], get_env, get_policy, step_sz=1E-2, sdv=1E-3,
                 n_directions=50, n_choice=None, hrz=1, normalizer=None):
        self.goodness = -np.inf
        assert (n_directions % 2) == 0
        n_choice = int(n_directions / 2) if n_choice is None else n_choice
        assert n_choice <= n_directions

        # Initialize parameters, mean and covariance matrices to zero
        parameters = [
            tensor.data.mul_(0)
            for tensor in parameters
        ]
        super().__init__(
            parameters,
            dict(step_sz=step_sz, sdv=sdv, n_directions=n_directions, n_choice=n_choice, hrz=hrz)
        )
        self.gen_env = get_env
        self.get_policy = get_policy
        self.n_input_shape = parameters[0].shape[0]
        self.normalizer = normalizer if normalizer else Normalizer(self.n_input_shape)

    def step(self, closure=None) -> Optional[float]:
        param_args = np.random.choice(self.param_groups)
        parameters: List[Tensor] = param_args["params"]
        n_directions = param_args["n_directions"]
        n_choice = param_args["n_choice"]
        step_sz = param_args["step_sz"]
        Horizon = param_args["hrz"]
        sdv = param_args["sdv"]

        if not self.state["step"]:
            self.state["step"] = 1
        else:
            self.state["step"] += 1

        n_choice, n_directions = int(n_choice), int(n_directions)
        changes, rwd_minus, rwd_plus = self.ars_step(Horizon, n_choice, n_directions, parameters, sdv, step_sz)

        [
            tensor.data.add_(change)
            for (tensor, change) in zip(parameters, changes)
        ]  # new_params

        step_goodness = np.mean([*rwd_plus, *rwd_minus])
        if step_goodness > self.goodness:
            self.goodness = step_goodness

        return None

    def ars_step(self, Horizon, n_choice, n_directions, parameters, sdv, step_sz):
        # Line 4 of Algorithm
        deviations = [
            # i.i.d standard normal distribution
            [th.randn(param.shape) for param in parameters]
            for _ in range(n_directions)
        ]  # TODO(Deviations should be uniform round the point zero)

        # Line 5 of Algorithm
        params_plus = [
            [param + (sdv * delta) for (param, delta) in zip(parameters, deltas)]
            for deltas in deviations
        ]
        params_minus = [
            [param - (sdv * delta) for (param, delta) in zip(parameters, deltas)]
            for deltas in deviations
        ]

        rollouts_plus = [self.query_oracle(params, Horizon) for params in params_plus]
        rollouts_minus = [self.query_oracle(params, Horizon) for params in params_minus]

        rwd_plus = [
            np.sum([tup[2] for tup in rollouts])
            for rollouts in rollouts_plus
        ]
        rwd_minus = [
            np.sum([tup[2] for tup in rollouts])
            for rollouts in rollouts_minus
        ]

        # Line 6 of Algorithm
        rwd_max = [np.max([rwd_p, rwd_m]) for (rwd_p, rwd_m) in zip(rwd_plus, rwd_minus)]
        rwd_max_sorted = np.argsort(rwd_max)[::-1]

        deviations_sorted = [deviations[idx] for idx in rwd_max_sorted]
        rwd_plus_sorted = [rwd_plus[idx] for idx in rwd_max_sorted]
        rwd_minus_sorted = [rwd_minus[idx] for idx in rwd_max_sorted]

        # Line 7 of Algorithm
        # TODO(what to do if this is zero)
        sdv_rwd = np.std([*rwd_plus_sorted[:n_choice], *rwd_minus_sorted[:n_choice]]).clip(1E-5)
        param_length = len(parameters)
        group_deltas = [
            [deviations_sorted[k][j] for k in range(n_choice)]
            for j in range(param_length)
        ]
        changes = [
            (step_sz / (sdv_rwd * n_choice)) * th.stack([
                (rwd_plus_sorted[j] - rwd_minus_sorted[j]) * group_deltas[k][j]
                for j in range(n_choice)
            ]).sum(dim=0)
            for k in range(param_length)
        ]

        return changes, rwd_minus, rwd_plus

    def query_oracle(self, params, horizon):
        get_policy = self.get_policy
        get_env = self.gen_env

        env = get_env(params)
        x_prev = env.reset()

        policy = get_policy(params, self.normalizer)

        triple, done, h = [], False, 0
        while (not done) and (h < int(horizon)):
            action = policy(x_prev)
            x_prev, rwd, done = env.step(action)
            # rwd += (h * rwd)
            triple.append((x_prev, action, rwd))
            h += 1

        return triple


class Normalizer:
    def __init__(self, n_inputs):
        self.n = 0
        self.mean = np.zeros(n_inputs)
        self.cov = np.zeros((n_inputs, n_inputs))

        self.A = np.zeros((n_inputs, n_inputs))
        self.b = np.zeros(n_inputs)

    def observe(self, x):
        self.n += 1
        self.A += np.array(x) @ np.array(x).T
        self.b += np.array(x)

        self.mean = (1 / self.n) * self.b
        self.cov = (1 / self.n) * (
                self.A - (self.mean @ self.b.T) - (self.b @ self.mean.T) + (self.n * self.mean @ self.mean.T)
        )

    def normalize(self, inputs):
        return self.cov @ (np.array(inputs) - self.mean)

    def obs_norm(self, inputs):
        self.observe(inputs)
        return self.normalize(inputs)

    def save_state(self, path: str):
        np.savez(path, mean=self.mean, cov=self.cov, A=self.A, b=self.b)

    def load_state(self, path: str):
        states = np.load(path)
        self.mean = states["mean"]
        self.cov = states["cov"]
        self.A = states["A"]
        self.b = states["b"]

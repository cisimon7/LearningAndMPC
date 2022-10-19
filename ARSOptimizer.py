# https://math.stackexchange.com/questions/1147084/dynamic-update-of-co-variance-matrix-upon-new-sample
# https://arxiv.org/abs/1803.07055
# page 6

from collections.abc import Iterator

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable, Optional, Tuple, List


class ARSOptimizer(Optimizer):
    def __init__(self, parameters: Iterator[Tensor], fwd_fun, step_fun, reset_input, step_sz=1E-3, sdv=1,
                 n_directions=20, n_choice=None, hrz=1, normalizer=None):
        self.goodness = None
        assert (n_directions % 2) == 0
        n_choice = int(n_directions / 2) if n_choice is None else n_choice
        assert n_choice <= n_directions

        # Initialize parameters, mean and covariance matrices to zero
        parameters = [
            tensor.data.mul_(0)
            for tensor in parameters
        ]
        # mean
        # covariance
        super().__init__(
            parameters,
            dict(fwd_fun=fwd_fun, step_fun=step_fun, reset_input=reset_input, step_sz=step_sz, sdv=sdv,
                 n_directions=n_directions, n_choice=n_choice, hrz=hrz)
        )
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

        n_choice, n_directions = int(n_choice), int(n_directions)

        # Line 4 of Algorithm
        deviations = [
            # i.i.d standard normal distribution
            [torch.randn(param.shape) for param in parameters]
            for (_, gen) in zip(range(n_directions), [torch.Generator() for _ in range(n_directions)])
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
            np.mean(np.asarray([tup[2] for tup in rollouts]))
            for rollouts in rollouts_plus
        ]
        rwd_minus = [
            np.mean(np.asarray([tup[2] for tup in rollouts]))
            for rollouts in rollouts_minus
        ]

        # Line 6 of Algorithm
        rwd_max = [np.max([rwd_p, rwd_m]) for (rwd_p, rwd_m) in zip(rwd_plus, rwd_minus)]
        rwd_max_sorted = np.argsort(rwd_max)[::-1]

        deviations_sorted = [deviations[idx] for idx in rwd_max_sorted]
        rwd_plus_sorted = [rwd_plus[idx] for idx in rwd_max_sorted]
        rwd_minus_sorted = [rwd_minus[idx] for idx in rwd_max_sorted]

        # Line 7 of Algorithm
        sdv_rwd = np.std([*rwd_plus_sorted[:n_choice], *rwd_minus_sorted[:n_choice]])
        param_length = len(parameters)

        group_deltas = [
            [deviations_sorted[k][j] for k in range(n_choice)]
            for j in range(param_length)
        ]

        changes = [
            (step_sz / sdv_rwd * n_choice) * torch.stack([
                (rwd_plus_sorted[j] - rwd_minus_sorted[j]) * group_deltas[k][j]
                for j in range(n_choice)
            ]).sum(dim=0)
            for k in range(param_length)
        ]

        [
            tensor.data.add_(change)
            for (tensor, change) in zip(parameters, changes)
        ]  # new_params

        self.goodness = np.mean(rwd_max)

        return None

    def query_oracle(self, params, horizon):
        param_args = np.random.choice(self.param_groups)
        reset_input = param_args["reset_input"]
        step_fun = param_args["step_fun"]
        fwd_fun = param_args["fwd_fun"]

        x_prev = reset_input(params)  # TODO(how this is chosen)
        triple = []
        done = False

        for h in range(int(horizon)):
            if done:
                triple.append((x_prev, None, 0))

            action = fwd_fun(params, self.normalizer.obs_norm(x_prev))  # Policy predicting action
            x_prev, rwd, done = step_fun(x_prev, action)  # Environment step function
            triple.append((x_prev, action, rwd))

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

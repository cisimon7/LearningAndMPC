import collections
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from torch.optim import Optimizer, Adam, SGD
from typing import Callable, Optional, Tuple, List, Iterable, Union

ForwardFun = Callable[[List[Tensor], Tensor | np.ndarray], Tensor | np.ndarray]


class ARSOptimizer(Optimizer):
    def __init__(self, params: List[Tensor], fwd_fun, step_fun, step_sz=0.02, sdv=1, n_directions=50, n_choice=None,
                 hrz=1):
        assert (n_directions % 2) == 0
        n_choice = int(n_directions / 2) if n_choice is None else n_choice
        super().__init__(
            params,
            dict(fwd_fun=fwd_fun, step_fun=step_fun, step_sz=step_sz, sdv=sdv, n_directions=n_directions,
                 n_choice=n_choice, hrz=hrz)
        )
        self.n_input_shape = params[0].shape
        self.str = []  # sequence of state, action, reward

    def step(self, closure=None) -> Optional[float]:
        param_args = np.random.choice(self.param_groups)
        parameters: List[Tensor] = param_args["params"]
        n_directions = param_args["n_directions"]
        n_choice = param_args["n_choice"]
        step_sz = param_args["step_sz"]
        Horizon = param_args["hrz"]

        # print(param_args["params"])

        n_choice, n_directions = int(n_choice), int(n_directions)

        deviations = [
            [torch.randn(param.shape) for param in parameters]
            for _ in range(n_directions)
        ]  # TODO(Deviations should be uniform round the point zero)

        params_plus = [
            [param + delta for (param, delta) in zip(parameters, deltas)]
            for deltas in deviations
        ]
        params_minus = [
            [param + delta for (param, delta) in zip(parameters, deltas)]
            for deltas in deviations
        ]

        rollouts_plus = [self.query_oracle(params, Horizon) for params in params_plus]
        rollouts_minus = [self.query_oracle(params, Horizon) for params in params_minus]

        rwd_plus = [
            np.sum(np.array(np.asarray(rollouts)[:, 2]))
            for rollouts in rollouts_plus
        ]
        rwd_minus = [
            np.sum(np.array(np.asarray(rollouts)[:, 2]))
            for rollouts in rollouts_minus
        ]

        rwd_max = [np.max([rwd_p, rwd_m]) for (rwd_p, rwd_m) in zip(rwd_plus, rwd_minus)]
        rwd_max_sorted = np.argsort(rwd_max)[::-1]

        deviations_sorted = [deviations[idx] for idx in rwd_max_sorted]
        rwd_plus_sorted = [rwd_plus[idx] for idx in rwd_max_sorted]
        rwd_minus_sorted = [rwd_minus[idx] for idx in rwd_max_sorted]

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
            param.data.add_(change)
            for (param, change) in zip(parameters, changes)
        ]  # new_params

        return None

    def query_oracle(self, params, horizon):
        param_args = np.random.choice(self.param_groups)
        step_fun = param_args["step_fun"]
        fwd_fun = param_args["fwd_fun"]
        x_prev = torch.randn(self.n_input_shape)

        triple: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for h in range(int(horizon)):
            action = fwd_fun(params, x_prev)
            x_next, rwd = step_fun(x_prev, action)
            triple.append((x_next, action, rwd))

        return triple


def rosenbrock(xy):
    x, y = xy
    return torch.pow(1 - x, 2) + torch.pow(y - torch.pow(x, 2), 2)


if __name__ == '__main__':
    xy_init = (0.3, 0.8)
    xy_t = torch.tensor(xy_init, requires_grad=True)
    LinearModel = torch.nn.Linear(3, 3)

    # optimizer = Adam([xy_t])
    # optimizer = SGD([xy_t], lr=0.01)
    optimizer = ARSOptimizer(
        [xy_t],
        fwd_fun=lambda params, x: None,
        step_fun=lambda x, action: (None, -rosenbrock(x))
    )

    n_steps = 1_000
    path = np.empty((n_steps + 1, 2))
    path[0, :] = xy_init

    with tqdm(total=n_steps, postfix={"loss": torch.inf}) as tqdm_control:
        for t in range(1, n_steps + 1):

            loss = rosenbrock(xy_t)
            # optimizer.zero_grad() # Not needed for ARS type
            # loss.backward()  # Not needed for ARS type
            # torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
            optimizer.step()

            path[t, :] = xy_t.detach().numpy()

            tqdm_control.update()
            if t % 10 == 0:
                tqdm_control.set_postfix({"loss": loss})

    xy_t.detach_()
    print(f"Minimum at: {xy_t}")

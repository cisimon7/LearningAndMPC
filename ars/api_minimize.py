from typing import Callable, Any

import torch as th
from torch import Tensor
from tqdm import tqdm

from .ARSOptimizer import ARSOptimizer


def ars_minimize(
        obj_func: Callable,
        n_vars=1,
        n_steps=1_000,
        on_step: Callable[[Tensor, float, int], Any] = lambda goodness: None,
        **ars_opti_kwargs
):
    class MinimizationEnv:
        def __init__(self, params):
            self.params = params

        def reset(self):
            return self.params

        def step(self, action):
            x = self.params
            return None, - obj_func(x).detach().numpy(), True

    xy_t = th.Tensor([0 for _ in range(n_vars)])
    optimizer = ARSOptimizer(
        xy_t,
        get_env=lambda params: MinimizationEnv(params),
        action_sz=1,
        get_policy=(lambda params, normalizer: (lambda x: None)),
        sdv=1E-3,
        **ars_opti_kwargs
    )

    with tqdm(total=n_steps, postfix={"loss": th.inf}) as tqdm_updater:
        for t in range(1, n_steps + 1):
            optimizer.step()

            tqdm_updater.update()
            goodness = optimizer.goodness
            tqdm_updater.set_postfix({"goodness": goodness})

            on_step(xy_t, goodness, t)

    xy_t.detach_()
    print(f"Minimum at: {xy_t}")

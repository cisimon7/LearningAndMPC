from typing import Callable, Any

import torch as th
from torch import Tensor
from tqdm import tqdm

from .ARSOptimizer import ARSOptimizer


class MinimizationEnv:
    def __init__(self, params, obj_func):
        """
        The minimization problem is treated like an RL problem, but a much simpler problem than typical RL problems
        :param params: the arguments to be minimized
        :param obj_func: the objective function of the minimization function
        """
        self.params = params
        self.obj_func = obj_func

    def reset(self):
        return self.params

    def step(self, action):
        """
        The reward function is simply the negative of the objective function at the current values of the minimization
        arguments. Unlike RL problems, we do not need to define a policy for this problem.
        :param action: Action isn't needed for the minimization problem
        :return: A tuple of (next_state, reward, done)
        """
        x = self.params
        return None, - self.obj_func(x).detach().numpy(), True


def ars_minimize(
        obj_func: Callable,
        n_vars=1,
        n_steps=1_000,
        on_step: Callable[[Tensor, float, int], Any] = lambda goodness: None,
        **ars_opti_kwargs
):
    """
    Unconstrained Optimization problem formulation using the ARS algorithm.
    :param obj_func: Objective function to be minimized
    :param n_vars:  number of minimization args
    :param n_steps: Number of optimization steps to be taken
    :param on_step: Function to perform on each optimization step given the current minimization values, reward function and step count
    :return: None
    """
    xy_t = th.Tensor([0 for _ in range(n_vars)])
    optimizer = ARSOptimizer(
        xy_t,
        get_env=lambda params: MinimizationEnv(params, obj_func),
        action_sz=1,
        get_policy=(lambda params, normalizer: (lambda x: None)),  # No need for a policy
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

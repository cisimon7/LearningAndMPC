import torch as th
from torch import Tensor
# from collections.abc import Callable
from typing import Callable, List


# Ode function that takes states of system and current time
OdeFunc = Callable[[Tensor, float], Tensor]


# Methods for numerically integrating ordinary differential equations
class RungeKutta2ndOrder:
    def __init__(self, step_sz: float, ode_func: OdeFunc):
        pass


class RungeKutta4thOrder:
    def __init__(self, step_sz: float, ode_func: OdeFunc):
        self.step_sz = step_sz
        self.ode_func = ode_func

    def k1(self, x0: Tensor, t0: float) -> Tensor:
        return th.mul(self.ode_func(x0, t0), self.step_sz)

    def k2(self, x0: Tensor, t0: float):
        k_1 = self.k1(x0, t0)
        return th.mul(self.ode_func(x0 + th.mul(k_1, 0.5), t0 + th.mul(self.step_sz, 0.5)), self.step_sz)

    def k3(self, x0: Tensor, t0: float):
        k_2 = self.k2(x0, t0)
        return th.mul(self.ode_func(x0 + th.mul(k_2, 0.5), t0 + th.mul(self.step_sz, 0.5)), self.step_sz)

    def k4(self, x0: Tensor, t0: float):
        k_3 = self.k3(x0, t0)
        return th.mul(self.ode_func(x0 + k_3, t0 + self.step_sz), self.step_sz)

    def step(self, x0: Tensor, t0: float):
        return (self.ode_func(x0, t0) +
                (1/6)*self.k1(x0, t0) +
                (1/3)*self.k2(x0, t0) +
                (1/3)*self.k3(x0, t0) +
                (1/6)*self.k4(x0, t0))

    def steps(self, count: int, x0: Tensor, t0: float):
        pass

    def interval(self, x0: Tensor, t0: float, tf: float):
        pass

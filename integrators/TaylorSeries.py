import math

import torch as th
from torch import Tensor
from torch.autograd import functional as fth
from typing import Callable, Any, List
from torch.autograd import Function


class TaylorSeriesApproximation:
    def __init__(self, func: Callable[[Tensor], Tensor], about: Tensor, order: int = 2):
        # Generates a function that is nth order approximation of the given function about the point of [focus]
        diff_func, self.focus, self.order = func, about, order
        self.diffs = [diff_func] + [
            diff_func := lambda tensor: fth.jacobian(diff_func, tensor, create_graph=False, vectorize=True)
            for _ in range(1, self.order + 1)
        ]

    def forward(self, x: Tensor) -> Tensor:
        total = th.zeros_like(x)
        polys = [th.pow(self.focus - x, n) / math.factorial(n) for n in range(0, self.order + 1)]
        for (diff, poly) in zip(self.diffs, polys):
            total += diff(x) * poly

        return total

        # diff_func = lambda tensor: fth.jacobian(self.func, tensor, create_graph=False, vectorize=True)
        # for n in range(1, self.order):
        #     total += fth.jacobian(diff_func, x, create_graph=False, vectorize=True) * th.pow(self.focus - x, n)
        #     diff_func = lambda tensor: fth.jacobian(diff_func, tensor, create_graph=False, vectorize=True)
        #
        # return total

    def __call__(self, x: Tensor):
        return self.forward(x)


if __name__ == '__main__':
    total = 0
    print([
        total := total + 1
        for _ in range(10)
    ])

    function = th.sin
    func_approx = TaylorSeriesApproximation(function, about=th.tensor(0), order=3)
    point = th.tensor(3 * th.pi)

    actual, estimate = (function(point), func_approx(point))
    print(actual, estimate)

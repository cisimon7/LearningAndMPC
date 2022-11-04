from functools import partial
from time import sleep

import torch as th
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ars import ars_minimize

# Unconstrained optimization problems

writer = SummaryWriter("../../logs/ars/minimization")


def tensor_board(title: str, params: Tensor, goodness: float, count: int):
    values = params.tolist()
    names = [f"x{i}" for i in range(len(values))]

    writer.add_scalar(title + "/objective", goodness, count)
    writer.add_scalars(title + "/parameters", dict(zip(names, values)), count)


def rosenbrock(xy):
    x, y = xy
    return th.pow(1 - x, 2) + 100 * th.pow(y - th.pow(x, 2), 2)


def quadratic_program(x: th.Tensor):
    A = th.eye(3)
    b = th.ones(3)
    x = x.reshape(1, -1)
    return (x @ A @ th.transpose(x, dim0=0, dim1=1)) + (b @ th.transpose(x, dim0=0, dim1=1))


if __name__ == '__main__':
    ars_minimize(
        obj_func=rosenbrock,
        n_vars=2,
        n_steps=1_000,
        on_step=partial(tensor_board, "Rosenbrock")
    )

    sleep(0.01)

    ars_minimize(
        obj_func=quadratic_program,
        n_vars=3,
        n_steps=1_000,
        on_step=partial(tensor_board, "Quadratic")
    )

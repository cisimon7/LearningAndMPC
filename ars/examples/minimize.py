import torch as th

from ars.api import ars_minimize


# Unconstrained optimization problems


def rosenbrock(xy):
    x, y = xy
    return th.pow(1 - x, 2) + 100 * th.pow(y - th.pow(x, 2), 2)


def quadratic_program(x: th.Tensor):
    A = th.eye(3)
    b = th.ones(3)
    x = x.reshape(1, -1)
    return (x @ A @ th.transpose(x, dim0=0, dim1=1)) + (b @ th.transpose(x, dim0=0, dim1=1))


if __name__ == '__main__':
    # ars_minimize(
    #     obj_func=rosenbrock,
    #     n_vars=2
    # )

    ars_minimize(
        obj_func=quadratic_program,
        n_vars=3,
        n_steps=1_000
    )

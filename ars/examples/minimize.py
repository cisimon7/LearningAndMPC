import torch as th

from ars.api import ars_minimize


def rosenbrock(xy):
    x, y = xy
    return th.pow(1 - x, 2) + 100 * th.pow(y - th.pow(x, 2), 2)


def quadratic(x):
    A = th.eye(3)
    b = th.ones(3)
    x = th.unsqueeze(x, dim=0)
    return (x @ A @ th.transpose(x, dim0=0, dim1=1)) + (b @ th.transpose(x, dim0=0, dim1=1))


if __name__ == '__main__':
    ars_minimize(
        obj_func=rosenbrock,
        n_vars=2
    )

    # ars_minimize(
    #     obj_func=lambda x: 100*x - 2*th.pow(x, 2),
    #     n_vars=1
    # )

    ars_minimize(
        obj_func=quadratic,
        n_vars=3
    )

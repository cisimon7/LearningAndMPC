import torch as th
from time import sleep
from ars import ars_minimize


def rosenbrock(xy):
    # opti_val = 0, opti_var = [1, 1]
    x, y = xy
    return th.pow(1 - x, 2) + 100 * th.pow(y - th.pow(x, 2), 2)


def quadratic_program(x: th.Tensor):
    # opti_val = 0.75, opt_var = [-0.5, -0.5, -0.5]
    A = th.eye(3)
    b = th.ones(3)
    x = x.reshape(1, -1)
    return (x @ A @ th.transpose(x, dim0=0, dim1=1)) + (b @ th.transpose(x, dim0=0, dim1=1))


if __name__ == '__main__':
    th.manual_seed(73)  # 42, 73

    opt = ars_minimize(
        obj_func=rosenbrock,
        n_vars=2,
        n_steps=1000,
    )
    print(f"Minimum at: {opt.numpy()}")

    sleep(0.01)

    opt = ars_minimize(
        obj_func=quadratic_program,
        n_vars=3,
        n_steps=1_000,
        # on_step=partial(tensor_board, "Quadratic")
    )
    print(f"Minimum at: {opt.numpy()}")

import math
import torch as th
from torch import Tensor
from plotters import go, line_plot
from integrators import RungeKutta4thOrder


def test_rk4_lorenz(show=False):
    def lorenz(xyz: Tensor, time: float) -> Tensor:
        sigma, rho, beta = 10, 28, 8/3
        x, y, z = xyz
        return th.tensor([
            sigma*(y - x),
            x*(rho - z) - y,
            x*y - z*beta
        ])

    ode = RungeKutta4thOrder(0.01, lorenz)
    x0 = th.tensor([-8, 8, 20])
    trajectory = ode.interval(x0, 0, 20)

    go.Figure([
        line_plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    ]).show() if show else None


def test_rk4_pendulum(show=False):
    g, L, b = 9.81, 1, 0.1

    def pendulum(state: Tensor, time: float) -> Tensor:
        theta, dtheta = state
        return th.tensor([dtheta, -(g/L) * th.sin(theta) - b*dtheta])

    ode = RungeKutta4thOrder(0.01, pendulum)

    # System starting from stable equalibrium point remains at that point
    trajectory1 = ode.interval(th.tensor([0, 0]), 0, 20)
    th.testing.assert_close(trajectory1[-1], th.tensor([0.0, 0.0]))

    # When damping factor is greater than 0, the system should eventually come to rest after a long time at
    # stable equilibrium point
    duration = 1_000
    x0 = th.tensor([math.pi, 1])
    trajectory2 = ode.interval(x0, 0, duration)
    assert (b >= 0) and (duration >= 1_000) and th.testing.assert_close(trajectory2[-1], th.tensor([0.0, 0.0]))

    go.Figure([
        line_plot(trajectory1[:, 0], trajectory1[:, 1]),
        line_plot(trajectory2[:, 0], trajectory2[:, 1])
    ]).show() if show else None

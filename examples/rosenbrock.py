import torch as th
import numpy as np
from tqdm import tqdm

from ARSOptimizer import ARSOptimizer

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")


def rosenbrock(xy):
    x, y = xy
    return th.pow(1 - x, 2) + 100 * th.pow(y - th.pow(x, 2), 2)


class RosenbrockEnv:
    def __init__(self, params):
        self.params = params

    def reset(self):
        return self.params

    def step(self, action):
        x = self.params
        return None, -rosenbrock(x), True


if __name__ == '__main__':
    n_steps = 2_000

    xy_init = (0.3, -0.8)
    xy_t = th.tensor(xy_init)
    optimizer = ARSOptimizer(
        xy_t,
        get_env=lambda params: RosenbrockEnv(params),
        action_sz=1,
        get_policy=(lambda params, normalizer: (lambda x: None)),
        sdv=1E-3
    )

    with tqdm(total=n_steps, postfix={"loss": th.inf}) as tqdm_updater:
        for t in range(1, n_steps + 1):
            optimizer.step()

            tqdm_updater.update()
            if t % 10 == 0:
                goodness = optimizer.goodness
                tqdm_updater.set_postfix({"goodness": goodness})

            writer.add_scalar(f"rosenbrock/ars", optimizer.goodness, t)

    xy_t.detach_()
    print(f"Minimum at: {xy_t}")

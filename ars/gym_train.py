import gym
import torch as th
from tqdm import tqdm
from typing import Callable, Any
from .ARSOptimizer import ARSOptimizer


def ars_policy_train(
        train_env: gym.Env,
        train_policy: th.nn.Module,
        duration: int = 500,
        n_directions=100,
        train_steps=1_000,
        policy_params_path: str = None,
        save_on_improve=False,
        on_step: Callable[[float, int], Any] = lambda goodness, step: None,
):

    ars_optimizer = ARSOptimizer(env=train_env, policy=train_policy, n_directions=n_directions,
                                 sdv=0.05, lr=0.02, hrz=duration)

    rwd, prev_rwd = - th.inf, - th.inf
    with tqdm(total=train_steps, postfix={"goodness": rwd}) as tqdm_:
        for t in range(1, train_steps + 1):
            ars_optimizer.step()
            tqdm_.update()
            rwd = ars_optimizer.loss
            tqdm_.set_postfix({"goodness": rwd})

            on_step(rwd, t)

            if save_on_improve and (rwd > prev_rwd):
                ars_optimizer.load()
                th.save(train_policy.state_dict(), policy_params_path + f"_{int(rwd)}")
                prev_rwd = rwd

    ars_optimizer.load()
    th.save(train_policy.state_dict(), policy_params_path + f"_{int(rwd)}")

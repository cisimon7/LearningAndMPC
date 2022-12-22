import gym
import numpy as np
import torch as th
from tqdm import tqdm
from copy import deepcopy
from gym.core import ObsType
from .Normalizer import Normalizer
from typing import Callable, Any, Tuple
from .ARSOptimizer import ARSOptimizer


def ars_policy_train(
        train_env: gym.Env,
        train_policy: th.nn.Module,
        train_normalizer: Normalizer = None,
        train_steps=1_000,
        policy_params_path: str = None,
        normalizer_params_path: str = None,
        on_step: Callable[[float, int], Any] = lambda goodness, step: None,
        save_on_improve=False,
        n_directions=100
):

    obs_dim = train_env.observation_space.shape[0]
    action_dim = 1 if isinstance(train_env.action_space, gym.spaces.Discrete) else train_env.action_space.shape[0]

    if train_normalizer is None:
        train_normalizer = Normalizer(obs_dim)

    # TODO(Probably why model parameter not updating in place)
    param_vector = th.nn.utils.parameters_to_vector(train_policy.parameters()).detach().cpu()

    ars_optimizer = ARSOptimizer(env=train_env, policy=train_policy, n_directions=n_directions,
                                 sdv=0.05, lr=0.02, normalizer=train_normalizer, hrz=500)

    goodness, prev_goodness = - th.inf, - th.inf
    with tqdm(total=train_steps, postfix={"goodness": goodness}) as tqdm_:
        for t in range(1, train_steps + 1):
            ars_optimizer.step()
            tqdm_.update()
            goodness = ars_optimizer.loss
            tqdm_.set_postfix({"goodness": goodness})

            on_step(goodness, t)

            if save_on_improve and (goodness > prev_goodness):
                th.nn.utils.vector_to_parameters(
                    ars_optimizer.param_groups[0]["params"][0],
                    train_policy.parameters()
                )
                train_normalizer.save_state(policy_params_path + f"_{np.round(goodness, 4)}")
                th.save(train_policy.state_dict(), normalizer_params_path + f"_{np.round(goodness, 4)}")
                prev_goodness = goodness

    ars_optimizer.load()
    if normalizer_params_path is not None:
        train_normalizer.save_state(policy_params_path)
    if policy_params_path is not None:
        th.save(train_policy.state_dict(), normalizer_params_path)
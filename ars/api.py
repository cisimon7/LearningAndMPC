from abc import ABC
from copy import deepcopy
from typing import Callable, Any, Tuple

import gym
import numpy as np
import torch as th
from gym.core import ObsType, ActType
from tqdm import tqdm
from .Normalizer import Normalizer
from .ARSOptimizer import ARSOptimizer


def ars_minimize(
        obj_func: Callable,
        n_vars=1,
        n_steps=1_000,
        on_step: Callable[[float], Any] = lambda goodness: None,
        **ars_opti_kwargs
):
    class MinimizationEnv(gym.Env, ABC):
        def __init__(self):
            self.observation_space = gym.spaces.Discrete(1)
            self.action_space = gym.spaces.Box(
                low=np.array([-np.inf for _ in range(n_vars)]),
                high=np.array([np.inf for _ in range(n_vars)])
            )

        def reset(self, seed=None, options=None) -> Tuple[ObsType, dict]:
            return 0, {}

        def step(self, min_args: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
            reward = - obj_func(th.from_numpy(min_args))
            return 0, reward.numpy(), True, False, {}

    class MinimizationPolicy(th.nn.Module):
        def __init__(self):
            super(MinimizationPolicy, self).__init__()
            self.min_args = th.nn.Parameter(th.Tensor([0 for _ in range(n_vars)]))

        def forward(self, nothing):
            return self.min_args

    policy = MinimizationPolicy()
    optimizer = ARSOptimizer(
        policy.min_args,
        env=MinimizationEnv(),
        action_sz=1,
        policy=policy,
        sdv=1E-3,
        **ars_opti_kwargs
    )

    with tqdm(total=n_steps, postfix={"loss": th.inf}) as tqdm_updater:
        for t in range(1, n_steps + 1):
            optimizer.step()

            tqdm_updater.update()
            goodness = optimizer.goodness
            tqdm_updater.set_postfix({"goodness": goodness})
            on_step(goodness)

    print(f"Minimum at: {policy.min_args.detach().numpy()}")


def ars_policy_train(
        train_env: gym.Env,
        train_policy: th.nn.Module,
        train_normalizer: Normalizer = None,
        train_steps=100,
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

    class WrapperModule(th.nn.Module):
        def __init__(self):
            super(WrapperModule, self).__init__()
            self.model = deepcopy(train_policy)

        def forward(self, state):
            x = state
            x = train_normalizer.obs_norm(x)  # TODO(Look into using normalization of gym environment)
            action = self.model(th.Tensor(x))
            return action

    ars_cartpole_opti = ARSOptimizer(parameters=param_vector, n_directions=n_directions, env=train_env, action_sz=4,
                                     sdv=0.05, step_sz=0.02, policy=train_policy, normalizer=train_normalizer,
                                     hrz=1_000)

    goodness, prev_goodness = - np.inf, - np.inf
    with tqdm(total=train_steps, postfix={"goodness": goodness}) as tqdm_:
        for t in range(1, train_steps + 1):
            ars_cartpole_opti.step()
            tqdm_.update()
            goodness = ars_cartpole_opti.goodness
            tqdm_.set_postfix({"goodness": goodness})

            on_step(goodness, t)

            if save_on_improve and (goodness > prev_goodness):
                th.nn.utils.vector_to_parameters(
                    ars_cartpole_opti.param_groups[0]["params"][0],
                    train_policy.parameters()
                )
                train_normalizer.save_state(policy_params_path + f"_{np.round(goodness, 4)}")
                th.save(train_policy.state_dict(), normalizer_params_path + f"_{np.round(goodness, 4)}")
                prev_goodness = goodness

    th.nn.utils.vector_to_parameters(
        ars_cartpole_opti.param_groups[0]["params"][0],
        train_policy.parameters()
    )
    if normalizer_params_path is not None:
        train_normalizer.save_state(policy_params_path)
    if policy_params_path is not None:
        th.save(train_policy.state_dict(), normalizer_params_path)


def ars_policy_eval(
        eval_env: gym.Env,
        eval_policy: th.nn.Module,
        policy_post_process: Callable[[th.Tensor], np.ndarray] = lambda tensor: tensor.detach().numpy(),
        eval_normalizer: Normalizer = None,
        policy_params_path: str = None,
        normalizer_params_path: str = None,
        eval_steps=10,
        on_step: Callable[[float, int], Any] = lambda goodness, step: None,
):
    obs_dim = eval_env.observation_space.shape[0]
    action_dim = 1 if isinstance(eval_env.action_space, gym.spaces.Discrete) else eval_env.action_space.shape[0]

    if policy_params_path is not None:
        eval_policy.load_state_dict(th.load(policy_params_path))

    eval_normalizer = eval_normalizer if eval_normalizer else Normalizer(obs_dim)
    if normalizer_params_path is not None:
        eval_normalizer.load_state(normalizer_params_path)

    reward_sequence = []

    def policy(state):
        state = eval_normalizer.normalize(state)
        state = th.FloatTensor(state)
        return eval_policy(state)

    for i in range(eval_steps):

        x0, _ = eval_env.reset()
        done, fitness = False, 0
        while not done:
            action = policy(x0)
            action = policy_post_process(action)
            x0, reward, terminated, truncated, _ = eval_env.step(action)
            fitness += reward
            done = terminated or truncated

        on_step(fitness, i)
        reward_sequence.append(np.asarray(fitness))

    avg_reward = np.mean(reward_sequence)
    print(f"Final {avg_reward=}")

from copy import deepcopy
from functools import partial
from typing import Callable, Any

import functorch
import gym
import numpy as np
import torch as th
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
    class ObjEnv:
        def __init__(self, params):
            self.params = params

        def reset(self):
            return self.params

        def step(self, action):
            x = self.params
            return None, - obj_func(x), True

    xy_t = th.Tensor([0 for _ in range(n_vars)])
    optimizer = ARSOptimizer(
        xy_t,
        get_env=lambda params: ObjEnv(params),
        action_sz=1,
        get_policy=(lambda params, normalizer: (lambda x: None)),
        sdv=1E-3,
        **ars_opti_kwargs
    )

    with tqdm(total=n_steps, postfix={"loss": th.inf}) as tqdm_updater:
        for t in range(1, n_steps + 1):
            optimizer.step()

            tqdm_updater.update()
            goodness = optimizer.goodness
            # if t % 10 == 0:
            tqdm_updater.set_postfix({"goodness": goodness})

            on_step(goodness)

    xy_t.detach_()
    print(f"Minimum at: {xy_t}")


def ars_policy_train(
        train_env: gym.Env,
        train_policy: th.nn.Module,
        policy_post_process: Callable[[th.Tensor], np.ndarray] = lambda tensor: tensor.detach().numpy(),
        train_normalizer: Normalizer = None,
        train_steps=100,
        policy_params_path: str = None,
        normalizer_params_path: str = None,
        on_step: Callable[[float, int], Any] = lambda goodness, step: None,
        save_on_improve=False
):
    obs_dim = train_env.observation_space.shape[0]
    action_dim = 1 if isinstance(train_env.action_space, gym.spaces.Discrete) else train_env.action_space.shape[0]

    if train_normalizer is None:
        train_normalizer = Normalizer(obs_dim)

    # TODO(Probably why model parameter not updating in place)
    param_vector = th.nn.utils.parameters_to_vector(train_policy.parameters()).detach().cpu()

    def get_policy(params, normalizer):
        model = deepcopy(train_policy)
        print(params.shape)
        # TODO(Look into composing the models for the batch equivalent)
        # TODO(Look into vmap the vector_to_parameters function for the batching)
        th.nn.utils.vector_to_parameters(params, model.parameters())

        def forward(state):
            x = state
            x = normalizer.obs_norm(x)
            action = model(th.Tensor(x))
            action = policy_post_process(action)
            return action

        return forward

    def get_env(params):
        class ARSEnv:
            def __init__(self, env: gym.Env):
                self.env = env

            def step(self, action):
                x, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                return x, reward, done

            def reset(self):
                return self.env.reset()[0]

        return ARSEnv(train_env)

    ars_cartpole_opti = ARSOptimizer(parameters=param_vector, n_directions=50, get_env=get_env, action_sz=4, sdv=0.05,
                                     step_sz=0.02, get_policy=get_policy, normalizer=train_normalizer, hrz=1_000)

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

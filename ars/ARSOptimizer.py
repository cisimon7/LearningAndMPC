# https://arxiv.org/abs/1803.07055
# page 6
import gym
import copy
import math
import torch as th
import numpy as np
import functorch as fth
from torch import Tensor
from copy import deepcopy
from typing import Optional
from multiprocessing import Pool
from torch.optim import Optimizer
from .Normalizer import Normalizer
from gym.spaces import Discrete, MultiDiscrete


class ARSOptimizer(Optimizer):
    def __init__(self, env: gym.Env, policy: th.nn.Module, lr=1E-2, sdv=1E-3, n_directions=16,
                 n_choice=None, hrz=1, normalizer=None, alive_bonus=0):

        # ars directional requirements
        assert (n_directions % 2) == 0
        n_choice = int(n_directions / 2) if n_choice is None else n_choice
        assert n_choice <= n_directions

        # environment input and putput shapes and sizes
        obs_shape = ((1,) if (isinstance(env.observation_space, Discrete)
                              or isinstance(env.observation_space, MultiDiscrete))
                     else env.observation_space.sample().shape)
        obs_sz = math.prod(obs_shape)
        act_shape = ((1,) if (isinstance(env.action_space, Discrete)
                              or isinstance(env.action_space, MultiDiscrete))
                     else env.action_space.sample().shape)
        action_sz = math.prod(act_shape)

        # Initialize parameters, mean and covariance matrices to zero
        parameters = th.nn.utils.parameters_to_vector(policy.parameters()).detach().cpu()
        parameters.data.mul_(0)
        super().__init__([parameters], dict())
        self.normalizer = normalizer if normalizer else Normalizer(obs_sz)

        self.n_choice, self.duration, self.sdv = n_choice, hrz, sdv
        self.loss, self.env, self.policy = -th.inf, env, policy
        self.n_directions, self.step_sz, self.alive_bonus = n_directions, lr, alive_bonus

        # Things for parallelzing
        # TODO(replace with python multiprocessing)
        # self.vec_env = gym.vector.SyncVectorEnv([lambda: env for _ in range(n_directions)])
        # self.vec_models, self.vec_params, self.vec_buffers = fth.combine_state_for_ensemble(
        #     [copy.deepcopy(policy) for _ in range(n_directions)]
        # )  # TODO(Model not changing on each step)

        self.shapes = []  # store shapes of different layers of model
        for param in self.policy.parameters():
            self.shapes.append(param.shape)

    # Reshape parameter vector into model parameterm structure
    def reshape_tensor(self, tensor: Tensor):
        parameters, start = (), 0
        for layer in self.shapes:
            length = start + math.prod(layer)
            parameters += (tensor.index_select(0, th.arange(start=start, end=length)).view(layer),)
            start += length

        return parameters

    def vec_query_oracle(self, parameters) -> Tensor:
        def explore(params: Tensor) -> Tensor:
            # Loading parameters to model
            policy = deepcopy(self.policy)
            th.nn.utils.vector_to_parameters(params, policy.parameters())

            env, duration = self.env, self.duration
            obs, _ = env.reset()
            rwds, done, step = [], False, 0

            while step < duration:  # TODO(same length in order for pytorch vectorization)
                step += 1
                if done:
                    rwds.append(0)
                else:
                    obs = self.normalizer.obs_norm(th.from_numpy(obs))
                    action = policy(obs).detach().numpy()
                    obs, rwd, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    rwds.append(rwd)

            return th.tensor(rwds)

        # with Pool(5) as p:
        #     vec_rwds = p.map(explore, parameters)
        vec_rwds = []
        for param in parameters:
            vec_rwds.append(explore(param))

        return th.vstack(vec_rwds)

    def step(self, closure=None) -> Optional[float]:

        parameters: Tensor = self.param_groups[0]["params"][0]

        # Line 4 of Algorithm: i.i.d standard normal distribution
        # Using a standard deviation of 0.5 so as most of generated vectors would be within a distance of
        # 1 from mean and the step size can be used to control it
        deltas = th.normal(mean=0.0, std=1.0, size=(self.n_directions, *parameters.shape))

        # Line 5 of Algorithm (pytorch broadcasting)
        deltas_plus = th.add(parameters, (self.sdv * deltas))
        deltas_minus = th.sub(parameters, (self.sdv * deltas))

        # Preprocessing to load parameters into model structure
        # vec_deltas_plus = fth.vmap(self.reshape_tensor)(deltas_plus)
        # vec_deltas_minus = fth.vmap(self.reshape_tensor)(deltas_minus)

        vec_rwd_roll_p = self.vec_query_oracle(deltas_plus)
        vec_rwd_roll_m = self.vec_query_oracle(deltas_minus)

        rwd_p = vec_rwd_roll_p.sum(dim=1, dtype=th.float32)
        rwd_m = vec_rwd_roll_m.sum(dim=1, dtype=th.float32)

        # Line 6 of Algorithm (Sorting by max of either positive or negative)
        rwd_max = th.maximum(rwd_p, rwd_m)
        arg_rwd_max = rwd_max.argsort(dim=0, descending=True)

        deviations_sorted = deltas[arg_rwd_max][:self.n_choice]
        rwd_plus_sorted = rwd_p[arg_rwd_max][:self.n_choice]
        rwd_minus_sorted = rwd_m[arg_rwd_max][:self.n_choice]

        # Line 7 of Algorithm
        stack_rwds = th.hstack([rwd_plus_sorted, rwd_minus_sorted])
        sdv_rwd = th.std(stack_rwds)  # .clip(1e-6)
        grad = ((self.step_sz / (self.n_choice * sdv_rwd)) *
                (rwd_plus_sorted - rwd_minus_sorted) @ deviations_sorted)
        parameters.data.add_(grad)

        mean_rwd = th.mean(stack_rwds)
        if mean_rwd > self.loss:
            self.loss = mean_rwd

        return None

    def load(self):
        th.nn.utils.vector_to_parameters(
            self.param_groups[0]["params"][0],
            self.policy.parameters()
        )

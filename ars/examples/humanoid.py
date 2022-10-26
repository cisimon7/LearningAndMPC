from copy import deepcopy

import gym
import numpy as np
import torch as th
from torch import Tensor
from tqdm import tqdm

from ars import ars_policy_eval, ars_policy_train
from ars.ARSOptimizer import ARSOptimizer
from ars.Normalizer import Normalizer

# writer = SummaryWriter("logs")

env_train = gym.make("Humanoid-v4")
env_train.reset()

humanoid_normalizer = Normalizer(376)


class ConstScaleLayer(th.nn.Module):
    def __init__(self, constant: th.float):
        super(ConstScaleLayer, self).__init__()
        self.constant = constant

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor * self.constant


humanoid_policy = th.nn.Sequential(
    th.nn.Linear(in_features=376, out_features=17, bias=True),
    th.nn.Tanh(),
    ConstScaleLayer(constant=0.4)
)

if __name__ == '__main__':
    # ars_policy_train(
    #     train_env=gym.make("Humanoid-v4"),
    #     train_policy=humanoid_policy,
    #     train_normalizer=humanoid_normalizer,
    #     train_steps=1
    # )
    ars_policy_eval(
        eval_env=gym.make("Humanoid-v4", render_mode="human", terminate_when_unhealthy=True),
        eval_policy=humanoid_policy,
        policy_params_path="../models/best/humanoid_518.0877",
        normalizer_params_path="../models/best/humanoid_518.0877.npz",
        eval_steps=1_000
    )

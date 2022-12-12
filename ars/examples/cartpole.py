from functools import partial

import gym
import numpy as np
import torch as th
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ars import Normalizer, ars_policy_train, ars_policy_eval

writer = SummaryWriter("../../logs/ars/cartpole")


def tensor_board(title: str, params: Tensor, goodness: float, count: int):
    values = params.tolist()
    names = [f"x{i}" for i in range(len(values))]

    writer.add_scalar(title + "/objective", goodness, count)
    writer.add_scalars(title + "/parameters", dict(zip(names, values)), count)


env_train = gym.make("CartPole-v1")
env_train.reset()

normalizer_main = Normalizer(4)
cartpole_model = th.nn.Sequential(
    th.nn.Linear(in_features=4, out_features=2, bias=True),
    th.nn.Tanh()
)

if __name__ == '__main__':
    # ars_policy_train(
    #     train_env=gym.make("CartPole-v1"),
    #     train_policy=cartpole_model,
    #     train_steps=100,
    #     policy_post_process=lambda action: np.argmax(action.abs().detach().numpy()),
    #     # on_step=partial(tensor_board, "CartPole"),
    #     save_on_improve=True,
    #     policy_params_path="../models/cartpole/temp_",
    #     normalizer_params_path="../models/cartpole/temp_"
    # )
    ars_policy_eval(
        eval_env=gym.make("CartPole-v1", render_mode="human"),
        eval_policy=cartpole_model,
        eval_normalizer=normalizer_main,
        policy_post_process=lambda action: np.argmax(action.abs().detach().numpy()),
        policy_params_path="../models/cartpole/goodness__500.0",
        normalizer_params_path="../models/cartpole/goodness__500.0.npz",
        eval_steps=100
    )

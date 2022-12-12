import gym
import torch as th
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ars import ars_policy_eval, ars_policy_train
from ars.Normalizer import Normalizer

writer = SummaryWriter("../../logs/ars/cheetah")

env_train = gym.make("HalfCheetah-v4")
env_train.reset()

cheetah_normalizer = Normalizer(17)


cheetah_policy = th.nn.Sequential(
    th.nn.Linear(in_features=17, out_features=6, bias=True),
    th.nn.Tanh()
)

if __name__ == '__main__':
    ars_policy_train(
        train_env=gym.make("HalfCheetah-v4"),
        train_policy=cheetah_policy,
        train_normalizer=cheetah_normalizer,
        train_steps=1_000,
        # on_step=lambda fitness, step: writer.add_scalar("cheetah", fitness, step),
        save_on_improve=True,
        policy_params_path="../models/cheetah/goodness_",
        normalizer_params_path="../models/cheetah/goodness_"
    )
    ars_policy_eval(
        eval_env=gym.make("HalfCheetah-v4", render_mode="human"),
        eval_policy=cheetah_policy,
        # policy_params_path="../models/cheetah/goodness__1188",
        # normalizer_params_path="../models/cheetah/goodness__1188.npz",
        eval_steps=1_000
    )

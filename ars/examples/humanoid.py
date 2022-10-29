import gym
import torch as th
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from ars import ars_policy_eval, ars_policy_train
from ars.Normalizer import Normalizer

writer = SummaryWriter("../../logs/ars/humanoid")

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

# humanoid_policy = th.nn.Sequential(
#     th.nn.Linear(in_features=376, out_features=752, bias=True),
#     th.nn.Tanh(),
#     th.nn.Linear(in_features=752, out_features=752, bias=True),
#     th.nn.Tanh(),
#     th.nn.Linear(in_features=752, out_features=376, bias=True),
#     th.nn.Tanh(),
#     th.nn.Linear(in_features=376, out_features=376, bias=True),
#     th.nn.Tanh(),
#     th.nn.Linear(in_features=376, out_features=188, bias=True),
#     th.nn.Tanh(),
#     th.nn.Linear(in_features=188, out_features=47, bias=True),
#     th.nn.Tanh(),
#     th.nn.Linear(in_features=47, out_features=17, bias=True),
#     th.nn.Tanh(),
#     ConstScaleLayer(constant=0.4)
# )

if __name__ == '__main__':
    # ars_policy_train(
    #     train_env=gym.make("Humanoid-v4"),
    #     train_policy=humanoid_policy,
    #     train_normalizer=humanoid_normalizer,
    #     train_steps=200_000,
    #     on_step=lambda fitness, step: writer.add_scalar("humanoid2", fitness, step),
    #     save_on_improve=True,
    #     policy_params_path="../models/humanoid/goodness2_",
    #     normalizer_params_path="../models/humanoid/goodness2_"
    # )
    ars_policy_eval(
        eval_env=gym.make("Humanoid-v4", render_mode="human", terminate_when_unhealthy=True),
        eval_policy=humanoid_policy,
        policy_params_path="../models/humanoid/goodness__1188",
        normalizer_params_path="../models/humanoid/goodness__1188.npz",
        eval_steps=1_000
    )

import gym
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

from ars import Normalizer, ars_policy_train, ars_policy_eval

# writer = SummaryWriter("../../logs/ars/cartpole")

env_train = gym.make("CartPole-v1")
env_train.reset()

normalizer_main = Normalizer(4)


class CartPoleModule(th.nn.Module):
    def __init__(self):
        super(CartPoleModule, self).__init__()
        self.network = th.nn.Sequential(
            th.nn.Linear(in_features=4, out_features=2, bias=True),
            th.nn.Softmax(dim=-1)
        )

    def forward(self, x: th.Tensor):
        x = self.network(x)
        return x.argmax()


if __name__ == '__main__':
    cartpole_module = CartPoleModule()
    ars_policy_train(
        train_env=gym.make("CartPole-v1"),
        train_policy=cartpole_module,
        train_steps=1_000,
        # on_step=lambda fitness, step: writer.add_scalar("humanoid", fitness, step),
        policy_params_path="../models/cartpole/batch_model/best_",
        normalizer_params_path="../models/cartpole/batch_model/best_",
        save_on_improve=True,
    )
    ars_policy_eval(
        eval_env=gym.make("CartPole-v1", render_mode="human"),
        eval_policy=cartpole_module,
        eval_normalizer=normalizer_main,
        policy_post_process=lambda action: np.argmax(action.abs().detach().numpy()),
        # policy_params_path="../models/cartpole/batch_model/best__500.0",
        # normalizer_params_path="../models/cartpole/batch_model/best__500.0.npz",
        eval_steps=100
    )

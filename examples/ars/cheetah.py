import gym
import torch as th
from torch.utils.tensorboard import SummaryWriter

from ars import Normalizer, ars_policy_train, ars_policy_eval

# writer = SummaryWriter("../../logs/ars/cartpole")


class CheetahModule(th.nn.Module):
    def __init__(self):
        super(CheetahModule, self).__init__()
        self.network = th.nn.Sequential(
            th.nn.Linear(in_features=17, out_features=6, bias=True),
            th.nn.Tanh()
        )

    def forward(self, x: th.Tensor):
        return self.network(x.float())


if __name__ == '__main__':
    th.manual_seed(73)  # 42, 73
    cheetah_module = CheetahModule()

    run_train = False
    if run_train:
        ars_policy_train(
            train_env=gym.make("HalfCheetah-v4", ctrl_cost_weight=0.3),
            train_policy=cheetah_module,
            train_steps=100,
            policy_params_path="examples/ars/models/cheetah/best",
            duration=1_000,
            save_on_improve=False,
        )

    ars_policy_eval(
        eval_env=gym.make("HalfCheetah-v4", render_mode="human"),
        eval_policy=cheetah_module,
        policy_params_path="examples/ars/models/cheetah/best_35",
        eval_steps=100
    )


# export CC=/opt/homebrew/bin/gcc-12

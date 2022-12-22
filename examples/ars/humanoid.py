import gym
import torch as th
from torch.utils.tensorboard import SummaryWriter

from ars import Normalizer, ars_policy_train, ars_policy_eval

# writer = SummaryWriter("../../logs/ars/cartpole")


class HumanoidModule(th.nn.Module):
    def __init__(self):
        super(HumanoidModule, self).__init__()
        self.network = th.nn.Sequential(
            th.nn.Linear(in_features=378, out_features=17, bias=True),
            th.nn.Tanh()
        )

    def forward(self, x: th.Tensor):
        return 0.4 * self.network(x.float())


if __name__ == '__main__':
    th.manual_seed(73)  # 42, 73
    humanoid_module = HumanoidModule()

    run_train = False
    if run_train:
        ars_policy_train(
            train_env=gym.make(
                "Humanoid-v4",
                ctrl_cost_weight=0.2,
                exclude_current_positions_from_observation=False
            ),
            train_policy=humanoid_module,
            train_steps=100,
            policy_params_path="examples/ars/models/humanoid/best",
            duration=1_000,
            save_on_improve=True,
        )

    ars_policy_eval(
        eval_env=gym.make(
            "Humanoid-v4",
            exclude_current_positions_from_observation=False,
            render_mode="human"
        ),
        eval_policy=humanoid_module,
        policy_params_path="examples/ars/models/humanoid/best_429",
        eval_steps=100
    )


# export CC=/opt/homebrew/bin/gcc-12

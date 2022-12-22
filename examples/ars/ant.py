import gym
import torch as th
from ars import ars_policy_train, ars_policy_eval


class CartPoleModule(th.nn.Module):
    def __init__(self):
        super(CartPoleModule, self).__init__()
        self.network = th.nn.Sequential(
            th.nn.Linear(in_features=27, out_features=8, bias=True),
            th.nn.Tanh()
        )

    def forward(self, x: th.Tensor):
        return self.network(x.float())


if __name__ == '__main__':
    th.manual_seed(73)  # 42, 73
    cartpole_module = CartPoleModule()

    run_train = True
    if run_train:
        ars_policy_train(
            train_env=gym.make("Ant-v4", ctrl_cost_weight=1),
            train_policy=cartpole_module,
            train_steps=100,
            policy_params_path="examples/ars/models/ant/best",
            save_on_improve=False,
        )

    ars_policy_eval(
        eval_env=gym.make("Ant-v4", render_mode="human"),
        eval_policy=cartpole_module,
        # policy_params_path="examples/ars/models/ant/best_499",
        eval_steps=100
    )

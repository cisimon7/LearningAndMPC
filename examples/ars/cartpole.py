import gym
import torch as th
from ars import Normalizer, ars_policy_train, ars_policy_eval


env_train = gym.make("CartPole-v1")
env_train.reset()

normalizer_main = Normalizer(4)


class CartPoleModule(th.nn.Module):
    def __init__(self):
        super(CartPoleModule, self).__init__()
        self.network = th.nn.Sequential(
            th.nn.Linear(in_features=4, out_features=2, bias=True),
            th.nn.Tanh()
        )

    def forward(self, x: th.Tensor):
        x = self.network(x).abs()
        return x.argmax()


if __name__ == '__main__':
    th.manual_seed(73)  # 42, 73
    cartpole_module = CartPoleModule()

    run_train = False
    if run_train:
        ars_policy_train(
            train_env=gym.make("CartPole-v1"),
            train_policy=cartpole_module,
            train_steps=1,
            policy_params_path="examples/ars/models/cartpole/best",
            save_on_improve=False,
        )

    ars_policy_eval(
        eval_env=gym.make("CartPole-v1", render_mode="human"),
        eval_policy=cartpole_module,
        policy_params_path="examples/ars/models/cartpole/best_499",
        eval_steps=100
    )

import gym
import numpy as np
import torch as th
from ars import Normalizer, ars_policy_train, ars_policy_eval

# writer = SummaryWriter("logs")

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
    #     policy_post_process=lambda action: np.argmax(action.abs().detach().numpy()),
    #     train_steps=1
    # )
    ars_policy_eval(
        eval_env=gym.make("CartPole-v1", render_mode="human"),
        eval_policy=cartpole_model,
        eval_normalizer=normalizer_main,
        policy_post_process=lambda action: np.argmax(action.abs().detach().numpy()),
        policy_params_path="../models/best/cartpole_500.0",
        normalizer_params_path="../models/best/cartpole_500.0.npz",
    )

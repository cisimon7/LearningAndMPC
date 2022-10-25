from copy import deepcopy

import gym
import torch as th
import numpy as np
from tqdm import tqdm
from Normalizer import Normalizer
from ARSOptimizer import ARSOptimizer

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

env_train = gym.make("CartPole-v1")
env_train.reset()

normalizer_main = Normalizer(4)
cartpole_model = th.nn.Sequential(
    th.nn.Linear(in_features=4, out_features=2, bias=True),
    th.nn.Tanh()
)


def train(steps=100, log_name="reward"):
    parameters = th.nn.utils.parameters_to_vector(cartpole_model.parameters()).detach().cpu()

    def get_policy_cart(params, normalizer):
        model = deepcopy(cartpole_model)
        th.nn.utils.vector_to_parameters(params, model.parameters())

        def forward(state):
            x = state
            x = normalizer.obs_norm(x)
            action = model(th.Tensor(x))
            return action

        return forward

    def get_env_cart(params):
        class ARSCartEnv:
            def __init__(self, env: gym.Env):
                self.env = env

            def step(self, action):
                action = np.argmax(action.abs().detach().numpy())
                x, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                return x, reward, done

            def reset(self):
                return self.env.reset()[0]

        return ARSCartEnv(env_train)

    ars_cartpole_opti = ARSOptimizer(
        parameters=parameters,
        n_directions=50,
        get_env=get_env_cart,
        action_sz=4,
        sdv=0.05,
        step_sz=0.02,
        get_policy=get_policy_cart,
        normalizer=normalizer_main,
        hrz=1_000
    )

    goodness = - np.inf
    with tqdm(total=steps, postfix={"goodness": goodness}) as tqdm_:
        for t in range(1, steps + 1):
            ars_cartpole_opti.step()
            tqdm_.update()
            goodness = ars_cartpole_opti.goodness
            tqdm_.set_postfix({"goodness": goodness})

            writer.add_scalar(f"ars_cartpole/{log_name}", goodness, t)

    th.nn.utils.vector_to_parameters(
        ars_cartpole_opti.param_groups[0]["params"][0],
        cartpole_model.parameters()
    )
    normalizer_main.save_state(f"../models/ars/ars_normalizer_{np.round(goodness, 4)}")
    th.save(cartpole_model.state_dict(), f"../models/ars/ars_model_{np.round(goodness, 4)}")


if __name__ == '__main__':
    # train(steps=100)

    env_test = gym.make("CartPole-v1", render_mode="human")

    cartpole_model.load_state_dict(th.load("../models/ars/ars_model_500.0"))
    normalizer_main.load_state("../models/ars/ars_normalizer_500.0.npz")
    reward_sequence = []


    def policy(state):
        state = normalizer_main.normalize(state)
        state = th.FloatTensor(state)
        return cartpole_model(state)


    for i in range(10):
        x0, _ = env_test.reset()

        done, fitness = False, 0
        while not done:
            action = policy(x0)
            action = np.argmax(action.abs().detach().numpy())
            x0, reward, done, _, _ = env_test.step(action)
            fitness += reward

        reward_sequence.append(np.asarray(fitness))

    avg_reward = np.mean(reward_sequence)
    print(f"Final {avg_reward=}")

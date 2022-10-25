from copy import deepcopy

import gym
import numpy as np
import torch as th
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ARSOptimizer import ARSOptimizer
from Normalizer import Normalizer

writer = SummaryWriter("logs")

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
    th.nn.Linear(in_features=376, out_features=376, bias=True),
    th.nn.Tanh(),
    th.nn.Linear(in_features=376, out_features=188, bias=True),
    th.nn.Tanh(),
    th.nn.Linear(in_features=188, out_features=94, bias=True),
    th.nn.Tanh(),
    th.nn.Linear(in_features=94, out_features=17, bias=True),
    th.nn.Tanh(),
    ConstScaleLayer(constant=0.4)
)


def train(steps=500, log_name="reward"):
    policy_params = th.nn.utils.parameters_to_vector(humanoid_policy.parameters()).detach().cpu()

    def get_policy(params, normalizer):
        model = deepcopy(humanoid_policy)
        th.nn.utils.vector_to_parameters(params, model.parameters())

        def forward(state):
            x = state
            x = normalizer.obs_norm(x)
            action = model(th.Tensor(x))
            return action

        return forward

    def get_env(params):
        class ARSHumanoid:
            def __init__(self, humanoid_env: gym.Env):
                self.env = humanoid_env

            def step(self, action):
                x, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                return x, reward, done

            def reset(self):
                return self.env.reset()[0]

        return ARSHumanoid(env_train)

    ars_cartpole_opti = ARSOptimizer(
        parameters=policy_params,
        n_directions=100,
        get_env=get_env,
        action_sz=17,
        sdv=0.05,
        step_sz=0.02,
        get_policy=get_policy,
        normalizer=humanoid_normalizer,
        hrz=1_000
    )

    fitness = - np.inf
    with tqdm(total=steps, postfix={"fitness": fitness}) as tqdm_:
        for t in range(1, steps + 1):
            ars_cartpole_opti.step()
            tqdm_.update()
            fitness = ars_cartpole_opti.goodness
            tqdm_.set_postfix({"fitness": fitness})

            writer.add_scalar(f"ars_humanoid/{log_name}", fitness, t)

    th.nn.utils.vector_to_parameters(
        ars_cartpole_opti.param_groups[0]["params"][0],
        humanoid_policy.parameters()
    )
    humanoid_normalizer.save_state(f"../models/ars/humanoid_{np.round(fitness, 4)}")
    th.save(humanoid_policy.state_dict(), f"../models/ars/humanoid_{np.round(fitness, 4)}")


if __name__ == '__main__':

    train()

    # humanoid_policy.load_state_dict(th.load("../models/ars/humanoid_518.0877"))
    # humanoid_normalizer.load_state("../models/ars/humanoid_518.0877.npz")
    reward_sequence = []

    env = gym.make("Humanoid-v4", render_mode="human", terminate_when_unhealthy=True)

    terminated, truncated, fitness = False, False, 0
    observation, info = env.reset(seed=42)
    for _ in range(1_000):
        # action = env.action_space.sample()
        observation = th.from_numpy(observation).float()
        action = humanoid_policy(observation).detach().numpy()
        observation, reward, terminated, truncated, _ = env.step(action)
        fitness += reward

        reward_sequence.append(np.asarray(fitness))

        if terminated or truncated:
            env.reset()

    avg_reward = np.mean(reward_sequence).round(4)
    print(f"Final {avg_reward=}")

    env.close()

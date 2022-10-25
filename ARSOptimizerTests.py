from collections.abc import Iterator

import gym
import torch as th
import numpy as np
import torch.nn.utils
from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, SGD
from Normalizer import Normalizer
from ARSOptimizer import ARSOptimizer


def rosenbrock(xy):
    x, y = xy
    return th.pow(1 - x, 2) + 100 * th.pow(y - th.pow(x, 2), 2)


class ARSOptimizerTests:
    def __init__(self):
        self.path = None
        self.env_train = gym.make("CartPole-v1")
        self.env_train.reset()
        self.env_test = gym.make("CartPole-v1", render_mode="human")

        self.normalizer = Normalizer(4)
        self.cartpole_model = th.nn.Sequential(
            th.nn.Linear(in_features=4, out_features=2, bias=True),
            th.nn.Tanh()
        )

    def opti_rosenbrock(self, x_init, n_steps=2_000):
        self.path = np.empty((n_steps + 1, 2))
        self.path[0, :] = x_init
        xy_t = th.tensor(x_init, requires_grad=True)
        # optimizer = Adam([xy_t])
        optimizer = SGD([xy_t], lr=0.01)

        with tqdm(total=n_steps, postfix={"loss": th.inf}) as tqdm_:
            for t in range(1, n_steps + 1):
                loss = rosenbrock(xy_t)
                optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(xy_t, 1.0)
                optimizer.step()

                self.path[t, :] = xy_t.detach().numpy()
                tqdm_.update()
                if t % 10 == 0:
                    tqdm_.set_postfix({"loss": loss})

        xy_t.detach_()
        print(f"Minimum at: {xy_t}")

    def ars_rosenbrock(self, x_init, n_steps=100):
        self.path = np.empty((n_steps + 1, 2))
        self.path[0, :] = x_init
        xy_t = th.tensor(x_init)

        class RosenbrockEnv:
            def __init__(self, params):
                self.params = params

            def reset(self):
                return self.params

            def step(self, action):
                x = self.params
                return None, -rosenbrock(x), False

        optimizer = ARSOptimizer(
            xy_t,
            get_env=lambda params: RosenbrockEnv(params),
            action_sz=1,
            get_policy=(lambda params, normalizer: (lambda x: None)),
            sdv=1E-3
        )

        with tqdm(total=n_steps, postfix={"loss": th.inf}) as tqdm_:
            for t in range(1, n_steps + 1):
                optimizer.step()

                self.path[t, :] = xy_t.detach().numpy()
                tqdm_.update()
                if t % 10 == 0:
                    goodness = optimizer.goodness
                    tqdm_.set_postfix({"goodness": goodness})

        xy_t.detach_()
        print(f"Minimum at: {xy_t}")

    def ars_cartpole_train(self, n_steps=100):
        parameters = th.nn.utils.parameters_to_vector(self.cartpole_model.parameters()).detach().cpu()

        def get_policy_cart(params, normalizer):
            model = deepcopy(self.cartpole_model)
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

            return ARSCartEnv(self.env_train)

        ars_cartpole_opti = ARSOptimizer(
            parameters=parameters,
            n_directions=50,
            get_env=get_env_cart,
            action_sz=4,
            sdv=0.05,
            step_sz=0.02,
            get_policy=get_policy_cart,
            normalizer=self.normalizer,
            hrz=10
        )

        goodness = - np.inf
        with tqdm(total=n_steps, postfix={"goodness": goodness}) as tqdm_:
            for t in range(1, n_steps + 1):
                ars_cartpole_opti.step()
                tqdm_.update()
                goodness = ars_cartpole_opti.goodness
                tqdm_.set_postfix({"goodness": goodness})

        th.nn.utils.vector_to_parameters(
            ars_cartpole_opti.param_groups[0]["params"][0],
            self.cartpole_model.parameters()
        )
        # self.normalizer.save_state(f"models/ars/ars_normalizer_{np.round(goodness, 4)}_good")
        # th.save(self.cartpole_model.state_dict(), f"models/ars/ars_model_{np.round(goodness, 4)}_good")

    def ars_cartpole_evaluate(self):
        # self.cartpole_model.load_state_dict(th.load("models/ars/ars_model_500.0_good"))
        # self.normalizer.load_state("models/ars/ars_normalizer_500.0_good.npz")
        reward_sequence = []
        env = self.env_test

        def policy(state):
            state = self.normalizer.normalize(state)
            state = th.FloatTensor(state)
            return self.cartpole_model(state)

        for i in range(10):
            x0, _ = env.reset()

            done, fitness = False, 0
            while not done:
                action = policy(x0)
                action = np.argmax(action.abs().detach().numpy())
                x0, reward, done, _, _ = env.step(action)
                fitness += reward

            reward_sequence.append(np.asarray(fitness))

        goodness = np.mean(reward_sequence)
        print(f"Final {goodness=}")


if __name__ == '__main__':
    tester = ARSOptimizerTests()
    xy_init = (0.3, -0.8)

    # tester.opti_rosenbrock(xy_init)
    tester.ars_rosenbrock(xy_init)

    # tester.ars_cartpole_train()
    # tester.ars_cartpole_evaluate()

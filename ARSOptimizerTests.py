import gym
import torch
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from torch.optim import Adam, SGD
from ARSOptimizer import ARSOptimizer, Normalizer


def rosenbrock(xy):
    x, y = xy
    return torch.pow(1 - x, 2) + 1 * torch.pow(y - torch.pow(x, 2), 2)
    # return torch.pow(x, 2) + torch.pow(y, 2)


class ARSOptimizerTests:
    def __init__(self):
        self.path = None
        self.env_train = gym.make("CartPole-v1")
        self.env_train.reset()
        self.env_test = gym.make("CartPole-v1", render_mode="human")

        self.normalizer = Normalizer(4)
        self.cartpole_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=4, out_features=2, bias=False),
            torch.nn.Sigmoid()
        )

    def opti_rosenbrock(self, x_init, n_steps=2_000):
        self.path = np.empty((n_steps + 1, 2))
        self.path[0, :] = x_init
        xy_t = torch.tensor(x_init, requires_grad=True)
        # optimizer = Adam([xy_t])
        optimizer = SGD([xy_t], lr=0.01)

        with tqdm(total=n_steps, postfix={"loss": torch.inf}) as tqdm_:
            for t in range(1, n_steps + 1):
                loss = rosenbrock(xy_t)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(xy_t, 1.0)
                optimizer.step()

                self.path[t, :] = xy_t.detach().numpy()
                tqdm_.update()
                if t % 10 == 0:
                    tqdm_.set_postfix({"loss": loss})

        xy_t.detach_()
        print(f"Minimum at: {xy_t}")

    def ars_rosenbrock(self, x_init, n_steps=2_000):
        self.path = np.empty((n_steps + 1, 2))
        self.path[0, :] = x_init
        xy_t = torch.tensor(x_init)
        optimizer = ARSOptimizer(
            [xy_t],
            fwd_fun=lambda params, x: None,
            step_fun=lambda x, action: (None, -rosenbrock(x), False),
            reset_input=lambda params: params[0]
        )

        with tqdm(total=n_steps, postfix={"loss": torch.inf}) as tqdm_:
            for t in range(1, n_steps + 1):
                loss = rosenbrock(xy_t)
                optimizer.step()

                self.path[t, :] = xy_t.detach().numpy()
                tqdm_.update()
                if t % 10 == 0:
                    tqdm_.set_postfix({"loss": loss})

        xy_t.detach_()
        print(f"Minimum at: {xy_t}")

    def ars_cartpole_train(self, n_steps=2_000):

        def cart_step_fun(x, action):
            action = np.argmax(action)
            _, reward, terminated, truncated, _ = self.env_train.step(action)
            done = terminated or truncated
            if done:
                self.env_train.reset()

            return x, reward, done

        def cart_fwd_fun(params, x):
            x = self.normalizer.obs_norm(x)
            model = deepcopy(self.cartpole_model)
            model._parameters = params
            action = model(torch.Tensor(x))
            return action.detach().numpy()

        ars_cartpole_opti = ARSOptimizer(
            parameters=self.cartpole_model.parameters(),
            fwd_fun=cart_fwd_fun,
            step_fun=cart_step_fun,
            reset_input=lambda params: self.env_train.reset()[0],
            normalizer=self.normalizer,
            hrz=100
        )

        goodness = - np.inf
        with tqdm(total=n_steps, postfix={"variance": goodness}) as tqdm_:
            for t in range(1, n_steps + 1):
                ars_cartpole_opti.step()
                tqdm_.update()
                if t % 10 == 0:
                    goodness = ars_cartpole_opti.goodness
                    tqdm_.set_postfix({"loss": ars_cartpole_opti.normalizer.cov.sum()})

        torch.save(self.cartpole_model.state_dict(), f"models/ars/ars_model_{np.round(goodness, 4)}")

    def ars_cartpole_evaluate(self):
        # self.cartpole_model.load_state_dict(torch.load("models/ars/ars_model_0.9"))
        reward_sequence = []
        env = self.env_test

        def policy(state):
            state = self.normalizer.obs_norm(state)
            state = torch.from_numpy(state).float()
            return self.cartpole_model(state)

        for i in range(1_000):
            x0, _ = env.reset()

            done, fitness = False, 0
            while not done:
                action = policy(x0)
                action = action.detach().numpy()
                action = np.argmax(action)
                # action = env.action_space.sample()
                x0, reward, done, _, _ = env.step(action)
                fitness += reward

            reward_sequence.append(np.asarray(fitness))

        goodness = np.mean(reward_sequence)
        print(f"Final {goodness=}")


if __name__ == '__main__':
    tester = ARSOptimizerTests()
    xy_init = (0.3, 0.8)

    # tester.opti_rosenbrock(xy_init)

    tester.ars_rosenbrock(xy_init)

    # tester.ars_cartpole_train()
    # tester.ars_cartpole_evaluate()

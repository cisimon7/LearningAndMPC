import gym
from BenchMarker import measure_time_seconds
from EvolutionaryPolicy import EvolutionaryPolicy


class EvoTests:

    def __init__(self) -> None:
        self.env_train = gym.make("CartPole-v1")
        self.env_test = gym.make("CartPole-v1", render_mode="human")
        self.CartPolePolicy = EvolutionaryPolicy(
            n_input=4,  # Size of observation vector []
            n_output=[0, 1],  # Encoded list of possible actions [left, right]
            layers_heights=[4, 4, 4],  # internal neural net layers number of nodes
        )
        # self.CartPolePolicy.load("../models/evo/model_temp2_1021")
        self.CartPolePolicy.load("models/evo/model_temp_2022")

    def run_policy(self):
        observation, info = self.env_test.reset()
        for _ in range(1000):
            action = self.CartPolePolicy(observation)
            observation, reward, terminated, truncated, _ = self.env_test.step(action)

            if terminated or truncated:
                observation, info = self.env_test.reset()

        self.env_test.close()

    def train_policy(self):
        duration, _ = measure_time_seconds(
            lambda:
            self.CartPolePolicy.evolve(
                self.env_train.reset,
                self.env_train.step,
                population=20,
                n_generations=1_000,
                model_dir="models/evo/model_temp2"
            )
        )
        goodness = self.CartPolePolicy.evaluate(self.env_train.reset, self.env_train.step)
        print(f"Final {goodness=}, in {duration} seconds")

    def evaluate_policy(self):
        duration, goodness = measure_time_seconds(
            lambda:
            self.CartPolePolicy.evaluate(
                self.env_test.reset,
                self.env_test.step,
                n_episodes=100
            )
        )
        print(f"Final {goodness=}, in {duration} seconds")

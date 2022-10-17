import gym
from BenchMarker import measure_time_seconds


class ARSPolicyTests:
    def __init__(self):
        self.env_train = gym.make("CartPole-v1")
        self.env_test = gym.make("CartPole-v1", render_mode="human")
        self.CartPolePolicy = ARSPolicyTests(
            n_input=4,  # Size of observation vector []
            n_output=[0, 1],  # Encoded list of possible actions [left, right]
            layers_heights=[4, 4, 4],  # internal neural net layers number of nodes
        )


if __name__ == '__main__':
    ars_policy = ARSPolicyTests()

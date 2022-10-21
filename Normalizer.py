import numpy as np


class Normalizer:
    def __init__(self, n_inputs):
        self.n = 0
        self.mean = np.zeros(n_inputs)
        self.cov = np.zeros((n_inputs, n_inputs))

        self.A = np.zeros((n_inputs, n_inputs))
        self.b = np.zeros(n_inputs)

    def observe(self, x):
        self.n += 1
        self.A += np.array(x) @ np.array(x).T
        self.b += np.array(x)

        self.mean = (1 / self.n) * self.b
        self.cov = (1 / self.n) * (
                self.A - (self.mean @ self.b.T) - (self.b @ self.mean.T) + (self.n * self.mean @ self.mean.T)
        )

    def normalize(self, inputs):
        return self.cov @ (np.array(inputs) - self.mean)

    def obs_norm(self, inputs):
        self.observe(inputs)
        return self.normalize(inputs)

    def save_state(self, path: str):
        np.savez(path, mean=self.mean, cov=self.cov, A=self.A, b=self.b)

    def load_state(self, path: str):
        states = np.load(path)
        self.mean = states["mean"]
        self.cov = states["cov"]
        self.A = states["A"]
        self.b = states["b"]

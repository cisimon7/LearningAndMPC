# https://math.stackexchange.com/questions/1147084/dynamic-update-of-co-variance-matrix-upon-new-sample
import numpy as np


class Normalizer:
    def __init__(self, n_inputs):
        self.n = 1
        self.mean = np.zeros(n_inputs)
        self.cov = np.eye(n_inputs)

        self.A = np.zeros((n_inputs, n_inputs))
        self.b = np.zeros(n_inputs)

    def observe(self, x: np.ndarray):
        self.n += 1
        self.A += x @ x.T
        self.b += x

        self.mean = (1 / self.n) * self.b
        self.cov = (1 / (self.n - 1)) * (
                self.A - (self.mean @ self.b.T) - (self.b @ self.mean.T) + (self.n * self.mean @ self.mean.T)
        )

    def normalize(self, inputs: np.ndarray):
        return np.diag(np.power(np.diagonal(self.cov), -0.5)) @ (inputs - self.mean)

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

# https://math.stackexchange.com/questions/1147084/dynamic-update-of-co-variance-matrix-upon-new-sample
import numpy as np
import torch as th
from torch import Tensor


class Normalizer:
    DEVICES = ["cpu", "mps", "cuda"]

    def __init__(self, n_inputs: int, device="cpu"):
        assert device in self.DEVICES
        self.device = device
        self.n = 1
        self.mean = th.zeros((n_inputs, n_inputs), device=device, dtype=th.float32)
        self.cov = th.eye(n_inputs, device=device, dtype=th.float32)
        self.A = th.zeros((n_inputs, n_inputs), device=device, dtype=th.float32)
        self.b = th.zeros(n_inputs, device=device, dtype=th.float32)

    def observe(self, x: Tensor):
        self.n += 1
        self.A += x @ th.transpose(x, 0, -1)
        self.b += x

        self.mean = th.mul(self.b, (1 / self.n))
        self.cov = (1 / (self.n - 1)) * (
                self.A - (self.mean @ self.b.transpose(0, -1)) -
                (self.b @ self.mean.transpose(0, -1)) +
                (th.mul(self.mean, self.n) @ self.mean.transpose(0, -1)))

    def normalize(self, inputs: Tensor):
        return (inputs - self.mean) @ th.diag(th.pow(1/th.diag(self.cov), -0.5))

    def obs_norm(self, inputs: Tensor):
        self.observe(inputs)
        return self.normalize(inputs)

    def save_state(self, path: str):
        np.savez(
            path,
            mean=self.mean.cpu().numpy(),
            cov=self.cov.cpu().numpy(),
            A=self.A.cpu().numpy(),
            b=self.b.cpu().numpy()
        )

    def load_state(self, path: str):
        states = np.load(path)
        self.mean = states["mean"].to(device=self.device)
        self.cov = states["cov"].to(device=self.device)
        self.A = states["A"].to(device=self.device)
        self.b = states["b"].to(device=self.device)

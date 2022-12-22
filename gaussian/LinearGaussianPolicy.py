import torch as th
from torch import Tensor


class LinearGaussianModule(th.nn.Module):
    def __init__(self, n):
        super(LinearGaussianModule, self).__init__()
        self.mean = th.nn.Parameter(th.randn(n))
        self.variance = th.nn.Parameter(th.randn(n))

    def forward(self, x: Tensor = None):
        return th.distributions.multivariate_normal.MultivariateNormal(self.mean, self.variance).sample()

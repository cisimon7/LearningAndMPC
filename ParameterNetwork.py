from torch import nn
from typing import List
from ARSOptimizer import ARSOptimizer


class LinearARS(nn.Module):
    def __init__(self, n_input: int, n_output: (int | List[int])):
        super(LinearARS).__init__()


if __name__ == '__main__':
    ars_policy = LinearARS(
        n_input=4,
        n_output=[0, 1]
    )

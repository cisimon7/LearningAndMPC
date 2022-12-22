import torch as th


# A method of numerically integrating ordinary differential equations by using a trial step at the midpoint of an
# interval to cancel out lower-order error terms.
class RungeKutta:
    def __init__(self, order: int, step_sz: int, ode_func):
        self.rung_kutta_matrix = th.empty((order, order))
        self.weights = th.empty(order)
        self.nodes = th.empty(order)

    def step(self, x0):
        x1 = x0 + th.sum()
        return x1

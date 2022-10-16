import jax.numpy as jnp
import numpy as np
import jax


# variable = Tuple[float, float]
# ars_minimize(
#     objective=obj_func,
#     constraints=[cons_func1, cons_func2],
#     verbose=1
# )

class AugmentedRandomSearchPolicy:
    def __init__(self):
        pass

    def policy_search(self, reward_fun, step_size=0.02, s_dev=1, n_directions=10, n_choice=None):
        n_choice = int(n_directions / 2) if n_choice is None else n_choice
        error_msg = f"number of top-performing directions {n_choice} should be less than {n_directions}"
        assert n_choice < n_directions, error_msg

        shape = (10, 10)
        params = jnp.zeros(shape)
        deviation = 0
        params_plus = params + deviation  # policy update
        params_minus = params - deviation  # policy update

        return self

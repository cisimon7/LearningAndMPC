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

    def policy_search(self, query_oracle, rwd_fun, step_sz=0.02, sdv=1, n_directions=10, n_choice=None, hrz=1_000):
        n_choice = int(n_directions / 2) if n_choice is None else n_choice
        error_msg = f"number of top-performing directions {n_choice} should be less than {n_directions}"
        assert n_choice < n_directions, error_msg

        end = False

        shape = (10, 10)
        params = jnp.zeros(shape)

        while not end:
            deviations = []
            params_plus = params + deviations  # policy update
            params_minus = params - deviations  # policy update

            rollout_plus = [query_oracle(param, hrz) for param in params_plus]
            rollout_minus = [query_oracle(param, hrz) for param in params_minus]

            rwd_plus = [rwd for (_, rwd) in rollout_plus]
            rwd_minus = [rwd for (_, rwd) in rollout_minus]

            rwd_max = [np.max([rwd_p, rwd_m]) for (rwd_p, rwd_m) in zip(rwd_plus, rwd_minus)]
            rwd_max_sorted = np.argsort(rwd_max)[::-1]

            deviations_sorted = [deviations[idx] for idx in rwd_max_sorted]
            rwd_plus_sorted = [rwd_plus[idx] for idx in rwd_max_sorted]
            rwd_minus_sorted = [rwd_minus[idx] for idx in rwd_max_sorted]

            sdv_rwd = np.std([*rwd_plus_sorted[:n_choice], *rwd_minus_sorted[:n_choice]])

            params += (step_sz / sdv_rwd * n_choice) * np.sum([
                (rwd_plus_sorted[k] - rwd_minus_sorted[k]) * deviations_sorted[k]
                for k in range(n_choice)
            ])

            end = True

        return self

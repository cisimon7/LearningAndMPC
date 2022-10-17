import numpy as np
import pickle
from typing import List
import jax
from jax import jit, vmap
import jax.numpy as jnp
from copy import copy, deepcopy
from functools import partial


# TODO(Extract the params from here so as different architectures can be built and used separately)

class EvolutionaryPolicy:
    def __init__(self, n_input: int, n_output: int | List[int], n_hidden: int = None, layers_heights: List[int] = None):
        self.params = []
        self.n_input, self.n_output = n_input, n_output
        self.n_hidden, self.layers_heights = n_hidden, layers_heights
        self.reset()

    def reset(self):
        self.params = []
        n_input, n_output = self.n_input, self.n_output
        n_hidden, layers_heights = self.n_hidden, self.layers_heights

        if layers_heights is None:
            if isinstance(n_output, int):
                layers_heights = [n_input, n_input, n_output]
            else:
                layers_heights = [n_input, n_input, len(n_output)]
        else:
            if isinstance(n_output, int):
                layers_heights = [n_input, *layers_heights, n_output]
            else:
                layers_heights = [n_input, *layers_heights, len(n_output)]

        key = jax.random.PRNGKey(np.random.randint(0, 50))

        for (n_in, n_out) in zip(layers_heights[:-1], layers_heights[1:]):
            self.params.append(dict(
                weights=jax.random.uniform(key, shape=(n_in, n_out)),
                biases=jax.random.uniform(key, shape=(n_out,))
            ))
            key, _ = jax.random.split(key)

        return deepcopy(self)

    def __call__(self, observation):
        # @partial(jit, static_argnums=[2])
        def forward(params, x, action_set=self.n_output):
            # sensory, internal
            *hidden, last = params
            for layer in hidden:
                x = jax.nn.tanh(x @ layer['weights'] + layer['biases'])

            # motor
            action = x @ last['weights'] + last['biases']

            # set action to be within action type
            action = jnp.array(action_set)[np.argmax(action)] if type(action_set) == list else action

            return action

        a = forward(self.params, observation)
        return np.asarray(a)

    def update(self, change):
        size = len(change)
        for idx in range(size):
            self.params[idx]["weights"] += change[idx]["weights"]

        return self

    def evaluate(self, reset_fun, step_fun, n_episodes=10):
        reward_sequence = []

        for i in range(n_episodes):
            x0, _ = reset_fun(seed=42)

            done, fitness = False, 0
            while not done:
                action = self.__call__(x0)
                x0, reward, done, _, _ = step_fun(action)
                fitness += reward

            reward_sequence.append(np.asarray(fitness))

        return np.mean(np.asarray(reward_sequence))

    def evolve(self, reset_fun, step_fun, population=10, n_generations=300, mutation_step_size=0.02, verbose=True,
               model_dir="model_temp"):

        # Policy iteration or training process
        genotypes = [self.reset() for _ in range(population)]
        best_fit = 0

        for gen in range(n_generations):
            fitnesses = [
                geno.evaluate(reset_fun, step_fun)
                for geno in genotypes
            ]
            ordered_genotypes_idx = np.argsort(fitnesses)[::-1]
            noises = [
                jax.tree_map(
                    lambda param:
                    jax.random.normal(jax.random.PRNGKey(np.random.randint(0, 50)), param.shape) * mutation_step_size,
                    self.params
                )
                for _ in range(int(population / 2))
            ]
            dominants = [
                deepcopy(genotypes[int(idx)])
                for idx in ordered_genotypes_idx[:int(population / 2)]
            ]
            variants = [
                geno.update(noise)
                for (geno, noise) in zip(dominants, noises)
            ]
            genotypes = [
                *[genotypes[idx] for idx in ordered_genotypes_idx[:int(population / 2)]],
                *variants
            ]

            max_fit = np.max(fitnesses)
            if max_fit > best_fit:
                best_fit = max_fit
                best = genotypes[np.argmax(fitnesses)].params
                self.params = best
                self.save(model_dir + f"_{int(max_fit)}")

                print(f"generation {gen}: best {np.max(fitnesses)} average {np.mean(fitnesses)}")

            if verbose:
                print(f"Generation {gen}")

    def load(self, filepath: str):
        with open(filepath, 'rb') as file:
            self.params = pickle.load(file)
        return self

    def save(self, filepath: str):
        with open(filepath, 'wb') as file:
            pickle.dump(self.params, file)
        return self

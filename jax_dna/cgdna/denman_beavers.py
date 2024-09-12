from tqdm import tqdm
import pdb
import numpy as onp
import unittest

import jax
import jax.numpy as jnp
from jax import random
from jax import jit, grad

# from jax.config import config
# config.update("jax_enable_x64", True)
jax.config.update("jax_enable_x64", True)


def get_denman_beavers(n: int, num_iters: int):
    @jit
    def denman_beavers(a: jnp.ndarray):

        @jit
        def iter_fn(carry, num_iter):
            yk, zk = carry

            yk_inv = jnp.linalg.inv(yk)
            zk_inv = jnp.linalg.inv(zk)

            yk = 1/2 * (yk + zk_inv)
            zk = 1/2 * (zk + yk_inv)

            return (yk, zk), None

        y0 = a
        z0 = jnp.eye(n)
        (yn, zn), _ = jax.lax.scan(iter_fn, (y0, z0), jnp.arange(num_iters))
        return yn
    return denman_beavers



class TestDenmanBeavers(unittest.TestCase):
    def rand_test_n(self, n, n_samples, coeff, key, tol_places=4):
        sample_keys = random.split(key, n_samples)
        denman_beavers = get_denman_beavers(n, num_iters=25)
        for s_key in tqdm(sample_keys, desc="Sample"):

            rand_sqrt = coeff * jax.random.uniform(key, shape=(n, n))
            rand_sqr = rand_sqrt @ rand_sqrt
            calc_sqrt = denman_beavers(rand_sqr)

            err = (calc_sqrt @ calc_sqrt) - rand_sqr
            err = jnp.abs(err)

            n_elems = n*n
            avg_error = err.sum() / n_elems
            max_error = err.max()
            min_error = err.min()

            self.assertAlmostEqual(max_error, 0, places=tol_places)

    def test_grad(self):
        key = random.PRNGKey(0)
        n = 10
        coeff = 10
        denman_beavers = get_denman_beavers(n, num_iters=25)
        sum_entries_fn = lambda arr: denman_beavers(arr).sum()

        rand_sqrt = coeff * jax.random.uniform(key, shape=(n, n))
        rand_sqr = rand_sqrt @ rand_sqrt
        grad_sum_entries = grad(sum_entries_fn)(rand_sqr)
        self.assertNotEqual(grad_sum_entries.sum(), 0)

    def test_to_100(self):

        key = random.PRNGKey(0)

        ns = onp.arange(10, 101, 10)
        num_ns = len(ns)
        coeff = 10

        n_keys = random.split(key, num_ns)

        num_samples_per_n = 100

        for n_idx in tqdm(range(num_ns), desc="Dim"):
            n = ns[n_idx]
            n_key = n_keys[n_idx]
            self.rand_test_n(n, num_samples_per_n, coeff, n_key)



if __name__ == "__main__":
    unittest.main()

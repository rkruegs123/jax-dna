"""Tests for the base observable and utility functions."""

import jax.numpy as jnp
import pytest

import jax_dna.observables.base as jd_obs_base


def test_local_helical_axis():
    """Tests the local helical axis computation."""
    # Define the base sites
    base_sites = [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
    ]
    base_sites = jnp.array(base_sites)

    # Define the quartet
    quartet = jnp.array([[0, 1], [2, 3]])

    # Define the displacement function
    def displacement_fn(a, b):
        return b - a

    # Compute the local helical axis
    lha = jd_obs_base.local_helical_axis(quartet, base_sites, displacement_fn)

    # Check the result
    expected_lha = jnp.array([0.0, -1.0, 0.0])
    assert jnp.allclose(lha, expected_lha)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (3, jnp.array([[[0, 5], [1, 4]], [[1, 4], [2, 3]]])),
        (4, jnp.array([[[0, 7], [1, 6]], [[1, 6], [2, 5]], [[2, 5], [3, 4]]])),
    ],
)
def test_get_duplex_quartets(n: int, expected: jnp.ndarray):
    """Tests the duplex quartet computation."""

    duplex_quartets = jd_obs_base.get_duplex_quartets(n)
    assert jnp.allclose(duplex_quartets, expected)


if __name__ == "__main__":
    test_get_duplex_quartets()

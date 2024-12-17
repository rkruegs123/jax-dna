"""Tests for the propeller observable."""

import jax.numpy as jnp
import jax_dna.observables.propeller as p

test_normals = jnp.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
)


def test_single_propeller_twist_rad(a: int, b: int):
    """Test the single_propeller_twist_rad function."""
    bp = jnp.array([a, b])

    # this assumes the calculation from the original implementation is correct
    # need to confirm with rkreug
    expected = jnp.arccos(jnp.dot(test_normals[a], test_normals[b]))
    assert p.single_propeller_twist_rad(bp, test_normals) == expected


if __name__ == "__main__":
    test_single_propeller_twist_rad(0, 1)

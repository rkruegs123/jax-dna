"""Tests for the rise observable."""

import chex
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.observables.base as b
import jax_dna.observables.rise as r
import jax_dna.simulators.io as jd_sio


# the test values assume that the original implementation is correct
@pytest.mark.parametrize(
    ("quartet", "base_sites", "expected"),
    [
        (
            jnp.array([[0, 1], [1, 2]]),
            jnp.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]),
            14.753608,
        ),
    ],
)
def test_single_rise(
    quartet: jnp.ndarray,
    base_sites: jnp.ndarray,
    expected: jnp.ndarray,
) -> None:
    """Test for the single_rise function."""
    displacement_fn = lambda x, y: y - x
    result = r.single_rise(quartet, base_sites, displacement_fn)
    np.testing.assert_allclose(result, expected)


def test_rise_post_init_raises() -> None:
    """Test that the rise post init raises an error."""
    with pytest.raises(ValueError, match=b.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED):
        r.Rise(rigid_body_transform_fn=None, quartets=jnp.array([[0, 1], [1, 2]]), displacement_fn=lambda x, y: y - x)


# the test values assume that the original implementation is correct
@pytest.mark.parametrize(
    ("quartets", "expected"),
    [
        (
            jnp.array([[[0, 1], [1, 2]], [[1, 2], [2, 3]]]),
            jnp.array([11.065206, 11.065206, 11.065206, 11.065206, 11.065206]),
        ),
    ],
)
def test_rise_call(
    quartets: jnp.ndarray,
    expected: jnp.ndarray,
) -> None:
    """Test rise call."""

    @chex.dataclass
    class MockNucleotide:
        base_sites: jnp.ndarray

        @staticmethod
        def from_rigid_body(rigid_body: jax_md.rigid_body.RigidBody) -> "MockNucleotide":
            return MockNucleotide(base_sites=rigid_body.center)

    def mock_rbt(x: jax_md.rigid_body.RigidBody) -> jnp.ndarray:
        return MockNucleotide.from_rigid_body(x)

    t = 5
    centers = jnp.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]] * t)
    orientations = jnp.ones([t, 4, 4])
    trajectory = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=centers,
            orientation=orientations,
        )
    )

    actual = r.Rise(
        rigid_body_transform_fn=mock_rbt,
        quartets=quartets,
        displacement_fn=lambda x, y: y - x,
    )(trajectory)

    np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    test_rise_call()

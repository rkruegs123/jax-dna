"""Test for diameter observable."""

from collections.abc import Callable

import chex
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.observables.base as b
import jax_dna.observables.diameter as d
import jax_dna.simulators.io as jd_sio


@pytest.mark.parametrize(
    ("bp", "back_sites", "displacement_fn", "sigma_backbone", "expected"),
    [
        (
            jnp.array([[0, 1], [1, 2]]),
            jnp.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            lambda x, y: y - x,
            1.0,
            jnp.array([23.271608, 23.271608]),
        ),
    ],
)
def test_single_diameter(
    bp: jnp.ndarray,
    back_sites: jnp.ndarray,
    displacement_fn: Callable,
    sigma_backbone: float,
    expected: jnp.ndarray,
) -> None:
    result = d.single_diameter(bp, back_sites, displacement_fn, sigma_backbone)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    ("rigid_body_transform_fn", "displacement_fn", "expected_error"),
    [
        (None, lambda x, y: y - x, b.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED),
        (lambda x: x, None, d.ERR_DISPLACEMENT_FN_REQUIRED),
    ],
)
def test_diameter_post_init_raises(
    rigid_body_transform_fn: Callable,
    displacement_fn: Callable,
    expected_error: str,
) -> None:
    """Test that the diameter post init raises an error."""
    h_bonded_base_pairs = jnp.array([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match=expected_error):
        d.Diameter(
            rigid_body_transform_fn=rigid_body_transform_fn,
            h_bonded_base_pairs=h_bonded_base_pairs,
            displacement_fn=displacement_fn,
        )


@pytest.mark.parametrize(
    ("h_bonded_base_pairs", "expected"),
    [
        (jnp.array([[0, 1], [1, 2]]), jnp.array([23.271608, 23.271608, 23.271608, 23.271608, 23.271608])),
    ],
)
def test_diameter_call(
    h_bonded_base_pairs: jnp.ndarray,
    expected: jnp.ndarray,
):
    """Test the call method of the diameter."""

    @chex.dataclass
    class MockNucleotide:
        back_sites: jnp.ndarray

        @staticmethod
        def from_rigid_body(rigid_body: jax_md.rigid_body.RigidBody) -> "MockNucleotide":
            return MockNucleotide(back_sites=rigid_body.center)

    def mock_rbt(x: jax_md.rigid_body.RigidBody) -> jnp.ndarray:
        return MockNucleotide.from_rigid_body(x)

    h_bonded_base_pairs = jnp.array([[0, 1], [1, 2]])
    displacement_fn = lambda x, y: y - x
    sigma_backbone = 1.0
    diameter = d.Diameter(
        rigid_body_transform_fn=mock_rbt,
        h_bonded_base_pairs=h_bonded_base_pairs,
        displacement_fn=displacement_fn,
    )

    t = 5
    centers = jnp.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2]]] * t)
    orientations = jnp.ones([t, 4, 4])
    trajectory = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=centers,
            orientation=orientations,
        )
    )

    actual = diameter(trajectory, sigma_backbone)

    np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    test_diameter_call()

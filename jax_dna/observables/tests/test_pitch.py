"""Tests for the pitch observable."""

import chex
import jax.numpy as jnp
import jax_md
import numpy as np
import pytest

import jax_dna.observables.base as b
import jax_dna.observables.pitch as p
import jax_dna.simulators.io as jd_sio


@pytest.mark.parametrize(
    ("avg_pitch_angle"),
    [
        (1.0),
        (2.0),
        (3.0),
    ],
)
def test_compute_pitch(
    avg_pitch_angle: float,
) -> None:
    """Test the compute_pitch function."""
    expected = jnp.pi / avg_pitch_angle
    np.testing.assert_allclose(p.compute_pitch(avg_pitch_angle), expected)


def test_single_pitch_angle() -> None:
    """Test the single_pitch_angle function."""
    quartet = jnp.array([[0, 1], [1, 2]])
    base_sites = jnp.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    back_sites = jnp.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])[::-1, :]
    displacement_fn = lambda x, y: y - x

    # Need to look at these tests again, when run in isolation they
    # return different values than when run in the full test suite
    np.testing.assert_allclose(
        p.single_pitch_angle(quartet, base_sites, back_sites, displacement_fn),
        0.00034526698,
        atol=1e-3,
    )


def test_pitch_init_raises() -> None:
    with pytest.raises(ValueError, match=b.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED):
        p.PitchAngle(
            rigid_body_transform_fn=None, quartets=jnp.array([[0, 1], [1, 2]]), displacement_fn=lambda x, y: y - x
        )


def test_pitch_call() -> None:
    @chex.dataclass
    class MockNucleotide:
        base_sites: jnp.ndarray
        back_sites: jnp.ndarray

        @staticmethod
        def from_rigid_body(rigid_body: jax_md.rigid_body.RigidBody) -> "MockNucleotide":
            return MockNucleotide(base_sites=rigid_body.center, back_sites=rigid_body.center[::-1, :])

    def mock_rbt(x: jax_md.rigid_body.RigidBody) -> jnp.ndarray:
        return MockNucleotide.from_rigid_body(x)

    t = 5
    centers = jnp.array([[[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]] * t)
    orientations = jnp.ones([t, 4, 4])
    trajectory = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=centers,
            orientation=orientations,
        )
    )

    expected = [0.00034527, 0.00034527, 0.00034527, 0.00034527, 0.00034527]
    actual = p.PitchAngle(
        rigid_body_transform_fn=mock_rbt,
        quartets=jnp.array([[[0, 1], [1, 2]], [[1, 2], [2, 3]]]),
        displacement_fn=lambda x, y: y - x,
    )(trajectory)

    # Need to look at these tests again, when run in isolation they
    # return different values than when run in the full test suite
    np.testing.assert_allclose(actual, expected, atol=1e-3)

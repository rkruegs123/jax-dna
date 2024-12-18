"""Tests for the propeller observable."""

import chex
import jax.numpy as jnp
import jax_dna.observables.base as jd_obs
import jax_dna.observables.propeller as p
import jax_dna.simulators.io as jd_sio
import jax_md
import numpy as np
import pytest

TEST_NORMALS = jnp.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
)


@pytest.mark.parametrize(("a", "b"), [(0, 1), (1, 2), (2, 3)])
def test_single_propeller_twist_rad(a: int, b: int):
    """Test the single_propeller_twist_rad function."""
    bp = jnp.array([a, b])

    # this assumes the calculation from the original implementation is correct
    # need to confirm with rkreug
    expected = jnp.arccos(jnp.dot(TEST_NORMALS[a], TEST_NORMALS[b]))
    assert p.single_propeller_twist_rad(bp, TEST_NORMALS) == expected


def test_propeller_twist_init_raises():
    """Test that the propeller twist init raises an error."""
    with pytest.raises(ValueError, match=jd_obs.ERR_RIGID_BODY_TRANSFORM_FN_REQUIRED):
        p.PropellerTwist(rigid_body_transform_fn=None, h_bonded_base_pairs=jnp.array([0, 1]))


# Need to follow up with ryan k that this is correct
@pytest.mark.parametrize(
    ("pairs", "expected"),
    [
        (
            [
                [0, 1],
                [0, 2],
                [0, 3],
            ],
            jnp.array([120, 120, 120, 120, 120]),
        ),
    ],
)
def test_propeller_twist_call(pairs: list[tuple[int, int]], expected: jnp.ndarray):
    """Test the call method of the propeller twist."""

    @chex.dataclass
    class MockNucleotide:
        base_normals: jnp.ndarray

        @staticmethod
        def from_rigid_body(rigid_body: jax_md.rigid_body.RigidBody) -> "MockNucleotide":
            return MockNucleotide(base_normals=rigid_body.center)

    def mock_rbt(x: jax_md.rigid_body.RigidBody) -> jnp.ndarray:
        return MockNucleotide.from_rigid_body(x)

    ptwist = p.PropellerTwist(rigid_body_transform_fn=mock_rbt, h_bonded_base_pairs=jnp.array(pairs))

    t = 5
    trajectory = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.array([TEST_NORMALS] * t), orientation=jax_md.rigid_body.Quaternion(vec=jnp.ones([t, 4, 4]))
        )
    )

    actual = ptwist(trajectory)
    np.testing.assert_allclose(actual, expected)

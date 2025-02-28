import jax.numpy as jnp
import jax_md
import pytest

import jax_dna.simulators.io as jd_sio


@pytest.mark.parametrize(
    ("n", "key", "expected_n"),
    [
        (10, 5, 1),
        (10, slice(5), 5),
    ],
)
def test_simulatortrajectory_slice(
    n: int,
    key: int | slice,
    expected_n: int,
) -> None:
    """Test the slice method of the SimulatorTrajectory class."""
    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.zeros((n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        )
    )

    sliced_traj = traj.slice(key)

    assert sliced_traj.rigid_body.center.shape == (expected_n, 3)
    assert sliced_traj.rigid_body.orientation.vec.shape == (expected_n, 4)


@pytest.mark.parametrize(
    ("n"),
    [(10), (1)],
)
def test_simulatortrajectory_length(
    n: int,
) -> None:
    """Test the length method of the SimulatorTrajectory class."""

    traj = jd_sio.SimulatorTrajectory(
        rigid_body=jax_md.rigid_body.RigidBody(
            center=jnp.zeros((n, 3)),
            orientation=jax_md.rigid_body.Quaternion(
                vec=jnp.zeros((n, 4)),
            ),
        )
    )

    assert traj.length() == n

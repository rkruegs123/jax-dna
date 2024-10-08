"""Tests for the observable wrappers."""

import jax.numpy as jnp
import jax_dna.losses.observable_wrappers as obs_wrappers
import numpy as np
import pytest


def test_LossFn_init_raises():  # noqa: N802  -- special function name
    """Test that the LossFn class raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        obs_wrappers.LossFn()(None, None, None)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (jnp.arange(5), jnp.ones(5), (jnp.arange(5) - jnp.ones(5)) ** 2),
        (jnp.array(2), jnp.array(1), jnp.array(1)),
        (jnp.array(0), jnp.array(0), jnp.array(0)),
    ],
)
def test_SquaredError(x: jnp.ndarray, y: jnp.ndarray, expected: jnp.ndarray):  # noqa: N802  -- special function name
    """Test the SquaredError class."""
    np.testing.assert_array_equal(obs_wrappers.SquaredError()(x, y), expected)


class MockLossFunction:
    """Mock loss function for testing."""

    def __call__(self, actual, target):
        return actual - target


class MockObservable:
    """Mock observable for testing."""

    def __call__(self, trajectory: jnp.ndarray):
        return jnp.sum(trajectory, axis=1)


@pytest.mark.parametrize(
    "return_obs",
    [
        True,
        False,
    ],
)
def test_ObservableLossFn(return_obs: bool):  # noqa: N802,FBT001  -- special function name, ignore fixture
    """Test the ObservableLossFn class."""

    traj = jnp.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
        ]
    )
    weights = jnp.array([1, 1, 1]) / 3
    expected_obs = (traj.sum(axis=1) * weights).sum()
    target = jnp.array(1)

    obs = obs_wrappers.ObservableLossFn(
        observable=MockObservable(),
        loss_fn=MockLossFunction(),
        return_observable=return_obs,
    )(
        trajectory=traj,
        target=target,
        weights=weights,
    )

    if return_obs:
        np.testing.assert_array_equal(obs[0], expected_obs - target)
        np.testing.assert_array_equal(obs[1], expected_obs)
    else:
        np.testing.assert_array_equal(obs, expected_obs - target)

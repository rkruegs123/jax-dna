"""Tests for the jax_md.utils module."""

import jax.numpy as jnp
import numpy as np
import pytest

import jax_dna.simulators.jax_md.utils as jdna_utils


def test_NoNeighborList_init():  # noqa: N802 - class name
    """Test the NoNeighborList __init__ method."""
    nn = jdna_utils.NoNeighborList(
        unbonded_nbrs=np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        )
    )
    np.testing.assert_equal(
        nn.idx,
        np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ),
    )

    assert nn.allocate(None) == nn
    assert nn.update(None) == nn


def test_StaticSimulatorParams_init():  # noqa: N802 - class name
    """Test the StaticSimulatorParams __init__ method."""
    init_values = {
        "seq": 123,  # this is normally an array
        "mass": 456,  # this is normally an rigid_body
        "gamma": 789,  # this is normally an rigid_body
        "bonded_neighbors": 101112,  # this is normally an array
        "checkpoint_every": 5,
        "dt": 0.1,
        "kT": 296.0,
    }

    ssp = jdna_utils.StaticSimulatorParams(**init_values)

    def check_keys(ssp, attr_name, attr_keys):
        assert list(getattr(ssp, attr_name).keys()) == attr_keys
        for key in attr_keys:
            assert getattr(ssp, attr_name)[key] == init_values[key]

    check_keys(ssp, "sim_init_fn", ["dt", "kT", "gamma"])
    check_keys(ssp, "init_fn", ["seq", "mass", "bonded_neighbors"])
    check_keys(ssp, "step_fn", ["seq", "bonded_neighbors"])


def test_split_and_stack():
    """Test the split_and_stack function."""
    x = jnp.arange(12)
    n = 4
    expected = jnp.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
        ]
    )
    actual = jdna_utils.split_and_stack(x, n)
    np.testing.assert_array_equal(actual, expected)

    stacked_x = [x, x, x]
    actual = jdna_utils.split_and_stack(stacked_x, 4)
    assert len(actual) == len(stacked_x)
    for y in actual:
        np.testing.assert_array_equal(y, expected)


def test_flatten_n():
    """Test the flatten_n function."""
    x = jnp.arange(64).reshape(2, 4, 4, 2)

    for n in range(2):
        with pytest.raises(AssertionError, match="The argument must be positive"):
            actual = jdna_utils.flatten_n(x, n)

    expected_shapes = [
        (8, 4, 2),  # the first 2 dimensions are flattened
        (32, 2),  # the first 3 dimensions are flattened
        (64,),  # all dimensions are flattened
    ]

    for n, es in zip(range(2, 5), expected_shapes, strict=True):
        actual = jdna_utils.flatten_n(x, n)
        assert actual.shape == es


if __name__ == "__main__":
    test_flatten_n()

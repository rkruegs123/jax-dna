"""Tests for the jax_md.utils module."""

import jax_dna.simulators.jax_md.utils as jdna_utils
import numpy as np


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
    pass

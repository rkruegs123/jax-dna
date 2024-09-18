"""Tests for the jax_md.utils module."""

import numpy as np
import pytest

import jax_dna.simulators.jax_md.utils as jdna_utils

def test_NoNeighborList_init():
    """Test the NoNeighborList __init__ method."""
    nn = jdna_utils.NoNeighborList(
        unbonded_nbrs=np.array([
            [1, 2, 3],
            [4, 5, 6],
        ])
    )
    assert np.testing.assert_equal(nn.idx, np.array([
        [1, 2, 3],
        [4, 5, 6],
    ]))

    assert nn.allocate(None) == nn
    assert nn.update(None) == nn


def test_StaticSimulatorParams_init():
    pass

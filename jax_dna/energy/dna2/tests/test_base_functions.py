"""Tests for the base functions of the DNA2 energy model."""

import numpy as np
import pytest

import jax_dna.energy.dna2.base_functions as bf
import jax_dna.energy.dna2.tests.symbolic_base_functions as sp
import jax_dna.utils.types as typ


@pytest.mark.parametrize(
    ("theta", "a", "b"),
    [
        # Case 1
        (0.7, 0.5, 0.5),
        # Case 2 - should be 0
        (0.2, 0.5, 1.0),
    ],
)
def test_f6(theta: typ.ARR_OR_SCALAR, a: typ.Scalar, b: typ.Scalar) -> None:
    """Test the f6 base function."""
    actual = bf.f6(theta, a, b)
    expected = sp.f6(theta, a, b)
    np.testing.assert_allclose(actual, expected)

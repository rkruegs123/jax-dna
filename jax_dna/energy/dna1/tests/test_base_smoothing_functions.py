"""tests for base_smoothing_functions.py"""


import numpy as np
import pytest

import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.tests.symbolic_base_smoothing_functions as sbsf


@pytest.mark.parametrize(
    "x, a, x0, xc",
    [
        (0.5, 0.1, 0.2, 0.3),
        (0.1, 0.2, 0.3, 0.4),
        (0.3, 0.4, 0.5, 0.6),
        (0.4, 0.5, 0.6, 0.7),
    ],
)
def test_solve_f1_b(
    x: float,
    a: float,
    x0: float,
    xc: float
) -> None:

    actual = bsf._solve_f1_b(x, a, x0, xc)
    expected = sbsf._solve_f1_b(x, a, x0, xc)
    np.testing.assert_allclose(actual, expected)
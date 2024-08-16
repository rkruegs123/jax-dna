"""tests for jax_dna.energy.dna1.base_functions"""

import numpy as np
import pytest

import jax_dna.energy.dna1.base_functions as bf
import jax_dna.energy.dna1.tests.symbolic_base_functions as sp
import jax_dna.utils.types as typ


@pytest.mark.parametrize(
    ("r", "r_low", "r_high", "r_c_low", "r_c_high", "eps", "a", "r0", "r_c", "b_low", "b_high"),
    [
        # Case 1
        (0.7, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        # Case 2
        (0.2, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        # Case 3
        (1.2, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        # Case 4
        (1.7, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ],
)
def test_f1(
    r: typ.ARR_OR_SCALAR,
    r_low: typ.Scalar,
    r_high: typ.Scalar,
    r_c_low: typ.Scalar,
    r_c_high: typ.Scalar,
    eps: typ.Scalar,
    a: typ.Scalar,
    r0: typ.Scalar,
    r_c: typ.Scalar,
    b_low: typ.Scalar,
    b_high: typ.Scalar,
) -> None:
    """Test the f1 base function.

    This function as described has 4 cases:
    1. r_low < r < r_high
    2. r_c_low < r < r_low
    3. r_high < r < r_c_high
    4. Otherwise
    """
    actual = bf.f1(r, r_low, r_high, r_c_low, r_c_high, eps, a, r0, r_c, b_low, b_high)
    expected = sp.f1(r, r_low, r_high, r_c_low, r_c_high, eps, a, r0, r_c, b_low, b_high)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r", "r_low", "r_high", "r_c_low", "r_c_high", "k", "r0", "r_c", "b_low", "b_high"),
        [
            # Case 1
            (0.7, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0),
            # Case 2
            (0.2, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0),
            # Case 3
            (1.2, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0),
            # Case 4
            (1.7, 0.5, 1.0, 0.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
)
def test_f2(
    r: float,
    r_low: float,
    r_high: float,
    r_c_low: float,
    r_c_high: float,
    k: float,
    r0: float,
    r_c: float,
    b_low: float,
    b_high: float,
):
    """Test the f2 base function.

    This function as described has 4 cases:
    1. r_low < r < r_high
    2. r_c_low < r < r_low
    3. r_high < r < r_c_high
    4. Otherwise
    """

    actual = bf.f2(r, r_low, r_high, r_c_low, r_c_high, k, r0, r_c, b_low, b_high)
    expected = sp.f2(r, r_low, r_high, r_c_low, r_c_high, k, r0, r_c, b_low, b_high)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r", "r_star", "r_c", "eps", "sigma", "b"),
    [
        (0.3, 0.5, 1.0, 1.0, 1.0, 1.0),
        (0.7, 0.5, 1.0, 1.0, 1.0, 1.0),
        (1.2, 0.5, 1.0, 1.0, 1.0, 1.0),
    ],
)
def test_f3(
    r: float,
    r_star: float,
    r_c: float,
    eps: float,
    sigma: float,
    b: float,
) -> None:
    """Test the f3 base function.

    This function as described has 3 cases:
    1. r < r_star
    2. r_star < r < r_c
    3. Otherwise
    """
    actual = bf.f3(r, r_star, r_c, eps, sigma, b)
    expected = sp.f3(r, r_star, r_c, eps, sigma, b)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
        ("theta", "theta0", "delta_theta_star", "delta_theta_c", "a", "b"),
        [

            ( 0.3,  0.5, 1.0, 1.0, 1.0, 1.0),
            ( 5.7,  5.0, 0.0, 1.0, 1.0, 1.0),
            ( 10.2, 10.0, 0.0, 0.0, 1.0, 1.0),
            (100.0, 1.0,  1.0, 1.0, 1.0, 1.0),
        ],
)
def test_f4(
    theta: float,
    theta0: float,
    delta_theta_star: float,
    delta_theta_c: float,
    a: float,
    b: float,
) -> float:
    """The angular modulation factor used in stacking, hydrogen-bonding, cross-stacking and coaxial stacking.

    This is based on equation 2.10 from the oxDNA paper.

    This function has 4 cases:
    1. theta0 - delta_theta_star < theta < theta0 + delta_theta_star
    2  theta0 - delta_theta_c < theta < theta0 - delta_theta_star
    3. theta0 + delta_theta_star < theta < theta0 + delta_theta_c
    4. Otherwise
    """
    actual = bf.f4(theta, theta0, delta_theta_star, delta_theta_c, a, b)
    expected = sp.f4(theta, theta0, delta_theta_star, delta_theta_c, a, b)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "x_star", "x_c", "a", "b"),
    [
        ( 0.3,  1.0,  1.0, 1.0, 1.0),
        (-0.7, -1.5,  1.0, 1.0, 1.0),
        (-2.0, -0.5, -3.0, 1.0, 1.0),
        (-10.0,  1.0,  1.0, 1.0, 1.0),
    ],
)
def test_f5(
    x: float,
    x_star: float,
    x_c: float,
    a: float,
    b: float,
):
    actual = bf.f5(x, x_star, x_c, a, b)
    expected = sp.f5(x, x_star, x_c, a, b)
    np.testing.assert_allclose(actual, expected)

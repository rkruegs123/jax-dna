"""tests for base_smoothing_functions.py"""

import numpy as np
import pytest

import jax_dna.energy.dna1.base_smoothing_functions as bsf
import jax_dna.energy.dna1.tests.symbolic_base_smoothing_functions as sbsf


@pytest.mark.parametrize(
    ("x", "a", "x0", "xc"),
    [
        (0.5, 0.1, 0.2, 0.3),
        (0.1, 0.2, 0.3, 0.4),
        (0.3, 0.4, 0.5, 0.6),
        (0.4, 0.5, 0.6, 0.7),
    ],
)
def test_solve_f1_b(x: float, a: float, x0: float, xc: float) -> None:
    """Test the _solve_f1_b function."""
    actual = bsf._solve_f1_b(x, a, x0, xc)
    expected = sbsf._solve_f1_b(x, a, x0, xc)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "a", "x0", "xc"),
    [
        (0.5, 0.1, 0.2, 0.3),
        (0.1, 0.2, 0.3, 0.4),
        (0.3, 0.4, 0.5, 0.6),
        (0.4, 0.5, 0.6, 0.7),
    ],
)
def test_solve_f1_xc_star(x: float, a: float, x0: float, xc: float) -> None:
    """Test the _solve_f1_xc_star function."""
    actual = bsf._solve_f1_xc_star(x, a, x0, xc)
    expected = sbsf._solve_f1_xc_star(x, a, x0, xc)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x0", "a", "xc", "x_low", "x_high"),
    [
        (0.1, 0.2, 0.3, 0.4, 0.5),
        (0.2, 0.3, 0.4, 0.5, 0.6),
        (0.3, 0.4, 0.5, 0.6, 0.7),
        (0.4, 0.5, 0.6, 0.7, 0.8),
    ],
)
def test_get_f1_smoothing_params(
    x0: float,
    a: float,
    xc: float,
    x_low: float,
    x_high: float,
) -> None:
    """Test the get_f1_smoothing_params function."""
    actual = bsf.get_f1_smoothing_params(x0, a, xc, x_low, x_high)
    expected = sbsf.get_f1_smoothing_params(x0, a, xc, x_low, x_high)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "x0", "xc"),
    [
        (0.5, 0.1, 1.2),
        (0.1, 0.2, 1.3),
        (0.3, 0.4, 1.5),
        (0.4, 0.5, 1.6),
    ],
)
def test_solve_f2_b(x: float, x0: float, xc: float) -> None:
    """Test the _solve_f2_b function."""
    actual = bsf._solve_f2_b(x, x0, xc)
    expected = sbsf._solve_f2_b(x, x0, xc)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "x0", "xc"),
    [
        (0.5, 0.1, 1.2),
        (0.1, 0.2, 1.3),
        (0.3, 0.4, 1.5),
        (0.4, 0.5, 1.6),
    ],
)
def test_solve_f2_xc_star(x: float, x0: float, xc: float) -> None:
    """Test the _solve_f2_xc_star function."""
    actual = bsf._solve_f2_xc_star(x, x0, xc)
    expected = sbsf._solve_f2_xc_star(x, x0, xc)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x0", "xc", "x_low", "x_high"),
    [
        (0.1, 1.2, 0.3, 0.4),
        (0.2, 1.3, 0.4, 0.5),
        (0.3, 1.5, 0.5, 0.6),
        (0.4, 1.6, 0.6, 0.7),
    ],
)
def test_get_f2_smoothing_params(
    x0: float,
    xc: float,
    x_low: float,
    x_high: float,
) -> None:
    """Test the get_f2_smoothing_params function."""
    actual = bsf.get_f2_smoothing_params(x0, xc, x_low, x_high)
    expected = sbsf.get_f2_smoothing_params(x0, xc, x_low, x_high)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "sigma"),
    [
        (0.5, 1.2),
        (0.1, 1.3),
        (0.3, 1.5),
        (0.4, 1.6),
    ],
)
def test_solve_f3_b(x: float, sigma: float) -> None:
    """Test the _solve_f3_b function."""
    actual = bsf._solve_f3_b(x, sigma)
    expected = sbsf._solve_f3_b(x, sigma)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "sigma"),
    [
        (0.5, 1.2),
        (0.1, 1.3),
        (0.3, 1.5),
        (0.4, 1.6),
    ],
)
def test_solve_f3_xc(x: float, sigma: float) -> None:
    """Test the _solve_f3_xc function."""
    actual = bsf._solve_f3_xc(x, sigma)
    expected = sbsf._solve_f3_xc(x, sigma)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("r_star", "sigma"),
    [
        (0.5, 1.2),
        (0.1, 1.3),
        (0.3, 1.5),
        (0.4, 1.6),
    ],
)
def test_get_f3_smoothing_params(r_star: float, sigma: float) -> None:
    """Test the get_f3_smoothing_params function."""
    actual = bsf.get_f3_smoothing_params(r_star, sigma)
    expected = sbsf.get_f3_smoothing_params(r_star, sigma)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "x0", "a"),
    [
        (0.5, 0.1, 0.2),
        (0.1, 0.2, 0.3),
        (0.3, 0.4, 0.5),
        (0.4, 0.5, 0.6),
    ],
)
def test_solve_f4_b(x: float, x0: float, a: float) -> None:
    """Test the _solve_f4_b function."""
    actual = bsf._solve_f4_b(x, x0, a)
    expected = sbsf._solve_f4_b(x, x0, a)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "x0", "a"),
    [
        (0.5, 0.1, 0.2),
        (0.1, 0.2, 0.3),
        (0.3, 0.4, 0.5),
        (0.4, 0.5, 0.6),
    ],
)
def test_solve_f4_xc(x: float, x0: float, a: float) -> None:
    """Test the _solve_f4_xc function."""
    actual = bsf._solve_f4_xc(x, x0, a)
    expected = sbsf._solve_f4_xc(x, x0, a)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("a", "x0", "delta_x_star"),
    [
        (0.5, 0.1, 0.2),
        (0.1, 0.2, 0.3),
        (0.3, 0.4, 0.5),
        (0.4, 0.5, 0.6),
    ],
)
def test_get_f4_smoothing_params(a: float, x0: float, delta_x_star: float) -> None:
    """Test the get_f4_smoothing_params function."""
    actual = bsf.get_f4_smoothing_params(a, x0, delta_x_star)
    expected = sbsf.get_f4_smoothing_params(a, x0, delta_x_star)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize(
    ("x", "x0", "a"),
    [
        (0.5, 0.1, 0.2),
        (0.1, 0.2, 0.3),
        (0.3, 0.4, 0.5),
        (0.4, 0.5, 0.6),
    ],
)
def test_solve_f5_b(x: float, x0: float, a: float) -> None:
    """Test the _solve_f5_b function."""
    actual = bsf._solve_f5_b(x, x0, a)
    expected = sbsf._solve_f5_b(x, x0, a)
    np.testing.assert_allclose(actual, expected)

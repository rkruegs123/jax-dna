import jax
import numpy as np
import pytest
import sympy as sp

from jax_dna.energy import potentials

jax.config.update("jax_enable_x64", True)  # noqa: FBT003

# functional forms from oxDNA paper
# https://ora.ox.ac.uk/objects/uuid:b2415bb2-7975-4f59-b5e2-8c022b4a3719/files/mdcac62bc9133143fc05070ed20048c50
# Section 2.4.1


def _sympy_v_fene() -> sp.Expr:
    """String form of Equation 2.1 from the oxDNA paper."""
    return sp.parsing.sympy_parser.parse_expr("-eps / 2 * log(1 - (r-r0)**2 / delt**2)")


@pytest.mark.parametrize(
    ("r", "eps", "r0", "delt"),
    [
        (5, 3, 2, 10),
    ],
)
def test_v_fene(r: float, eps: float, r0: float, delt: float):
    """tests v_fene potential. Function is equation 2.1 form the oxDNA paper."""
    actual = float(potentials.v_fene(r, eps, r0, delt))
    expected = float(_sympy_v_fene().evalf(subs={"r": r, "eps": eps, "r0": r0, "delt": delt}))
    np.testing.assert_allclose(actual, expected)


def _sympy_v_morse() -> sp.Expr:
    """String form of Equation 2.2 from the oxDNA paper."""
    return sp.parsing.sympy_parser.parse_expr("eps * (1 - exp(-(r-r0)*a))**2")


@pytest.mark.parametrize(
    ("r", "eps", "r0", "a"),
    [
        (5, 3, 2, 10),
    ],
)
def test_v_morse(r: float, eps: float, r0: float, a: float):
    """tests v_morse potential. Function is equation 2.2 form the oxDNA paper."""
    actual = float(potentials.v_morse(r, eps, r0, a))
    expected = float(_sympy_v_morse().evalf(subs={"r": r, "eps": eps, "r0": r0, "a": a}))
    np.testing.assert_allclose(actual, expected)


def _sympy_v_harmonic() -> sp.Expr:
    """String form of Equation 2.3 from the oxDNA paper."""
    return sp.parsing.sympy_parser.parse_expr("k / 2 * (r - r0)**2")


@pytest.mark.parametrize(
    ("r", "k", "r0"),
    [
        (5, 3, 2),
    ],
)
def test_v_harmonic(r: float, k: float, r0: float):
    """tests v_harmonic potential. Function is equation 2.3 form the oxDNA paper."""
    actual = float(potentials.v_harmonic(r, k, r0))
    expected = float(_sympy_v_harmonic().evalf(subs={"r": r, "k": k, "r0": r0}))
    np.testing.assert_allclose(actual, expected)


def _sympy_harmonic() -> sp.Expr:
    """String form of Equation 2.3 from the oxDNA paper."""
    return sp.parsing.sympy_parser.parse_expr("k / 2 * (r - r0)**2")


@pytest.mark.parametrize(
    ("r", "k", "r0"),
    [
        (5, 3, 2),
    ],
)
def test_harmonic(r: float, k: float, r0: float):
    """tests harmonic potential. Function is equation 2.3 form the oxDNA paper."""
    actual = float(potentials.v_harmonic(r, k, r0))
    expected = float(_sympy_harmonic().evalf(subs={"r": r, "k": k, "r0": r0}))
    np.testing.assert_allclose(actual, expected)


def _sympy_v_lj() -> sp.Expr:
    """String form of the Lennard-Jones potential."""
    return sp.parsing.sympy_parser.parse_expr("4 * eps * ((sigma / r)**12 - (sigma / r)**6)")


@pytest.mark.parametrize(
    ("r", "eps", "sigma"),
    [
        (5, 3, 2),
    ],
)
def test_v_lj(r: float, eps: float, sigma: float):
    """tests Lennard-Jones potential. Function is equation 2.4 form the oxDNA paper."""
    actual = float(potentials.v_lj(r, eps, sigma))
    expected = float(_sympy_v_lj().evalf(subs={"r": r, "eps": eps, "sigma": sigma}))
    np.testing.assert_allclose(actual, expected)


def _sympy_v_mod() -> sp.Expr:
    """String form of the modified potential."""
    return sp.parsing.sympy_parser.parse_expr("1 - a * (theta - theta0)**2")


@pytest.mark.parametrize(
    ("theta", "a", "theta0"),
    [
        (5, 3, 2),
    ],
)
def test_v_mod(theta: float, a: float, theta0: float):
    """tests modified potential. Function is equation 2.5 form the oxDNA paper."""
    actual = float(potentials.v_mod(theta, a, theta0))
    expected = float(_sympy_v_mod().evalf(subs={"theta": theta, "a": a, "theta0": theta0}))
    np.testing.assert_allclose(actual, expected)


def _sympy_v_smooth() -> sp.Expr:
    """String form of the smooth potential."""
    return sp.parsing.sympy_parser.parse_expr("b * (x_c - x)**2")


@pytest.mark.parametrize(
    ("x", "b", "x_c"),
    [
        (5, 3, 2),
    ],
)
def test_v_smooth(x: float, b: float, x_c: float):
    """tests smooth potential. Function is equation 2.6 form the oxDNA paper."""
    actual = float(potentials.v_smooth(x, b, x_c))
    expected = float(_sympy_v_smooth().evalf(subs={"x": x, "b": b, "x_c": x_c}))
    np.testing.assert_allclose(actual, expected)
